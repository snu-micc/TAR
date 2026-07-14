"""
extract_templates_pistachio2m.py
================================
Extracts LocalTemplate reaction templates (radii r0-r5) for every reaction
in a source CSV using multiprocessing.

Each output row is the source row with six extra columns:
    template_r0, template_r1, template_r2, template_r3, template_r4, template_r5

Results are appended to disk every --save_batch rows (OOM-safe; resumable —
reruns pick up from the last completed row in --out_path).
Failed extractions are stored as empty string '' in the main CSV.
The error CSV stores per-radius failure reasons:
  None:no_product          — products_list empty (no mapped product)
  None:Could not sanitize… — RDKit sanitization failed internally
  None:No changes…         — no atom changes detected between reactants/products
  None:Could not validate… — SMARTS validation failed
  Exception:ErrorType:msg  — unhandled exception from the extractor

Usage:
    python extract_templates_pistachio2m.py \
        --src_path Data/data_val_simple.csv \
        --out_path Data/data_val_template.csv
"""

import os
import sys
import time
import argparse
import multiprocessing as mp

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT       = os.path.dirname(os.path.abspath(__file__))
LOCAL_TMPL = os.path.join(ROOT, 'LocalTemplate')

# ---------------------------------------------------------------------------
# Config (defaults; overridden by CLI args in __main__)
# ---------------------------------------------------------------------------
RADII      = [0, 1, 2, 3, 4, 5]
SMILES_COL = 'SMILES'
N_WORKERS  = max(1, mp.cpu_count() - 2)
CHUNK_SIZE = 200
SAVE_BATCH = 5000

TMPL_COLS = [f'template_r{r}' for r in RADII]

SRC_PATH = None
OUT_PATH = None
ERR_PATH = None


# ---------------------------------------------------------------------------
# Worker (runs in subprocess)
# ---------------------------------------------------------------------------

def _worker_init():
    global _extractors
    sys.path.insert(0, LOCAL_TMPL)
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    from localtemplate.template_extractor import TemplateExtractor
    _extractors = {}
    for r in RADII:
        _extractors[r] = TemplateExtractor(
            simplify_aromatic=False, simplify_halide=False,
            include_LGs=True, map_all_atoms=False, radius=r,
        )


def _diagnose(r: int, smi: str) -> str:
    """
    Step through TemplateExtractor.__call__ manually to pinpoint the failure.

    Returns one of:
      'None:no_product'               — products_list empty after split_reagents
      'None:sanitize_failed:<e>'      — sanitize_mols raised an exception
      'None:no_changed_atoms'         — changed_maps is empty
      'None:invalid_smarts'           — rxn.Validate() failed
      'None:unknown'                  — extractor returned None for unknown reason
      'Exception:<step>:<Type>:<msg>' — unhandled exception at a specific step
    """
    from rdkit.Chem import AllChem
    from localtemplate.template_utils import clean_map_and_sort, sanitize_mols
    from localtemplate.template_utils import extend_atom_maps

    ext = _extractors[r]
    try:
        reactants_list, reagents_list, products_list, product_maps = ext.split_reagents(smi)
    except Exception as e:
        return f'Exception:split_reagents:{type(e).__name__}:{str(e)[:120]}'

    if not products_list:
        return 'None:no_product'

    try:
        products  = clean_map_and_sort(products_list,  product_maps, return_mols=True)
        reactants = clean_map_and_sort(reactants_list, product_maps, return_mols=True)
        if ext.map_all_atoms:
            reactants = extend_atom_maps(reactants, product_maps, neighbor_only=False)
    except Exception as e:
        return f'Exception:clean_map:{type(e).__name__}:{str(e)[:120]}'

    try:
        sanitize_mols(reactants + products)
    except Exception as e:
        return f'None:sanitize_failed:{type(e).__name__}:{str(e)[:120]}'

    try:
        _, changed_maps = ext.get_changed_atoms(reactants, products)
    except Exception as e:
        return f'Exception:get_changed_atoms:{type(e).__name__}:{str(e)[:120]}'

    if not changed_maps:
        return 'None:no_changed_atoms'

    try:
        react_frags, reactants, special_atoms = ext.get_fragments_for_changed_atoms(
            reactants, changed_maps, 'reactant', [])
        prod_frags, _, _ = ext.get_fragments_for_changed_atoms(
            products, changed_maps, 'product', special_atoms)
        canonical_template, _ = ext.canonicalize_transform(react_frags, prod_frags)
        rxn = AllChem.ReactionFromSmarts(canonical_template)
        if rxn.Validate()[1] != 0:
            return f'None:invalid_smarts:{canonical_template[:120]}'
    except Exception as e:
        return f'Exception:template_build:{type(e).__name__}:{str(e)[:120]}'

    return 'None:unknown'


def _worker_extract(args):
    """
    args: (start_idx, smiles_list)
    returns: (start_idx, [{r: tmpl}, ...], [(local_idx, {r: reason}), ...])

    failure reasons stored per radius:
      'None:no_product'               — products_list empty after split_reagents
      'None:sanitize_failed:<e>'      — RDKit sanitization failed
      'None:no_changed_atoms'         — no atom changes detected
      'None:invalid_smarts:<tmpl>'    — SMARTS validation failed
      'Exception:<step>:<Type>:<msg>' — unhandled exception at a specific step
    """
    start_idx, smiles_list = args
    results      = []
    failed_local = []
    for local_idx, smi in enumerate(smiles_list):
        row         = {}
        failed_info = {}
        smi_str = str(smi).strip()
        for r, ext in _extractors.items():
            try:
                tmpl = ext(smi_str, return_template_only=True)
                if tmpl is not None:
                    row[r] = tmpl
                else:
                    row[r] = ''
                    failed_info[r] = _diagnose(r, smi_str)
            except Exception as e:
                row[r] = ''
                failed_info[r] = f'Exception:{type(e).__name__}:{str(e)[:150]}'
        if failed_info:
            failed_local.append((local_idx, failed_info))
        results.append(row)
    return start_idx, results, failed_local


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def _count_done_rows(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, 'r', encoding='utf-8') as f:
        return max(0, sum(1 for _ in f) - 1)


def process() -> None:
    print(f'Source : {SRC_PATH}')
    print(f'Output : {OUT_PATH}')
    print(f'Workers: {N_WORKERS}   chunk={CHUNK_SIZE}   save_batch={SAVE_BATCH}')
    print(f'Radii  : {RADII}')

    df_src = pd.read_csv(SRC_PATH)
    n      = len(df_src)
    print(f'Total rows: {n:,}')

    start_row    = _count_done_rows(OUT_PATH)
    write_header = (start_row == 0)

    if start_row >= n:
        print(f'Already complete ({n:,} rows). Nothing to do.')
        return
    if start_row > 0:
        print(f'Resuming from row {start_row:,} ({start_row / n * 100:.1f}% done).')

    smiles_all      = df_src[SMILES_COL].tolist()
    row_ids_all     = df_src['row_id'].tolist() if 'row_id' in df_src.columns else list(range(n))
    n_batches       = (n - start_row + SAVE_BATCH - 1) // SAVE_BATCH
    t0              = time.time()
    done_rows       = 0
    total_fails     = 0
    err_header_done = os.path.exists(ERR_PATH)

    with mp.Pool(processes=N_WORKERS, initializer=_worker_init) as pool:
        bar_fmt = '{l_bar}{bar}| {n_fmt}/{total_fmt} rxn  [{elapsed}<{remaining}  {rate_fmt}]'
        with tqdm(total=n - start_row, desc='extracting', unit='rxn',
                  bar_format=bar_fmt, dynamic_ncols=True) as pbar:

            for batch_idx, batch_start in enumerate(
                    range(start_row, n, SAVE_BATCH), start=1):
                batch_end = min(batch_start + SAVE_BATCH, n)

                batch_df = df_src.iloc[batch_start:batch_end].copy()
                for col in TMPL_COLS:
                    batch_df[col] = ''

                batch_smiles = smiles_all[batch_start:batch_end]
                chunks = [
                    (batch_start + i, batch_smiles[i: i + CHUNK_SIZE])
                    for i in range(0, len(batch_smiles), CHUNK_SIZE)
                ]

                result_map = {}
                failed_abs = []   # list of (abs_idx, {r: reason})
                for start_idx, rows, failed_local in pool.imap_unordered(
                        _worker_extract, chunks, chunksize=1):
                    for offset, row in enumerate(rows):
                        result_map[start_idx + offset] = row
                    for local_idx, failed_info in failed_local:
                        failed_abs.append((start_idx + local_idx, failed_info))
                    pbar.update(len(rows))
                    done_rows += len(rows)

                for abs_idx, row in result_map.items():
                    local_idx = abs_idx - batch_start
                    for r in RADII:
                        batch_df.iat[local_idx,
                                     batch_df.columns.get_loc(f'template_r{r}')] = row[r]

                batch_df.to_csv(OUT_PATH, mode='a', header=write_header, index=False)
                write_header = False
                del batch_df, result_map

                if failed_abs:
                    err_df = pd.DataFrame({
                        'row_id':       [row_ids_all[i]                              for i, _ in failed_abs],
                        'SMILES':       [smiles_all[i]                               for i, _ in failed_abs],
                        'failed_radii': [','.join(f'r{r}' for r in info)             for _, info in failed_abs],
                        **{f'r{r}_reason': [info.get(r, '')                          for _, info in failed_abs] for r in RADII},
                    })
                    err_df.to_csv(ERR_PATH, mode='a',
                                  header=not err_header_done, index=False)
                    err_header_done = True
                    total_fails += len(failed_abs)
                del failed_abs

                elapsed   = time.time() - t0
                total_done = start_row + done_rows
                pct        = total_done / n * 100
                rate       = done_rows / elapsed if elapsed > 0 else 0
                eta_sec    = (n - total_done) / rate if rate > 0 else 0
                eta_str    = time.strftime('%H:%M:%S', time.gmtime(eta_sec))
                tqdm.write(
                    f'  [batch {batch_idx}/{n_batches}] '
                    f'saved {batch_end:,}/{n:,} ({pct:.1f}%)  '
                    f'failed={total_fails}  '
                    f'speed={rate:.0f} rxn/s  '
                    f'ETA={eta_str}'
                )

    filled        = _count_done_rows(OUT_PATH)
    elapsed_total = time.strftime('%H:%M:%S', time.gmtime(time.time() - t0))
    print(f'\nDone. rows={filled:,}  total_failed={total_fails}  elapsed={elapsed_total}')
    print(f'Saved -> {OUT_PATH}')
    if total_fails:
        print(f'Errors -> {ERR_PATH}')


def summarize() -> None:
    if not os.path.exists(OUT_PATH):
        print('Output file not found.')
        return

    df = pd.read_csv(OUT_PATH, usecols=[f'template_r{r}' for r in RADII])
    for c in df.columns:
        df[c] = df[c].fillna('').astype(str).str.strip()

    has = {r: df[f'template_r{r}'] != '' for r in RADII}

    n = len(df)
    print('\n--- Template Coverage Summary ---')
    print(f'  {"Radius":<8}  {"Exists":>10}  {"Pct":>7}')
    print('  ' + '-' * 30)
    for r in RADII:
        cnt = has[r].sum()
        print(f'  r{r:<7}  {cnt:>10,}  ({cnt / n * 100:.2f}%)')

    all_exist = all(has[r] for r in RADII)  # scalar — use mask logic below
    all_exist_mask = has[RADII[0]]
    for r in RADII[1:]:
        all_exist_mask = all_exist_mask & has[r]
    no_tmpl_mask = ~has[RADII[0]]
    for r in RADII[1:]:
        no_tmpl_mask = no_tmpl_mask & ~has[r]

    print()
    print(f'  {"all exist (r0~r5)":<32} {all_exist_mask.sum():>8,}  ({all_exist_mask.sum() / n * 100:.2f}%)')
    print(f'  {"no template (all failed)":<32} {no_tmpl_mask.sum():>8,}  ({no_tmpl_mask.sum() / n * 100:.2f}%)')
    print(f'  {"Total":<32} {n:>8,}')

    # --- Failure reason breakdown from error CSV ---
    if not os.path.exists(ERR_PATH):
        return

    err_df = pd.read_csv(ERR_PATH, dtype=str).fillna('')
    records = []
    for r in RADII:
        col = f'r{r}_reason'
        if col not in err_df.columns:
            continue
        subset = err_df[err_df[col] != ''][['row_id', col]].copy()
        subset.rename(columns={col: 'reason'}, inplace=True)
        subset['radius'] = f'r{r}'
        records.append(subset)

    if not records:
        return

    breakdown = pd.concat(records, ignore_index=True)
    breakdown['reason_category'] = breakdown['reason'].str.split(':').str[:3].str.join(':')

    sorted_path = ERR_PATH.replace('.csv', '_reasons_sorted.csv')
    breakdown[['row_id', 'radius', 'reason_category', 'reason']].sort_values(
        ['reason_category', 'radius', 'row_id']
    ).to_csv(sorted_path, index=False)

    print(f'\n--- Failure Reason Summary ({len(err_df):,} failed reactions) ---')
    counts = breakdown['reason_category'].value_counts()
    print(f'  {"Count":>8}  reason_category')
    print('  ' + '-' * 55)
    for cat, cnt in counts.items():
        print(f'  {cnt:>8,}  {cat}')
    print(f'\n  Sorted breakdown saved -> {sorted_path}')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Extract LocalTemplate reaction templates (r0-r5) from a reaction SMILES CSV.')
    parser.add_argument('--src_path', required=True,
                        help='Input CSV with a reaction SMILES column (mapped reactants>>products)')
    parser.add_argument('--out_path', required=True,
                        help='Output CSV path (source columns + template_r0..r5)')
    parser.add_argument('--err_path', default=None,
                        help='Error CSV path (default: <out_path>_errors.csv)')
    parser.add_argument('--smiles_col', default=SMILES_COL,
                        help=f'Column in --src_path holding the reaction SMILES (default: {SMILES_COL})')
    parser.add_argument('--n_workers', type=int, default=N_WORKERS)
    parser.add_argument('--chunk_size', type=int, default=CHUNK_SIZE)
    parser.add_argument('--save_batch', type=int, default=SAVE_BATCH)
    return parser.parse_args()


if __name__ == '__main__':
    mp.freeze_support()
    args = parse_args()

    SRC_PATH   = args.src_path
    OUT_PATH   = args.out_path
    ERR_PATH   = args.err_path or OUT_PATH.replace('.csv', '_errors.csv')
    SMILES_COL = args.smiles_col
    N_WORKERS  = args.n_workers
    CHUNK_SIZE = args.chunk_size
    SAVE_BATCH = args.save_batch

    process()
    summarize()
