"""
evaluate.py
===========
Beam search inference (Phase 1) for the Reacon baseline.

Runs the 5 trained GCN slot-classifiers (cat, solv0, solv1, reag0, reag1)
on the test set and enumerates the top-N candidate condition sets.
Saves:  models/inference_results_m{m}_{split}.csv
        columns: _id, reaction, gt, cand1, score1, ..., cand{N}, score{N}
  gt / cand format: "cat;solv0;solv1;reag0;reag1"  (none for missing)
  score: log P(condition) = Σ log p_slot

Template re-ranking (Phase 2, including entropy-adaptive λ) now lives in
tar/rerank_all.py, which consumes the CSV mirrored to
tar/InferenceResults/{split}/reacon.csv by this script.

Usage
-----
  cd models/reacon

  python evaluate.py --split test --m_per_slot 5
  python evaluate.py --split val  --m_per_slot 5
  python evaluate.py --skip_inference   # reuse existing inference CSV, just re-copy
"""

import os
import sys
import math
import argparse
import itertools
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ── Reacon's bundled chemprop (v1 API) ───────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REACON_DIR  = os.path.normpath(os.path.join(SCRIPT_DIR, '.'))
ROOT_DIR    = os.path.normpath(os.path.join(SCRIPT_DIR, '../..'))
if REACON_DIR not in sys.path:
    sys.path.insert(0, REACON_DIR)

from chemprop.args import PredictArgs
from chemprop.train.make_predictions import make_predictions

NULL    = 'none'
ROLES   = ['cat', 'solv0', 'solv1', 'reag0', 'reag1']


# ══════════════════════════════════════════════════════════════════════════════
# Label helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_labels(label_dir):
    cat_list  = pd.read_csv(os.path.join(label_dir, 'cat_labels.csv'))['cat'].astype(str).tolist()
    solv_list = pd.read_csv(os.path.join(label_dir, 'solv_labels.csv'))['solv'].astype(str).tolist()
    reag_list = pd.read_csv(os.path.join(label_dir, 'reag_labels.csv'))['reag'].astype(str).tolist()
    return cat_list, solv_list, reag_list


def norm(s):
    """Normalise a label string; return NULL for missing entries."""
    s = str(s).strip()
    return NULL if s in {'None', 'nan', '', 'none', 'NaN'} else s


def decode_row(row, cat_list, solv_list, reag_list):
    """Decode one GCN_data row → list of 5 normalised SMILES strings."""
    return [
        norm(cat_list[int(row['cat'])]),
        norm(solv_list[int(row['solv0'])]),
        norm(solv_list[int(row['solv1'])]),
        norm(reag_list[int(row['reag0'])]),
        norm(reag_list[int(row['reag1'])]),
    ]


def encode_cand(parts):
    """List of 5 SMILES → compact string 'cat;s0;s1;r0;r1'."""
    return ';'.join(parts)


def decode_cand(s):
    """Compact string → list of 5 SMILES."""
    if not s or str(s).strip() in {'', 'nan'}:
        return [NULL] * 5
    parts = str(s).split(';')
    while len(parts) < 5:
        parts.append(NULL)
    return [p.strip() or NULL for p in parts[:5]]


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1 — Inference
# ══════════════════════════════════════════════════════════════════════════════

def run_gcn(model_dir, smiles, gcn_csv):
    """
    chemprop v1 multiclass inference.
    Returns list[list[float]] — preds[i] = probability vector for sample i.
    """
    preds_tmp = os.path.join(SCRIPT_DIR, '_reacon_tmp_preds.csv')
    pa = PredictArgs().parse_args([
        '--test_path', gcn_csv,
        '--checkpoint_dir', model_dir,
        '--preds_path', preds_tmp,
    ])
    raw = make_predictions(pa, smiles)
    result = []
    for sample in raw:
        result.append(sample[0] if isinstance(sample[0], list) else sample)
    return result


def enumerate_candidates(per_slot, cat_list, solv_list, reag_list,
                         m_per_slot, n_sets):
    """
    Cross-product enumeration of top-m per slot.

    m_per_slot = 5  →  5^5 = 3,125 raw combos  →  dedup  →  top n_sets by log_score.
    Returns list of (cand_str, log_score).
    """
    def top_m(probs, m):
        arr = np.clip(np.array(probs, dtype=np.float64), 1e-12, None)
        idxs = np.argsort(arr)[::-1][:m]
        return [(int(i), float(arr[i])) for i in idxs]

    tops = {
        'cat':   top_m(per_slot['cat'],   m_per_slot),
        'solv0': top_m(per_slot['solv0'], m_per_slot),
        'solv1': top_m(per_slot['solv1'], m_per_slot),
        'reag0': top_m(per_slot['reag0'], m_per_slot),
        'reag1': top_m(per_slot['reag1'], m_per_slot),
    }

    candidates = []
    for (ci,cp),(s0i,s0p),(s1i,s1p),(r0i,r0p),(r1i,r1p) in itertools.product(
            tops['cat'], tops['solv0'], tops['solv1'], tops['reag0'], tops['reag1']):
        log_score = math.log(cp)+math.log(s0p)+math.log(s1p)+math.log(r0p)+math.log(r1p)
        parts = [
            norm(cat_list[ci]),
            norm(solv_list[s0i]), norm(solv_list[s1i]),
            norm(reag_list[r0i]), norm(reag_list[r1i]),
        ]
        candidates.append((log_score, encode_cand(parts)))

    candidates.sort(key=lambda x: x[0], reverse=True)

    seen, unique = set(), []
    for log_score, cs in candidates:
        key = frozenset(v for v in decode_cand(cs) if v != NULL)
        if key not in seen:
            seen.add(key)
            unique.append((cs, log_score))
        if len(unique) >= n_sets:
            break
    return unique   # [(cand_str, log_score), ...]


def run_inference(gcn_csv, model_base, label_dir, m_per_slot, n_sets, out_csv):
    """Phase 1: run all 5 GCN models and save candidates to CSV."""
    cat_list, solv_list, reag_list = load_labels(label_dir)
    gcn_df = pd.read_csv(gcn_csv)
    smiles = [[str(r)] for r in gcn_df['reaction']]
    N = len(gcn_df)

    # Run 5 GCN models
    slot_probs = {}
    for target in ROLES:
        model_dir = os.path.join(model_base, f'GCN_{target}')
        print(f'  Running GCN_{target} ({N:,} samples) ...')
        slot_probs[target] = run_gcn(model_dir, smiles, gcn_csv)

    # Build header
    header = ['_id', 'reaction', 'gt']
    for j in range(1, n_sets + 1):
        header += [f'cand{j}', f'score{j}']

    print(f'\nEnumerating top-{n_sets} candidates (m_per_slot={m_per_slot}) ...')
    first_chunk = True
    CHUNK = 2048

    for chunk_start in tqdm(range(0, N, CHUNK), desc='Writing CSV'):
        chunk_end = min(chunk_start + CHUNK, N)
        records = []
        for i in range(chunk_start, chunk_end):
            row = gcn_df.iloc[i]
            gt_parts = decode_row(row, cat_list, solv_list, reag_list)
            per_slot = {t: slot_probs[t][i] for t in ROLES}
            cands = enumerate_candidates(per_slot, cat_list, solv_list, reag_list,
                                         m_per_slot, n_sets)
            rec = {
                '_id':      str(row.get('_id', i)),
                'reaction': str(row['reaction']),
                'gt':       encode_cand(gt_parts),
            }
            for j, (cs, sc) in enumerate(cands, 1):
                rec[f'cand{j}'] = cs
                rec[f'score{j}'] = round(sc, 6)

            records.append(rec)

        pd.DataFrame(records, columns=header).to_csv(
            out_csv, mode='w' if first_chunk else 'a',
            header=first_chunk, index=False)
        first_chunk = False

    print(f'Saved → {out_csv}')

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Reacon GCN inference (Phase 1 only)')
    parser.add_argument('--reacon_dir', default=REACON_DIR)
    parser.add_argument('--data_dir',   default=None,
                        help='Root data dir (default: ../Data)')
    parser.add_argument('--split',      choices=['test','val'], default='test')
    parser.add_argument('--m_per_slot', type=int, default=5,
                        help='Top-m per slot for cross-product enumeration  '
                             '(m=5 → 5^5=3125 raw combos → ≤50 unique; default: 5)')
    parser.add_argument('--n_sets',     type=int, default=50,
                        help='Max unique candidates to keep per sample (default: 50)')
    parser.add_argument('--skip_inference', action='store_true',
                        help='Skip inference if CSV already exists (just re-copy)')
    args = parser.parse_args()

    if args.data_dir is None:
        args.data_dir = os.path.normpath(os.path.join(SCRIPT_DIR, '../../Data'))
    label_dir  = os.path.join(args.data_dir, 'labels')
    mpnn_dir   = os.path.join(args.data_dir, 'MPNN_data')
    gcn_csv    = os.path.join(mpnn_dir, f'GCN_data_{args.split}.csv')
    model_base = args.reacon_dir   # GCN_{cat,solv0,...} live directly under models/reacon/

    inf_csv = os.path.join(model_base,
                           f'inference_results_m{args.m_per_slot}_{args.split}.csv')

    # mirror inference CSV to InferenceResults/{split}/ with canonical name
    IR_DIR = os.path.join(ROOT_DIR, 'tar', 'InferenceResults', args.split)
    ir_csv = os.path.join(IR_DIR, 'reacon.csv')

    print(f'Split      : {args.split}')
    print(f'Data       : {gcn_csv}')
    print(f'Models     : {model_base}')
    print(f'm_per_slot : {args.m_per_slot}  (→ {args.m_per_slot**5:,} raw combos)')
    print(f'n_sets     : {args.n_sets}')
    print(f'Inf CSV    : {inf_csv}')

    # ── Phase 1: Inference ───────────────────────────────────────────────────
    if args.skip_inference and os.path.exists(inf_csv):
        print(f'\nSkipping inference — loading {inf_csv}')
    else:
        print('\n--- Phase 1: Inference ---')
        run_inference(gcn_csv, model_base, label_dir,
                      args.m_per_slot, args.n_sets, inf_csv)

    # copy inference CSV to InferenceResults/{split}/
    import shutil
    os.makedirs(IR_DIR, exist_ok=True)
    shutil.copy2(inf_csv, ir_csv)
    print(f'  Copied → {ir_csv}')
    print('\nPhase 1 complete.')


if __name__ == '__main__':
    main()
