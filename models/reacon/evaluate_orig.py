"""
reacon_library_eval.py
======================
Original Reacon inference style using our template condition library.

Pipeline (mirrors prediction.py):
  1. Run 5 GCN slot classifiers → per-slot probability vectors
  2. For each test sample walk r5 → r0:
       - Look up template in condition library
       - Score every (cat, solv, reag) condition in the matched bucket
         using GCN log-probs                         (= condition_selector)
  3. Fallback when no template matches:
       - Top-m per slot × cross-product              (= Nu_condition_selector)
  4. Evaluate and save report

Two-phase design (skip slow GCN inference on reruns):
  Phase 1 → slot_probs_{split}.npz   (one float32 matrix per slot)
  Phase 2 → library_results_{split}.csv  +  library_report_{split}.txt

Usage
-----
  cd models/reacon

  # Full run (Phase 1 + 2)
  python evaluate_orig.py --split test

  # Phase 2 only (slot probs already saved)
  python evaluate_orig.py --split test --skip_inference
"""

import os, sys, math, argparse, pickle, itertools, warnings
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REACON_DIR = SCRIPT_DIR
if REACON_DIR not in sys.path:
    sys.path.insert(0, REACON_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from chemprop.args import PredictArgs
from chemprop.train.make_predictions import make_predictions
from template_reranker import TemplateReranker

NULL       = 'none'
KS         = [1, 3, 5, 10]
ROLES      = ['cat', 'solv0', 'solv1', 'reag0', 'reag1']
GROUPS     = {'Catalyst': [0], 'Solvent': [1, 2], 'Reagent': [3, 4]}
TMPL_RADII = ('r0', 'r1', 'r2', 'r3', 'r4', 'r5')


# ══════════════════════════════════════════════════════════════════════════════
# Labels
# ══════════════════════════════════════════════════════════════════════════════

def load_labels(label_dir):
    cat  = pd.read_csv(os.path.join(label_dir, 'cat_labels.csv'))['cat'].astype(str).tolist()
    solv = pd.read_csv(os.path.join(label_dir, 'solv_labels.csv'))['solv'].astype(str).tolist()
    reag = pd.read_csv(os.path.join(label_dir, 'reag_labels.csv'))['reag'].astype(str).tolist()
    return cat, solv, reag

def norm(s):
    s = str(s).strip()
    return NULL if s in {'None', 'nan', '', 'none', 'NaN'} else s

def decode_row(row, cat_list, solv_list, reag_list):
    return [norm(cat_list[int(row['cat'])]),
            norm(solv_list[int(row['solv0'])]), norm(solv_list[int(row['solv1'])]),
            norm(reag_list[int(row['reag0'])]), norm(reag_list[int(row['reag1'])])]

def encode_cand(parts):
    return ';'.join(parts)

def decode_cand(s):
    if not s or str(s).strip() in {'', 'nan'}:
        return [NULL] * 5
    parts = str(s).split(';')
    while len(parts) < 5:
        parts.append(NULL)
    return [p.strip() or NULL for p in parts[:5]]

def cand_full_set(s):
    return frozenset(v for v in decode_cand(s) if v != NULL)


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1 — GCN inference → slot probability vectors
# ══════════════════════════════════════════════════════════════════════════════

def run_gcn(model_dir, gcn_csv, smiles):
    preds_tmp = os.path.join(SCRIPT_DIR, '_reacon_lib_tmp.csv')
    pa = PredictArgs().parse_args([
        '--test_path',      gcn_csv,
        '--checkpoint_dir', model_dir,
        '--preds_path',     preds_tmp,
    ])
    raw = make_predictions(pa, smiles)
    return [s[0] if isinstance(s[0], list) else s for s in raw]


def run_inference(gcn_csv, model_base, label_dir, out_npz):
    cat_list, solv_list, reag_list = load_labels(label_dir)
    gcn_df = pd.read_csv(gcn_csv)
    smiles = [[str(r)] for r in gcn_df['reaction']]
    N = len(gcn_df)

    slot_probs = {}
    for target in ROLES:
        model_dir = os.path.join(model_base, f'GCN_{target}')
        print(f'  Running GCN_{target} ({N:,} samples) ...')
        slot_probs[target] = run_gcn(model_dir, gcn_csv, smiles)

    # Save as float32 arrays  [N, n_classes]
    arrays = {role: np.array(slot_probs[role], dtype=np.float32)
              for role in ROLES}
    # Also save row ids and ground truth
    arrays['_ids'] = gcn_df['_id'].astype(str).values
    arrays['_gt']  = np.array([
        encode_cand(decode_row(gcn_df.iloc[i], cat_list, solv_list, reag_list))
        for i in range(N)
    ])
    np.savez_compressed(out_npz, **arrays)
    print(f'Slot probs → {out_npz}  ({N:,} samples)')
    return arrays


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2 — Library lookup + scoring
# ══════════════════════════════════════════════════════════════════════════════

def make_index(label_list):
    return {s: i for i, s in enumerate(label_list)}


def log_score_candidate(cat_t, solv_t, reag_t,
                        cat_probs, solv0_probs, solv1_probs,
                        reag0_probs, reag1_probs,
                        cat2idx, solv2idx, reag2idx):
    """
    Log-score a (cat_tuple, solv_tuple, reag_tuple) against per-slot GCN probs.
    Sorted-tuple elements are assigned to slots in order (solv_t[0]→solv0, etc.).
    """
    ls = 0.0
    cat_smi = cat_t[0] if cat_t else NULL
    ls += math.log(max(float(cat_probs[cat2idx.get(cat_smi, 0)]),  1e-12))

    s0 = solv_t[0] if len(solv_t) > 0 else NULL
    s1 = solv_t[1] if len(solv_t) > 1 else NULL
    ls += math.log(max(float(solv0_probs[solv2idx.get(s0, 0)]), 1e-12))
    ls += math.log(max(float(solv1_probs[solv2idx.get(s1, 0)]), 1e-12))

    r0 = reag_t[0] if len(reag_t) > 0 else NULL
    r1 = reag_t[1] if len(reag_t) > 1 else NULL
    ls += math.log(max(float(reag0_probs[reag2idx.get(r0, 0)]), 1e-12))
    ls += math.log(max(float(reag1_probs[reag2idx.get(r1, 0)]), 1e-12))

    return ls


def tuple_to_cand(cat_t, solv_t, reag_t):
    parts = [
        cat_t[0]  if cat_t       else NULL,
        solv_t[0] if len(solv_t) > 0 else NULL,
        solv_t[1] if len(solv_t) > 1 else NULL,
        reag_t[0] if len(reag_t) > 0 else NULL,
        reag_t[1] if len(reag_t) > 1 else NULL,
    ]
    return encode_cand(parts)


def library_lookup(templates, reranker,
                   cat_probs, solv0_probs, solv1_probs,
                   reag0_probs, reag1_probs,
                   cat2idx, solv2idx, reag2idx,
                   n_sets):
    """
    Walk r5 → r0; on first bucket hit, score all conditions and return top-n_sets.
    Returns (cand_list, matched_radius).
      cand_list: [(cand_str, log_score), ...]
    """
    for radius in reranker.RADII:   # ('r5','r4','r3','r2','r1','r0')
        tmpl = templates.get(radius)
        if not tmpl or not str(tmpl).strip():
            continue
        bucket = reranker._libs[radius].get(str(tmpl).strip())
        if bucket is None:
            continue
        scored = []
        for (cat_t, solv_t, reag_t) in bucket.keys():
            ls = log_score_candidate(
                cat_t, solv_t, reag_t,
                cat_probs, solv0_probs, solv1_probs,
                reag0_probs, reag1_probs,
                cat2idx, solv2idx, reag2idx)
            scored.append((tuple_to_cand(cat_t, solv_t, reag_t), ls))
        scored.sort(key=lambda x: x[1], reverse=True)
        # Deduplicate by full set
        seen, unique = set(), []
        for cs, ls in scored:
            key = cand_full_set(cs)
            if key not in seen:
                seen.add(key)
                unique.append((cs, ls))
            if len(unique) >= n_sets:
                break
        return unique, radius
    return [], None


def fallback_lookup(cat_probs, solv0_probs, solv1_probs,
                    reag0_probs, reag1_probs,
                    cat_list, solv_list, reag_list,
                    cat2idx, solv2idx, reag2idx,
                    top_m, n_sets):
    """Top-m per slot cross-product (Nu_condition_selector equivalent)."""
    def topk(probs, m, labels):
        arr = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, None)
        idxs = np.argsort(arr)[::-1][:m]
        return [(labels[i], float(arr[i])) for i in idxs]

    tc = topk(cat_probs,   top_m, cat_list)
    ts0= topk(solv0_probs, top_m, solv_list)
    ts1= topk(solv1_probs, top_m, solv_list)
    tr0= topk(reag0_probs, top_m, reag_list)
    tr1= topk(reag1_probs, top_m, reag_list)

    scored = []
    for (c,cp),(s0,s0p),(s1,s1p),(r0,r0p),(r1,r1p) in itertools.product(
            tc, ts0, ts1, tr0, tr1):
        cat_t  = (c,)  if c  != NULL else ()
        solv_t = tuple(sorted(v for v in [s0, s1] if v != NULL))
        reag_t = tuple(sorted(v for v in [r0, r1] if v != NULL))
        ls = log_score_candidate(
            cat_t, solv_t, reag_t,
            cat_probs, solv0_probs, solv1_probs,
            reag0_probs, reag1_probs,
            cat2idx, solv2idx, reag2idx)
        scored.append((tuple_to_cand(cat_t, solv_t, reag_t), ls))

    scored.sort(key=lambda x: x[1], reverse=True)
    seen, unique = set(), []
    for cs, ls in scored:
        key = cand_full_set(cs)
        if key not in seen:
            seen.add(key)
            unique.append((cs, ls))
        if len(unique) >= n_sets:
            break
    return unique


def load_templates(gcn_csv, split, data_dir):
    tmpl_csv = os.path.join(data_dir, f'data_{split}_template.csv')
    tmpl_cols = [f'template_{r}' for r in TMPL_RADII]
    tmpl_df = pd.read_csv(tmpl_csv, usecols=['row_id'] + tmpl_cols,
                          dtype=str).fillna('').set_index('row_id')
    gcn_df  = pd.read_csv(gcn_csv, usecols=['_id'], dtype=str)
    tmpl_map, missing = {}, 0
    for row_id in gcn_df['_id']:
        if row_id in tmpl_df.index:
            row = tmpl_df.loc[row_id]
            tmpl_map[row_id] = {r: (row[f'template_{r}'] or None) for r in TMPL_RADII}
        else:
            missing += 1
            tmpl_map[row_id] = {r: None for r in TMPL_RADII}
    print(f'  Templates: {len(tmpl_map)-missing}/{len(tmpl_map)} matched')
    return tmpl_map


def run_library_eval(arrays, gcn_csv, data_dir, split, reranker,
                     cat_list, solv_list, reag_list,
                     top_m, n_sets, out_csv, report_path):
    N       = len(arrays['_ids'])
    tmpl_map = load_templates(gcn_csv, split, data_dir)

    cat2idx  = make_index(cat_list)
    solv2idx = make_index(solv_list)
    reag2idx = make_index(reag_list)

    empty_tmpl = {r: None for r in TMPL_RADII}
    radius_stats = Counter()

    header = ['_id', 'reaction_gt', 'gt']
    for j in range(1, n_sets + 1):
        header += [f'cand{j}', f'score{j}']

    # For metrics
    rows_for_metrics = []   # (gt_str, [cand_str, ...])

    records = []
    for i in tqdm(range(N), desc='Library eval'):
        row_id    = str(arrays['_ids'][i])
        gt_str    = str(arrays['_gt'][i])
        templates = tmpl_map.get(row_id, empty_tmpl)

        cat_probs  = arrays['cat'][i]
        solv0_probs= arrays['solv0'][i]
        solv1_probs= arrays['solv1'][i]
        reag0_probs= arrays['reag0'][i]
        reag1_probs= arrays['reag1'][i]

        cands, match_r = library_lookup(
            templates, reranker,
            cat_probs, solv0_probs, solv1_probs,
            reag0_probs, reag1_probs,
            cat2idx, solv2idx, reag2idx, n_sets)

        if not cands:
            cands = fallback_lookup(
                cat_probs, solv0_probs, solv1_probs,
                reag0_probs, reag1_probs,
                cat_list, solv_list, reag_list,
                cat2idx, solv2idx, reag2idx,
                top_m, n_sets)
            match_r = None

        radius_stats[match_r] += 1
        rows_for_metrics.append((gt_str, [cs for cs, _ in cands]))

        rec = {'_id': row_id, 'reaction_gt': gt_str, 'gt': gt_str}
        for j, (cs, ls) in enumerate(cands, 1):
            rec[f'cand{j}']  = cs
            rec[f'score{j}'] = round(ls, 6)
        records.append(rec)

    pd.DataFrame(records, columns=header).to_csv(out_csv, index=False)
    print(f'Results → {out_csv}')

    m = compute_metrics(rows_for_metrics, n_sets)
    report = build_report(m, N, radius_stats, top_m)
    print(report)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f'Report → {report_path}')


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

def set_metrics(gt, pred):
    if not gt and not pred: return 1.0, 1.0, 1.0, 1.0
    tp = len(gt & pred); fp = len(pred - gt); fn = len(gt - pred)
    p = tp/len(pred) if pred else (1.0 if not gt else 0.0)
    r = tp/len(gt)   if gt   else (1.0 if not pred else 0.0)
    f = 2*p*r/(p+r)  if (p+r) > 0 else 0.0
    j = tp/(tp+fp+fn) if (tp+fp+fn) > 0 else 1.0
    return p, r, f, j


def compute_metrics(rows, n_sets):
    """rows: [(gt_str, [cand_str, ...]), ...]"""
    N = len(rows)
    EM1, P1, R1, F1, J1 = [], [], [], [], []
    hit_EM = {k: np.zeros(N, dtype=bool) for k in KS}
    best_R = {k: np.zeros(N) for k in KS}
    best_P = {k: np.zeros(N) for k in KS}
    best_F = {k: np.zeros(N) for k in KS}
    best_J = {k: np.zeros(N) for k in KS}
    group_hit = {g: {k: np.zeros(N, dtype=bool) for k in KS} for g in GROUPS}
    group_R   = {g: {k: np.zeros(N) for k in KS} for g in GROUPS}
    group_pos = {g: np.zeros(N, dtype=bool) for g in GROUPS}

    prev_k = 0
    for k_idx, k in enumerate(KS):
        if k_idx > 0:
            pk = KS[k_idx - 1]
            hit_EM[k] = hit_EM[pk].copy()
            best_R[k]  = best_R[pk].copy()
            best_P[k]  = best_P[pk].copy()
            best_F[k]  = best_F[pk].copy()
            best_J[k]  = best_J[pk].copy()
            for g in GROUPS:
                group_hit[g][k] = group_hit[g][pk].copy()
                group_R[g][k]   = group_R[g][pk].copy()
        for pos in range(prev_k, k):
            for i, (gt_str, cands) in enumerate(rows):
                if pos >= len(cands): continue
                gt_s = cand_full_set(gt_str)
                pr_s = cand_full_set(cands[pos])
                p, r, f, jac = set_metrics(gt_s, pr_s)
                if p   > best_P[k][i]: best_P[k][i] = p
                if r   > best_R[k][i]: best_R[k][i] = r
                if f   > best_F[k][i]: best_F[k][i] = f
                if jac > best_J[k][i]: best_J[k][i] = jac
                if pr_s == gt_s:       hit_EM[k][i]  = True
                gt_parts = decode_cand(gt_str)
                pr_parts = decode_cand(cands[pos])
                for gname, pos_idxs in GROUPS.items():
                    gt_g = frozenset(gt_parts[pi] for pi in pos_idxs if gt_parts[pi] != NULL)
                    pr_g = frozenset(pr_parts[pi] for pi in pos_idxs if pr_parts[pi] != NULL)
                    if gt_g: group_pos[gname][i] = True
                    _, rg, _, _ = set_metrics(gt_g, pr_g)
                    if rg > group_R[gname][k][i]: group_R[gname][k][i] = rg
                    if pr_g == gt_g: group_hit[gname][k][i] = True
        prev_k = k

    for i, (gt_str, cands) in enumerate(rows):
        if not cands:
            EM1.append(False); P1.append(0.); R1.append(0.); F1.append(0.); J1.append(0.)
            continue
        gt_s = cand_full_set(gt_str)
        pr_s = cand_full_set(cands[0])
        p, r, f, jac = set_metrics(gt_s, pr_s)
        EM1.append(pr_s == gt_s); P1.append(p); R1.append(r); F1.append(f); J1.append(jac)

    return {'EM1': np.array(EM1), 'P1': np.array(P1), 'R1': np.array(R1),
            'F1': np.array(F1),  'J1': np.array(J1),
            'hit_EM': hit_EM, 'best_R': best_R, 'best_P': best_P,
            'best_F': best_F, 'best_J': best_J,
            'group_hit': group_hit, 'group_R': group_R, 'group_pos': group_pos}


# ══════════════════════════════════════════════════════════════════════════════
# Report
# ══════════════════════════════════════════════════════════════════════════════

def build_report(m, N, radius_stats, top_m):
    lines = []
    def h(t):
        lines.append('\n' + '═'*74)
        lines.append(f'  {t}')
        lines.append('═'*74)

    h(f'Reacon  Library Eval  |  N={N}  |  fallback top_m={top_m}')

    h('0. Template Radius Usage')
    lines.append(f'  {"Radius":<10} {"Count":>8}  {"Rate":>8}')
    for r in ('r5', 'r4', 'r3', 'r2', 'r1', 'r0', None):
        c = radius_stats.get(r, 0)
        lines.append(f'  {str(r):<10} {c:>8}  {c/N:>8.3f}')
    lines.append(f'\n  Fallback (no template match): {radius_stats.get(None,0)} '
                 f'({100*radius_stats.get(None,0)/N:.1f}%)')

    h('1. Full-Set Top-k EM and Recall')
    lines.append(f'  {"k":<6}  {"EM@k":>10}  {"R@k":>10}  {"P@k":>10}  {"F1@k":>10}')
    lines.append('  ' + '─'*50)
    for k in KS:
        lines.append(f'  top-{k:<3}  {m["hit_EM"][k].mean():>10.4f}'
                     f'  {m["best_R"][k].mean():>10.4f}'
                     f'  {m["best_P"][k].mean():>10.4f}'
                     f'  {m["best_F"][k].mean():>10.4f}')

    h('2. Group-level Recall@k and EM@k')
    for gname in GROUPS:
        pm    = m['group_pos'][gname]
        n_pos = pm.sum()
        lines.append(f'\n  Group: {gname}  (n_pos={n_pos})')
        lines.append(f'  {"k":<6}  {"R@k":>10}  {"EM@k":>10}')
        lines.append('  ' + '─'*30)
        for k in KS:
            br = m['group_R'][gname][k][pm].mean() if pm.any() else float('nan')
            em = m['group_hit'][gname][k].mean()
            lines.append(f'  top-{k:<3}  {br:>10.4f}  {em:>10.4f}')

    h('3. Top-1 Full-Set Metrics')
    lines.append(f'  {"Metric":<14}  {"Value":>10}')
    lines.append('  ' + '─'*28)
    for name, arr in [('Precision', m['P1']), ('Recall', m['R1']),
                      ('F1', m['F1']), ('Jaccard', m['J1']),
                      ('Exact Match', m['EM1'].astype(float))]:
        lines.append(f'  {name:<14}  {arr.mean():>10.4f}')

    return '\n'.join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description='Reacon library-based inference + evaluation')
    ap.add_argument('--split',          choices=['test', 'val'], default='test')
    ap.add_argument('--top_m',          type=int, default=3,
                    help='Top-m per slot for fallback cross-product (default: 3 = original Reacon)')
    ap.add_argument('--n_sets',         type=int, default=50,
                    help='Max candidates to keep per sample (default: 50)')
    ap.add_argument('--data_dir',       default=None)
    ap.add_argument('--reacon_dir',     default=None)
    ap.add_argument('--lib_path',       default=None,
                    help='Path to standard_template_library.pkl')
    ap.add_argument('--skip_inference', action='store_true',
                    help='Skip Phase 1 if slot_probs_{split}.npz already exists')
    ap.add_argument('--rebuild_lib',    action='store_true')
    args = ap.parse_args()

    if args.data_dir is None:
        args.data_dir = os.path.normpath(os.path.join(SCRIPT_DIR, '../../Data'))
    if args.reacon_dir is None:
        args.reacon_dir = REACON_DIR

    label_dir  = os.path.join(args.data_dir, 'labels')
    mpnn_dir   = os.path.join(args.data_dir, 'MPNN_data')
    gcn_csv    = os.path.join(mpnn_dir,      f'GCN_data_{args.split}.csv')
    model_base = args.reacon_dir   # GCN_{cat,solv0,...} live directly under models/reacon/

    out_dir  = SCRIPT_DIR   # outputs go to Reacon(orig)/ alongside this script
    npz_path = os.path.join(model_base, f'slot_probs_{args.split}.npz')  # large npz stays in models/
    out_csv  = os.path.join(out_dir, f'library_results_{args.split}.csv')
    rep_path = os.path.join(out_dir, f'library_report_{args.split}.txt')

    print(f'Split    : {args.split}')
    print(f'Data     : {gcn_csv}')
    print(f'top_m    : {args.top_m}  (fallback cross-product)')
    print(f'n_sets   : {args.n_sets}')

    # ── Phase 1 ──────────────────────────────────────────────────────────────
    cat_list, solv_list, reag_list = load_labels(label_dir)

    if args.skip_inference and os.path.exists(npz_path):
        print(f'\nLoading slot probs: {npz_path}')
        arrays = dict(np.load(npz_path, allow_pickle=True))
    else:
        print('\n--- Phase 1: GCN Inference ---')
        arrays = run_inference(gcn_csv, model_base, label_dir, npz_path)

    # ── Phase 2 ──────────────────────────────────────────────────────────────
    print('\n--- Phase 2: Library Lookup + Scoring ---')

    shared_lib = os.path.join(args.data_dir, 'standard_template_library.pkl')
    lib_path   = args.lib_path or shared_lib
    if not args.rebuild_lib and os.path.exists(lib_path):
        print(f'Loading template library: {lib_path}')
        with open(lib_path, 'rb') as f:
            reranker = pickle.load(f)
    else:
        print('Building template library from train+val ...')
        reranker = TemplateReranker(data_dir=args.data_dir)
        with open(lib_path, 'wb') as f:
            pickle.dump(reranker, f)
        print(f'Cached → {lib_path}')

    run_library_eval(arrays, gcn_csv, args.data_dir, args.split, reranker,
                     cat_list, solv_list, reag_list,
                     args.top_m, args.n_sets, out_csv, rep_path)


if __name__ == '__main__':
    main()
