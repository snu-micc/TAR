"""
entropy_rerank_all.py
=====================
Entropy-adaptive template re-ranking applied to all three models:
  - Roleagnostic GNN
  - GAO 2018
  - Reacon

Core idea
---------
Standard reranking uses a fixed λ for every sample:
    combined = beam_score + λ · log P(flat_set | template)

Entropy analysis shows that benefit decays with template entropy H:
    H=0  (one known condition-set)  →  high λ helps a lot
    H>2  (many plausible sets)      →  high λ may hurt

Entropy-adaptive formula:
    λ_eff = λ / (1 + α · H)
    H=None (no template match)  →  λ_eff = 0

All models use the same flat template library (standard_template_library.pkl)
with key = sorted tuple of all SMILES (role-agnostic flat set).

Outputs (per model, in their own output directories)
-------
  entropy_rerank_{split}_lam{L}_alpha{A}.txt
  predictions_entropy_reranked_{split}_lam{L}_alpha{A}.csv

Usage
-----
  cd Baseline
  python entropy_rerank_all.py --lam 2.0 --alpha 1.0
  python entropy_rerank_all.py --lam 4.0 --alpha 0.5
  python entropy_rerank_all.py --model roleagnostic --lam 2.0 --alpha 1.0
  python entropy_rerank_all.py --model gao         --lam 2.0 --alpha 1.0
  python entropy_rerank_all.py --model reacon       --lam 2.0 --alpha 1.0

  # α=0 reproduces standard fixed-λ reranking (sanity check)
  python entropy_rerank_all.py --lam 2.0 --alpha 0.0
"""

import os
import sys
import math
import argparse
import pickle
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.normpath(os.path.join(BASE_DIR, '..'))
DATA_DIR  = os.path.join(ROOT_DIR, 'Data')

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
from template_reranker import RoleAgnosticTemplateReranker

LIB_PKL             = os.path.join(DATA_DIR, 'flat_template_library.pkl')
LIB_PKL_TRAIN_ONLY  = os.path.join(DATA_DIR, 'flat_template_library_train_only.pkl')
ENTROPY_CSV         = os.path.join(DATA_DIR, 'entropy_analysis.csv')
ENTROPY_CSV_TRAIN   = os.path.join(DATA_DIR, 'entropy_train_only', 'entropy_analysis.csv')

TMPL_RADII = ('r0', 'r1', 'r2', 'r3', 'r4', 'r5')
RADII_DESC = ('r5', 'r4', 'r3', 'r2', 'r1', 'r0')
KS         = [1, 3, 5, 10]
NULL_SET   = {'none', 'None', 'nan', 'NaN', ''}

# ── model configs ─────────────────────────────────────────────────────────────
# Each entry:
#   beam_csv     : path to beam predictions
#   out_dir      : where to save outputs
#   id_col       : _id column name, or None for positional alignment
#   gt_col       : ground truth column
#   cand_prefix  : 'cand' or 'top' (for reranked top1, top2, ...)
#   n_beam_cands : max candidates to check (used to scan columns)

IR_TEST = os.path.join(BASE_DIR, 'InferenceResults', 'test')
IR_VAL  = os.path.join(BASE_DIR, 'InferenceResults', 'val')

MODEL_CONFIGS = {
    # ── QUARC (reaction-class conditioned) ────────────────────────────────────
    'QUARC_beam30_test': dict(
        beam_csv   = os.path.join(IR_TEST, 'quarc_rxnclass_beam30.csv'),
        out_dir    = os.path.join(BASE_DIR, 'TAR_Results/test/QUARC_beam30'),
        id_col='_id', gt_col='gt_flat', cand_prefix='cand', positional=False,
    ),
    'QUARC_beam50_test': dict(
        beam_csv   = os.path.join(IR_TEST, 'quarc_rxnclass_beam50.csv'),
        out_dir    = os.path.join(BASE_DIR, 'TAR_Results/test/QUARC_beam50'),
        id_col='_id', gt_col='gt_flat', cand_prefix='cand', positional=False,
    ),
    'QUARC_beam30_val': dict(
        beam_csv    = os.path.join(IR_VAL, 'quarc_rxnclass_beam30.csv'),
        out_dir     = os.path.join(BASE_DIR, 'TAR_Results/val/QUARC_beam30'),
        id_col='_id', gt_col='gt_flat', cand_prefix='cand', positional=False,
        lib_pkl     = LIB_PKL_TRAIN_ONLY,
        entropy_csv = ENTROPY_CSV_TRAIN,
    ),
    'QUARC_beam50_val': dict(
        beam_csv    = os.path.join(IR_VAL, 'quarc_rxnclass_beam50.csv'),
        out_dir     = os.path.join(BASE_DIR, 'TAR_Results/val/QUARC_beam50'),
        id_col='_id', gt_col='gt_flat', cand_prefix='cand', positional=False,
        lib_pkl     = LIB_PKL_TRAIN_ONLY,
        entropy_csv = ENTROPY_CSV_TRAIN,
    ),
    # ── GAO ───────────────────────────────────────────────────────────────────
    'gao_beam30_test': dict(
        beam_csv   = os.path.join(IR_TEST, 'gao_beam30.csv'),
        out_dir    = os.path.join(BASE_DIR, 'TAR_Results/test/gao_beam30'),
        id_col=None, gt_col='gt', cand_prefix='cand', positional=True,
    ),
    'gao_beam50_test': dict(
        beam_csv   = os.path.join(IR_TEST, 'gao_beam50.csv'),
        out_dir    = os.path.join(BASE_DIR, 'TAR_Results/test/gao_beam50'),
        id_col=None, gt_col='gt', cand_prefix='cand', positional=True,
    ),
    'gao_beam30_val': dict(
        beam_csv    = os.path.join(IR_VAL, 'gao_beam30.csv'),
        out_dir     = os.path.join(BASE_DIR, 'TAR_Results/val/gao_beam30'),
        id_col=None, gt_col='gt', cand_prefix='cand', positional=True,
        lib_pkl     = LIB_PKL_TRAIN_ONLY,
        entropy_csv = ENTROPY_CSV_TRAIN,
    ),
    'gao_beam50_val': dict(
        beam_csv    = os.path.join(IR_VAL, 'gao_beam50.csv'),
        out_dir     = os.path.join(BASE_DIR, 'TAR_Results/val/gao_beam50'),
        id_col=None, gt_col='gt', cand_prefix='cand', positional=True,
        lib_pkl     = LIB_PKL_TRAIN_ONLY,
        entropy_csv = ENTROPY_CSV_TRAIN,
    ),
    # ── Reacon* (single beam CSV; max_cands baked in per variant) ─────────────
    'reacon_beam30_test': dict(
        beam_csv   = os.path.join(IR_TEST, 'reacon.csv'),
        out_dir    = os.path.join(BASE_DIR, 'TAR_Results/test/reacon_beam30'),
        id_col='_id', gt_col='gt', cand_prefix='cand', positional=False,
        max_cands  = 30,
    ),
    'reacon_beam50_test': dict(
        beam_csv   = os.path.join(IR_TEST, 'reacon.csv'),
        out_dir    = os.path.join(BASE_DIR, 'TAR_Results/test/reacon_beam50'),
        id_col='_id', gt_col='gt', cand_prefix='cand', positional=False,
        max_cands  = 50,
    ),
    'reacon_beam30_val': dict(
        beam_csv    = os.path.join(IR_VAL, 'reacon.csv'),
        out_dir     = os.path.join(BASE_DIR, 'TAR_Results/val/reacon_beam30'),
        id_col='_id', gt_col='gt', cand_prefix='cand', positional=False,
        max_cands   = 30,
        lib_pkl     = LIB_PKL_TRAIN_ONLY,
        entropy_csv = ENTROPY_CSV_TRAIN,
    ),
    'reacon_beam50_val': dict(
        beam_csv    = os.path.join(IR_VAL, 'reacon.csv'),
        out_dir     = os.path.join(BASE_DIR, 'TAR_Results/val/reacon_beam50'),
        id_col='_id', gt_col='gt', cand_prefix='cand', positional=False,
        max_cands   = 50,
        lib_pkl     = LIB_PKL_TRAIN_ONLY,
        entropy_csv = ENTROPY_CSV_TRAIN,
    ),
}


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_flat(s):
    """Any semicolon-separated SMILES (flat or role-based 'cat;s1;s2;r1;r2')
    → frozenset of non-null SMILES tokens."""
    if pd.isna(s) or not str(s).strip():
        return frozenset()
    return frozenset(t.strip() for t in str(s).split(';')
                     if t.strip() and t.strip() not in NULL_SET)


def build_entropy_lookup(csv_path):
    df = pd.read_csv(csv_path)
    return {(r['radius'], str(r['template'])): float(r['entropy'])
            for _, r in df.iterrows()}


def sample_entropy(templates, entropy_lookup):
    """Find entropy of the finest matching template (r5→r0).
    Returns (H, radius) or (None, None)."""
    for r in RADII_DESC:
        tmpl = templates.get(r)
        if not tmpl or str(tmpl).strip() in ('', 'nan', 'None'):
            continue
        key = (r, str(tmpl).strip())
        if key in entropy_lookup:
            return entropy_lookup[key], r
    return None, None


def effective_lam(lam, H, alpha):
    """λ_eff = λ / (1 + α·H).  H=None (no template) → 0."""
    if H is None:
        return 0.0
    return lam / (1.0 + alpha * H)


def load_templates(data_dir, split, id_list, positional=False, tmpl_df=None):
    """
    Returns list of template dicts aligned to id_list (or position).
    tmpl_df: preloaded template DataFrame (shared across models).
    """
    if tmpl_df is None:
        tmpl_csv = os.path.join(data_dir, f'data_{split}_template.csv')
        tmpl_df  = pd.read_csv(tmpl_csv,
                               usecols=['row_id'] + [f'template_{r}' for r in TMPL_RADII],
                               dtype=str).fillna('').set_index('row_id')

    if positional:
        # Align by row position
        result = []
        for i in range(len(id_list)):
            if i < len(tmpl_df):
                row = tmpl_df.iloc[i]
                result.append({r: (row[f'template_{r}'] or None) for r in TMPL_RADII})
            else:
                result.append({r: None for r in TMPL_RADII})
        return result
    else:
        result = []
        for rid in id_list:
            rid = str(rid)
            if rid in tmpl_df.index:
                row = tmpl_df.loc[rid]
                result.append({r: (row[f'template_{r}'] or None) for r in TMPL_RADII})
            else:
                result.append({r: None for r in TMPL_RADII})
        return result


# ── metrics ───────────────────────────────────────────────────────────────────

def set_metrics(gt, pred):
    if not gt and not pred: return 1.0, 1.0, 1.0, 1.0
    tp = len(gt & pred); fp = len(pred - gt); fn = len(gt - pred)
    p  = tp / len(pred) if pred else (1.0 if not gt else 0.0)
    r  = tp / len(gt)   if gt   else (1.0 if not pred else 0.0)
    f  = 2*p*r/(p+r)   if (p+r) > 0 else 0.0
    j  = tp/(tp+fp+fn) if (tp+fp+fn) > 0 else 1.0
    return p, r, f, j


def compute_metrics(rows, ordered_ranks, cand_prefix):
    N = len(rows)
    EM1, P1, R1, F1 = [], [], [], []
    hit_EM = {k: np.zeros(N, bool) for k in KS}
    best_R = {k: np.zeros(N)       for k in KS}
    best_P = {k: np.zeros(N)       for k in KS}
    best_F = {k: np.zeros(N)       for k in KS}

    prev_k = 0
    for ki, k in enumerate(KS):
        if ki > 0:
            pk = KS[ki - 1]
            hit_EM[k] = hit_EM[pk].copy()
            best_R[k]  = best_R[pk].copy()
            best_P[k]  = best_P[pk].copy()
            best_F[k]  = best_F[pk].copy()
        for pos in range(prev_k, k):
            for i, row in enumerate(rows):
                if pos >= len(ordered_ranks[i]): continue
                old_j = ordered_ranks[i][pos]
                gt_s  = parse_flat(str(row.get('_gt', '')))
                pr_s  = parse_flat(str(row.get(f'{cand_prefix}{old_j}', '')))
                p, r, f, j = set_metrics(gt_s, pr_s)
                if p > best_P[k][i]: best_P[k][i] = p
                if r > best_R[k][i]: best_R[k][i] = r
                if f > best_F[k][i]: best_F[k][i] = f
                if pr_s == gt_s:     hit_EM[k][i]  = True
        prev_k = k

    for i, row in enumerate(rows):
        gt_s = parse_flat(str(row.get('_gt', '')))
        if not ordered_ranks[i]:
            EM1.append(False); P1.append(0.); R1.append(0.); F1.append(0.)
            continue
        pr_s = parse_flat(str(row.get(f'{cand_prefix}{ordered_ranks[i][0]}', '')))
        p, r, f, j = set_metrics(gt_s, pr_s)
        EM1.append(pr_s == gt_s); P1.append(p); R1.append(r); F1.append(f)

    return {'EM1': np.array(EM1), 'P1': np.array(P1),
            'R1': np.array(R1),  'F1': np.array(F1),
            'hit_EM': hit_EM, 'best_R': best_R, 'best_P': best_P, 'best_F': best_F}


# ── report ─────────────────────────────────────────────────────────────────────

def build_report(model_name, m_b, m_t, lam, alpha, N, radius_stats, entropy_bins):
    lines = []
    def h(t):
        lines.append('\n' + '═' * 78)
        lines.append(f'  {t}')
        lines.append('═' * 78)

    h(f'Entropy-Adaptive Re-ranking  |  {model_name}  |  N={N}  |  λ={lam}  |  α={alpha}')
    lines.append(f'  Formula: λ_eff = λ / (1 + α·H)')
    lines.append(f'  H=0: λ_eff={lam:.2f}  H=1: λ_eff={lam/(1+alpha):.2f}'
                 f'  H=2: λ_eff={lam/(1+2*alpha):.2f}  H=3: λ_eff={lam/(1+3*alpha):.2f}')

    h('0. Radius Distribution  (finest template matched per sample)')
    lines.append(f'  {"Radius":<10} {"Count":>8}  {"Rate":>8}')
    for r in ('r5', 'r4', 'r3', 'r2', 'r1', 'r0', None):
        c = radius_stats.get(r, 0)
        lines.append(f'  {str(r):<10} {c:>8}  {c/N:>8.3f}')
    lines.append(f'\n  No-template fallback: {radius_stats.get(None,0)} '
                 f'({100*radius_stats.get(None,0)/N:.1f}%)')

    h('0b. EM@k by Entropy Bin')
    for k in KS:
        em_key = 'EM1' if k == 1 else f'hit_EM'
        lines.append(f'\n  ── top-{k} {"─" * 56}')
        lines.append(f'  {"Bin":<14} {"N":>8}  {"Rate":>7}'
                     f'  {"Beam EM":>9}  {"Rrnk EM":>9}  {"ΔEM":>7}')
        lines.append('  ' + '─' * 62)
        for bin_name, idxs in entropy_bins.items():
            if not len(idxs): continue
            n    = len(idxs)
            if k == 1:
                b_em = m_b['EM1'][idxs].mean()
                r_em = m_t['EM1'][idxs].mean()
            else:
                b_em = m_b['hit_EM'][k][idxs].mean()
                r_em = m_t['hit_EM'][k][idxs].mean()
            lines.append(f'  {bin_name:<14} {n:>8,}  {n/N:>7.3f}'
                         f'  {b_em:>9.4f}  {r_em:>9.4f}  {r_em-b_em:>+7.4f}')
        lines.append('  ' + '─' * 62)
        if k == 1:
            b_ov = m_b['EM1'].mean(); r_ov = m_t['EM1'].mean()
        else:
            b_ov = m_b['hit_EM'][k].mean(); r_ov = m_t['hit_EM'][k].mean()
        lines.append(f'  {"OVERALL":<14} {N:>8,}  {"1.000":>7}'
                     f'  {b_ov:>9.4f}  {r_ov:>9.4f}  {r_ov-b_ov:>+7.4f}')

    h('1. Full-Set Top-k EM and Recall')
    lines.append(f'  {"k":<7}  {"Beam EM":>9}  {"Rrnk EM":>9}  {"ΔEM":>7}'
                 f'  {"Beam R":>9}  {"Rrnk R":>9}  {"ΔR":>7}')
    lines.append('  ' + '─' * 68)
    for k in KS:
        b_em = m_b['hit_EM'][k].mean(); r_em = m_t['hit_EM'][k].mean()
        b_r  = m_b['best_R'][k].mean(); r_r  = m_t['best_R'][k].mean()
        lines.append(f'  top-{k:<3}  {b_em:>9.4f}  {r_em:>9.4f}  {r_em-b_em:>+7.4f}'
                     f'  {b_r:>9.4f}  {r_r:>9.4f}  {r_r-b_r:>+7.4f}')

    h('2. Top-1 Full-Set Metrics')
    lines.append(f'  {"Metric":<12}  {"Beam":>9}  {"Reranked":>9}  {"Δ":>7}')
    lines.append('  ' + '─' * 44)
    for name, key in [('Precision','P1'), ('Recall','R1'),
                      ('F1','F1'), ('Exact Match','EM1')]:
        bv = m_b[key].mean(); rv = m_t[key].mean()
        lines.append(f'  {name:<12}  {bv:>9.4f}  {rv:>9.4f}  {rv-bv:>+7.4f}')

    return '\n'.join(lines)


# ── report-only (regenerate txt from existing predictions CSV) ────────────────

def run_model_report_only(model_name, cfg, lam, alpha, split, entropy_lookup, tmpl_df,
                          max_cands=None):
    """Regenerate .txt report from existing predictions CSV — no reranking."""
    max_cands = cfg.get('max_cands', max_cands)
    lam_str   = str(lam).replace('.', 'p')
    alpha_str = str(alpha).replace('.', 'p')
    tag       = f'lam{lam_str}_alpha{alpha_str}'
    if max_cands:
        tag += f'_top{max_cands}'
    out_dir  = cfg['out_dir']
    out_csv  = os.path.join(out_dir, f'predictions_entropy_reranked_{split}_{tag}.csv')
    out_txt  = os.path.join(out_dir, f'entropy_rerank_{split}_{tag}.txt')

    if not os.path.exists(out_csv):
        print(f'  [skip] CSV not found: {out_csv}')
        return None, None

    print(f'\n{"="*60}')
    print(f'  [report-only] {model_name}  |  λ={lam}  |  α={alpha}')
    print(f'{"="*60}')

    # load beam CSV (for beam metrics + GT + entropy bin assignment)
    df_beam = pd.read_csv(cfg['beam_csv'], low_memory=False)
    N = len(df_beam)
    n_beam = sum(1 for j in range(1, 500) if f'{cfg["cand_prefix"]}{j}' in df_beam.columns)
    if max_cands:
        n_beam = min(n_beam, max_cands)

    rows_beam = df_beam.fillna('').to_dict('records')
    for row in rows_beam:
        row['_gt'] = row.get(cfg['gt_col'], '')

    # load predictions CSV (for reranked metrics)
    df_rr = pd.read_csv(out_csv, low_memory=False)
    n_rr  = sum(1 for j in range(1, 500) if f'cand{j}' in df_rr.columns)
    rows_rr = df_rr.fillna('').to_dict('records')
    for row in rows_rr:
        row['_gt'] = row.get('gt', '')

    # id list for template alignment
    if cfg['positional']:
        id_list = list(range(N))
    else:
        id_list = df_beam[cfg['id_col']].astype(str).tolist()

    # load templates and assign entropy bins (fast: no reranking)
    print('  Assigning entropy bins ...')
    tmpls_list   = load_templates(DATA_DIR, split, id_list,
                                  positional=cfg['positional'], tmpl_df=tmpl_df)
    radius_stats = Counter()
    entropy_bins = {'H=0': [], '0<H≤1': [], '1<H≤2': [], 'H>2': [], 'no_template': []}

    for i, tmpls in enumerate(tqdm(tmpls_list, desc='  entropy bins', leave=False)):
        H, r_used = sample_entropy(tmpls, entropy_lookup)
        radius_stats[r_used] += 1
        if H is None:      entropy_bins['no_template'].append(i)
        elif H == 0.0:     entropy_bins['H=0'].append(i)
        elif H <= 1.0:     entropy_bins['0<H≤1'].append(i)
        elif H <= 2.0:     entropy_bins['1<H≤2'].append(i)
        else:              entropy_bins['H>2'].append(i)

    entropy_bins_np = {b: np.array(v, dtype=int) for b, v in entropy_bins.items()}

    # sequential rank lists (cand1=rank1, cand2=rank2, ...)
    beam_ranks    = [list(range(1, n_beam + 1))] * N
    reranked_ranks = [list(range(1, n_rr + 1))] * N

    print('  Computing metrics ...')
    m_beam = compute_metrics(rows_beam, beam_ranks, cfg['cand_prefix'])
    m_rrnk = compute_metrics(rows_rr,   reranked_ranks, 'cand')

    report = build_report(model_name, m_beam, m_rrnk, lam, alpha, N,
                          radius_stats, entropy_bins_np)
    print(report)
    with open(out_txt, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f'\n  Report → {out_txt}')

    return m_beam['EM1'].mean(), m_rrnk['EM1'].mean()


# ── per-model reranking ───────────────────────────────────────────────────────

def run_model(model_name, cfg, lam, alpha, split, reranker, entropy_lookup, tmpl_df,
              max_cands=None):
    max_cands = cfg.get('max_cands', max_cands)
    print(f'\n{"="*60}')
    print(f'  Model: {model_name}  |  λ={lam}  |  α={alpha}'
          + (f'  |  top-{max_cands}' if max_cands else ''))
    print(f'{"="*60}')

    lam_str   = str(lam).replace('.', 'p')
    alpha_str = str(alpha).replace('.', 'p')
    tag       = f'lam{lam_str}_alpha{alpha_str}'
    if max_cands:
        tag += f'_top{max_cands}'
    out_dir   = cfg['out_dir']
    os.makedirs(out_dir, exist_ok=True)
    out_csv   = os.path.join(out_dir, f'predictions_entropy_reranked_{split}_{tag}.csv')
    out_txt   = os.path.join(out_dir, f'entropy_rerank_{split}_{tag}.txt')

    # load beam CSV
    print(f'  Loading: {cfg["beam_csv"]}')
    df = pd.read_csv(cfg['beam_csv'], low_memory=False)
    n_cands = sum(1 for j in range(1, 500)
                  if f'{cfg["cand_prefix"]}{j}' in df.columns)
    if max_cands:
        n_cands = min(n_cands, max_cands)
    N = len(df)
    print(f'  {N:,} samples  |  {n_cands} candidates used'
          + (f' (capped from original)' if max_cands else ''))

    # id list for template alignment
    if cfg['positional']:
        id_list = list(range(N))
    else:
        id_list = df[cfg['id_col']].astype(str).tolist()

    # load templates
    print('  Loading templates ...')
    tmpls_list = load_templates(DATA_DIR, split, id_list,
                                positional=cfg['positional'], tmpl_df=tmpl_df)

    # build rows list with unified gt key '_gt'
    rows = df.fillna('').to_dict('records')
    for row in rows:
        row['_gt'] = row.get(cfg['gt_col'], '')

    # ── reranking loop ────────────────────────────────────────────────────────
    print(f'  Reranking ...')
    beam_ranks     = []
    reranked_ranks = []
    radius_stats   = Counter()
    entropy_bins   = {'H=0':[], '0<H≤1':[], '1<H≤2':[], 'H>2':[], 'no_template':[]}

    prefix = cfg['cand_prefix']

    for i, row in enumerate(tqdm(rows, desc=f'  {model_name}')):
        templates = tmpls_list[i]
        H, _      = sample_entropy(templates, entropy_lookup)
        lam_eff   = effective_lam(lam, H, alpha)

        # entropy bin tracking
        if H is None:       entropy_bins['no_template'].append(i)
        elif H == 0.0:      entropy_bins['H=0'].append(i)
        elif H <= 1.0:      entropy_bins['0<H≤1'].append(i)
        elif H <= 2.0:      entropy_bins['1<H≤2'].append(i)
        else:               entropy_bins['H>2'].append(i)

        # collect candidates
        cands = []
        for j in range(1, n_cands + 1):
            cs = str(row.get(f'{prefix}{j}', ''))
            sc = row.get(f'score{j}', None)
            if not cs or cs in ('nan', 'none') or sc is None or sc == '':
                continue
            cands.append({'rank': j, 'cand_str': cs, 'log_score': float(sc)})

        beam_ranks.append([c['rank'] for c in cands])

        # score with entropy-adaptive λ
        scored = []
        for c in cands:
            flat_key      = tuple(sorted(parse_flat(c['cand_str'])))
            log_p, r_used = reranker.log_prior_flat(flat_key, templates)
            combined      = c['log_score'] + lam_eff * log_p
            scored.append({**c, 'combined_score': combined, 'radius_used': r_used})

        scored.sort(key=lambda x: x['combined_score'], reverse=True)
        reranked_ranks.append([c['rank'] for c in scored])
        radius_stats[scored[0]['radius_used'] if scored else None] += 1

    # ── metrics ───────────────────────────────────────────────────────────────
    print('  Computing metrics ...')
    entropy_bins_np = {b: np.array(v, dtype=int) for b, v in entropy_bins.items()}
    m_beam = compute_metrics(rows, beam_ranks,     prefix)
    m_rrnk = compute_metrics(rows, reranked_ranks, prefix)

    report = build_report(model_name, m_beam, m_rrnk, lam, alpha, N,
                          radius_stats, entropy_bins_np)
    print(report)

    with open(out_txt, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f'\n  Report → {out_txt}')

    # ── save CSV ──────────────────────────────────────────────────────────────
    records = []
    id_col  = cfg['id_col']
    for row, ranked in zip(rows, reranked_ranks):
        rec = {}
        if id_col:
            rec[id_col] = row.get(id_col, '')
        rec['gt'] = row.get('_gt', '')
        for new_j, old_j in enumerate(ranked, 1):
            rec[f'cand{new_j}']  = row.get(f'{prefix}{old_j}', '')
            rec[f'score{new_j}'] = row.get(f'score{old_j}', '')
        records.append(rec)
    pd.DataFrame(records).to_csv(out_csv, index=False)
    print(f'  CSV → {out_csv}')

    return m_beam['EM1'].mean(), m_rrnk['EM1'].mean()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lam',   type=float, nargs='+', default=[2.0],
                    help='Base lambda value(s) (default: 2.0)')
    ap.add_argument('--alpha', type=float, nargs='+', default=[1.0],
                    help='Entropy decay rate α (default: 1.0). '
                         'α=0 → fixed λ (same as original). '
                         'α=1 → λ halved at H=1 bit. '
                         'Multiple values accepted.')
    ap.add_argument('--split', default='test')
    ap.add_argument('--model', default='all',
                    choices=['all',
                             'quarc', 'gao', 'reacon',   # short aliases (= *_beam30_test)
                             'QUARC_beam30_test', 'QUARC_beam50_test',
                             'QUARC_beam30_val',  'QUARC_beam50_val',
                             'gao_beam30_test',   'gao_beam50_test',
                             'gao_beam30_val',    'gao_beam50_val',
                             'reacon_beam30_test','reacon_beam50_test',
                             'reacon_beam30_val', 'reacon_beam50_val'],
                    help='Which model to run (default: all). Short aliases '
                         'quarc/gao/reacon map to the beam30 test-split config.')
    ap.add_argument('--max_cands', type=int, default=None,
                    help='Cap the number of beam candidates used for reranking '
                         '(e.g. --max_cands 30 to use only top-30 of a top-50 beam).')
    ap.add_argument('--lib_path', type=str, default=None,
                    help='Path to flat template library .pkl '
                         '(default: Data/flat_template_library.pkl). '
                         'Use Data/flat_template_library_train_only.pkl for val-set sweeps.')
    ap.add_argument('--entropy_path', type=str, default=None,
                    help='Path to entropy_analysis.csv '
                         '(default: Data/entropy_analysis.csv). '
                         'Use Data/entropy_train_only/entropy_analysis.csv for val-set sweeps.')
    ap.add_argument('--report_only', action='store_true',
                    help='Regenerate .txt reports from existing predictions CSVs only '
                         '(no reranking, no template library needed). '
                         'Use after changing build_report to refresh all txt files.')
    args = ap.parse_args()

    lam_list      = args.lam
    alpha_list    = args.alpha
    # short aliases → full config keys (default beam30, test split)
    MODEL_ALIASES = {'quarc': 'QUARC_beam30_test',
                     'gao':   'gao_beam30_test',
                     'reacon':'reacon_beam30_test'}
    selected      = MODEL_ALIASES.get(args.model, args.model)
    models_to_run = (list(MODEL_CONFIGS.keys())
                     if selected == 'all' else [selected])

    # ── template CSV (always needed) ──────────────────────────────────────────
    print(f'Loading template CSV ...')
    tmpl_csv = os.path.join(DATA_DIR, f'data_{args.split}_template.csv')
    tmpl_df  = pd.read_csv(tmpl_csv,
                            usecols=['row_id'] + [f'template_{r}' for r in TMPL_RADII],
                            dtype=str).fillna('').set_index('row_id')
    print(f'  {len(tmpl_df):,} rows')

    # ── run models (entropy/library loaded per-model, cached by path) ─────────
    _entropy_cache  = {}
    _reranker_cache = {}

    def get_entropy(path):
        if path not in _entropy_cache:
            print(f'Loading entropy table: {path}')
            _entropy_cache[path] = build_entropy_lookup(path)
            print(f'  {len(_entropy_cache[path]):,} entries')
        return _entropy_cache[path]

    def get_reranker(path):
        if path not in _reranker_cache:
            print(f'Loading flat template library: {path}')
            with open(path, 'rb') as fh:
                _reranker_cache[path] = pickle.load(fh)
            print(f'  Loaded.')
        return _reranker_cache[path]

    summary = {}
    for model_name in models_to_run:
        cfg          = MODEL_CONFIGS[model_name]
        ent_path     = args.entropy_path or cfg.get('entropy_csv', ENTROPY_CSV)
        entropy_lookup = get_entropy(ent_path)

        if args.report_only:
            for lam in lam_list:
                for alpha in alpha_list:
                    b_em, r_em = run_model_report_only(
                        model_name, cfg, lam, alpha, args.split,
                        entropy_lookup, tmpl_df, max_cands=args.max_cands)
                    if b_em is not None:
                        summary[(model_name, lam, alpha)] = (b_em, r_em)
        else:
            lib_path = args.lib_path or cfg.get('lib_pkl', LIB_PKL)
            reranker = get_reranker(lib_path)
            for lam in lam_list:
                for alpha in alpha_list:
                    b_em, r_em = run_model(model_name, cfg, lam, alpha, args.split,
                                           reranker, entropy_lookup, tmpl_df,
                                           max_cands=args.max_cands)
                    summary[(model_name, lam, alpha)] = (b_em, r_em)

    # ── overall summary ───────────────────────────────────────────────────────
    print(f'\n{"="*60}')
    print(f'  Summary  |  λ={lam_list}  α={alpha_list}')
    print(f'{"="*60}')
    print(f'  {"Model":<22}  {"λ":>5}  {"α":>5}  {"Beam EM@1":>10}  {"Rrnk EM@1":>10}  {"ΔEM@1":>8}')
    print(f'  {"─"*65}')
    for (name, lam, alpha), (b, r) in summary.items():
        print(f'  {name:<22}  {lam:>5.1f}  {alpha:>5.2f}  {b:>10.4f}  {r:>10.4f}  {r-b:>+8.4f}')

    print('\nDone.')


if __name__ == '__main__':
    main()
