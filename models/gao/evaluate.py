"""
evaluate_gao2018.py
===================
Beam search inference (Phase 1) for the Gao 2018 autoregressive model.

Template re-ranking (Phase 2, including entropy-adaptive λ) now lives in
tar/rerank_all.py, which consumes the beam CSV produced here. This script
only runs inference and reports beam-only metrics.

Steps:
  1. Load gao2018/best_model.pt
  2. Beam search over the test set
     → saves gao2018/inference_results_beam{N}.csv
  3. Beam-only metrics report

Usage:
    cd models/gao
    python evaluate.py
    python evaluate.py --beam_width 20
    python evaluate.py --fp_size 2048   # if model was trained with --fp_size 2048
    python evaluate.py --skip_inference # reuse existing inference CSV, just report metrics

Outputs:
    gao2018/inference_results_beam{N}.csv  — top-k predictions + beam scores
    gao2018/beam_metrics_beam{N}.txt       — beam-only metrics report
"""

import os
import sys
import argparse
import multiprocessing
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.normpath(os.path.join(BASE_DIR, "../.."))
DATA_DIR   = os.path.normpath(os.path.join(BASE_DIR, "../../Data"))
SAVE_DIR   = os.path.join(BASE_DIR, "gao2018")
LABEL_DIR  = os.path.normpath(os.path.join(BASE_DIR, "../../Data/labels"))
GCN_TEST_CSV   = os.path.join(DATA_DIR, "MPNN_data/GCN_data_test.csv")
MODEL_PATH     = os.path.join(SAVE_DIR, "best_model.pt")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
FP_RADIUS    = 2
BACKBONE_DIM = 1000
HEAD_HIDDEN  = 300
DROPOUT      = 0.5
TOPK         = [1, 3, 5, 10]
NULL         = "none"

ROLES = ["Catalyst", "Solvent1", "Solvent2", "Agent1", "Agent2"]
# Positions in the compact 'cat;s1;s2;r1;r2' format
GROUPS = {
    "Catalyst": [0],
    "Solvent":  [1, 2],
    "Agent":    [3, 4],
}


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Model definition (duplicated from train_baseline_gao2018.py)
# ─────────────────────────────────────────────────────────────────────────────

class PredictionHead(nn.Module):
    def __init__(self, in_dim, vocab_size, hidden=HEAD_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, vocab_size),
        )

    def forward(self, x):
        return self.net(x)


class Gao2018Model(nn.Module):
    def __init__(self, input_dim, cat_dim, solv_dim, reag_dim,
                 backbone_dim=BACKBONE_DIM, dense_dim=HEAD_HIDDEN, dropout=DROPOUT):
        super().__init__()
        self.cat_dim  = cat_dim
        self.solv_dim = solv_dim
        self.reag_dim = reag_dim

        # Matches train_baseline_gao2018.py: 4 FC layers, Dropout after both
        # 1000-dim layers, final Tanh(300) output (Dense FP).
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, backbone_dim),    # FC1: 32768→1000  [0]
            nn.ReLU(),                              #                  [1]
            nn.Linear(backbone_dim, backbone_dim), # FC2: 1000→1000   [2]
            nn.ReLU(),                              #                  [3]
            nn.Dropout(dropout),                   # after both 1000s [4]
            nn.Linear(backbone_dim, dense_dim),    # FC3: 1000→300    [5]
            nn.ReLU(),                              #                  [6]
            nn.Linear(dense_dim, dense_dim),       # FC4: 300→300     [7]
            nn.Tanh(),                             # Dense FP [B,300] [8]
        )

        D = dense_dim   # 300 — backbone output dimension
        self.cat_head   = PredictionHead(D,                                  cat_dim)
        self.solv1_head = PredictionHead(D + cat_dim,                        solv_dim)
        self.solv2_head = PredictionHead(D + cat_dim + solv_dim,             solv_dim)
        self.reag1_head = PredictionHead(D + cat_dim + 2*solv_dim,           reag_dim)
        self.reag2_head = PredictionHead(D + cat_dim + 2*solv_dim + reag_dim, reag_dim)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Fingerprint
# ─────────────────────────────────────────────────────────────────────────────

def rxn_to_fp(rxn_smiles, nbits, radius):
    try:
        parts = rxn_smiles.strip().split('>>')
        r_smi, p_smi = parts[0].strip(), parts[-1].strip()
        r_mol = Chem.MolFromSmiles(r_smi)
        p_mol = Chem.MolFromSmiles(p_smi)
        if r_mol is None or p_mol is None:
            return None
        fp_r = np.array(AllChem.GetMorganFingerprintAsBitVect(r_mol, radius, nBits=nbits), dtype=np.float32)
        fp_p = np.array(AllChem.GetMorganFingerprintAsBitVect(p_mol, radius, nBits=nbits), dtype=np.float32)
        return np.concatenate([fp_p, fp_p - fp_r])
    except Exception:
        return None


# ── parallel worker (must be top-level for multiprocessing on Windows) ──────
_FP_NBITS  = None
_FP_RADIUS = None

def _init_worker(nbits, radius):
    global _FP_NBITS, _FP_RADIUS
    _FP_NBITS  = nbits
    _FP_RADIUS = radius

def _worker_fn(smi):
    return rxn_to_fp(smi, _FP_NBITS, _FP_RADIUS)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Beam search (per-sample)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def beam_search_sample(model, dense, beam_width, device):
    """
    Beam search for one sample.
    dense : [1, backbone_dim]
    Returns list of beam_width tuples:
        (score, cat_id, s1_id, s2_id, r1_id, r2_id)
    sorted best-first.
    """
    bw = beam_width

    # ── Step 1: Catalyst ────────────────────────────────────────────────────
    cat_lp = F.log_softmax(model.cat_head(dense)[0], dim=0)  # [cat_dim]
    k1 = min(bw, model.cat_dim)
    top_v, top_i = cat_lp.topk(k1)
    beams = [(top_v[j].item(), top_i[j].item()) for j in range(k1)]

    # ── Step 2: Solvent 1 ────────────────────────────────────────────────────
    B = len(beams)
    cat_t  = torch.tensor([b[1] for b in beams], device=device, dtype=torch.long)
    cat_c  = F.one_hot(cat_t, model.cat_dim).float()
    s1_lp  = F.log_softmax(
        model.solv1_head(torch.cat([dense.expand(B, -1), cat_c], dim=-1)), dim=-1)

    new_beams = []
    for bi, (score, cat_id) in enumerate(beams):
        k2 = min(bw, model.solv_dim)
        tv, ti = s1_lp[bi].topk(k2)
        for j in range(k2):
            new_beams.append((score + tv[j].item(), cat_id, ti[j].item()))
    new_beams.sort(key=lambda x: -x[0])
    beams = new_beams[:bw]

    # ── Step 3: Solvent 2 ────────────────────────────────────────────────────
    B = len(beams)
    cat_t = torch.tensor([b[1] for b in beams], device=device, dtype=torch.long)
    s1_t  = torch.tensor([b[2] for b in beams], device=device, dtype=torch.long)
    cat_c = F.one_hot(cat_t, model.cat_dim).float()
    s1_c  = F.one_hot(s1_t,  model.solv_dim).float()
    s2_lp = F.log_softmax(
        model.solv2_head(torch.cat([dense.expand(B, -1), cat_c, s1_c], dim=-1)), dim=-1)

    new_beams = []
    for bi, (score, cat_id, s1_id) in enumerate(beams):
        k3 = min(bw, model.solv_dim)
        tv, ti = s2_lp[bi].topk(k3)
        for j in range(k3):
            new_beams.append((score + tv[j].item(), cat_id, s1_id, ti[j].item()))
    new_beams.sort(key=lambda x: -x[0])
    beams = new_beams[:bw]

    # ── Step 4: Reagent 1 ────────────────────────────────────────────────────
    B = len(beams)
    cat_t = torch.tensor([b[1] for b in beams], device=device, dtype=torch.long)
    s1_t  = torch.tensor([b[2] for b in beams], device=device, dtype=torch.long)
    s2_t  = torch.tensor([b[3] for b in beams], device=device, dtype=torch.long)
    cat_c = F.one_hot(cat_t, model.cat_dim).float()
    s1_c  = F.one_hot(s1_t,  model.solv_dim).float()
    s2_c  = F.one_hot(s2_t,  model.solv_dim).float()
    r1_lp = F.log_softmax(
        model.reag1_head(torch.cat([dense.expand(B, -1), cat_c, s1_c, s2_c], dim=-1)), dim=-1)

    new_beams = []
    for bi, (score, cat_id, s1_id, s2_id) in enumerate(beams):
        k4 = min(bw, model.reag_dim)
        tv, ti = r1_lp[bi].topk(k4)
        for j in range(k4):
            new_beams.append((score + tv[j].item(), cat_id, s1_id, s2_id, ti[j].item()))
    new_beams.sort(key=lambda x: -x[0])
    beams = new_beams[:bw]

    # ── Step 5: Reagent 2 ────────────────────────────────────────────────────
    B = len(beams)
    cat_t = torch.tensor([b[1] for b in beams], device=device, dtype=torch.long)
    s1_t  = torch.tensor([b[2] for b in beams], device=device, dtype=torch.long)
    s2_t  = torch.tensor([b[3] for b in beams], device=device, dtype=torch.long)
    r1_t  = torch.tensor([b[4] for b in beams], device=device, dtype=torch.long)
    cat_c = F.one_hot(cat_t, model.cat_dim).float()
    s1_c  = F.one_hot(s1_t,  model.solv_dim).float()
    s2_c  = F.one_hot(s2_t,  model.solv_dim).float()
    r1_c  = F.one_hot(r1_t,  model.reag_dim).float()
    r2_lp = F.log_softmax(
        model.reag2_head(torch.cat([dense.expand(B, -1), cat_c, s1_c, s2_c, r1_c], dim=-1)), dim=-1)

    new_beams = []
    for bi, (score, cat_id, s1_id, s2_id, r1_id) in enumerate(beams):
        k5 = min(bw, model.reag_dim)
        tv, ti = r2_lp[bi].topk(k5)
        for j in range(k5):
            new_beams.append((score + tv[j].item(), cat_id, s1_id, s2_id, r1_id, ti[j].item()))
    new_beams.sort(key=lambda x: -x[0])
    return new_beams[:bw]


# ─────────────────────────────────────────────────────────────────────────────
# 3b. Vectorized beam search — N reactions simultaneously
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def batch_beam_search_vectorized(model, dense_chunk, beam_width, device):
    """
    Vectorized beam search for N reactions at once.
    dense_chunk : [N, D]  — backbone output already on `device`
    Returns     : list[N] of list[beam_width] of
                  (score, cat_id, s1_id, s2_id, r1_id, r2_id)  sorted best-first
    """
    N, D = dense_chunk.shape
    bw   = beam_width

    # ── Step 1: Catalyst ────────────────────────────────────────────────────
    cat_lp = F.log_softmax(model.cat_head(dense_chunk), dim=-1)   # [N, cat_dim]
    k1     = min(bw, model.cat_dim)
    beam_scores, beam_cat = cat_lp.topk(k1, dim=1)                # [N, bw]

    # Expand dense for all beams once: [N*bw, D]
    dense_exp = (dense_chunk
                 .unsqueeze(1)
                 .expand(N, bw, D)
                 .reshape(N * bw, D))

    def step(head, vocab_dim, prev_ids):
        """
        prev_ids : list of (ids [N, bw], onehot_dim)
        Returns  : (new_beam_scores [N, bw], parent [N, bw], new_ids [N, bw])
        """
        cond = [dense_exp]
        for (ids, dim) in prev_ids:
            cond.append(F.one_hot(ids.reshape(N * bw), dim).float())
        lp   = F.log_softmax(head(torch.cat(cond, dim=-1)), dim=-1)   # [N*bw, vocab]
        comb = (beam_scores.reshape(N * bw, 1) + lp).reshape(N, bw * vocab_dim)
        new_sc, flat_idx = comb.topk(bw, dim=1)   # [N, bw]
        parent  = flat_idx // vocab_dim
        new_ids = flat_idx  % vocab_dim
        return new_sc, parent, new_ids

    # ── Step 2: Solvent 1 ────────────────────────────────────────────────────
    new_sc, par, beam_s1 = step(model.solv1_head, model.solv_dim,
                                 [(beam_cat, model.cat_dim)])
    beam_cat    = beam_cat.gather(1, par)
    beam_scores = new_sc

    # ── Step 3: Solvent 2 ────────────────────────────────────────────────────
    new_sc, par, beam_s2 = step(model.solv2_head, model.solv_dim,
                                 [(beam_cat, model.cat_dim),
                                  (beam_s1,  model.solv_dim)])
    beam_cat    = beam_cat.gather(1, par)
    beam_s1     = beam_s1.gather(1,  par)
    beam_scores = new_sc

    # ── Step 4: Reagent 1 ────────────────────────────────────────────────────
    new_sc, par, beam_r1 = step(model.reag1_head, model.reag_dim,
                                 [(beam_cat, model.cat_dim),
                                  (beam_s1,  model.solv_dim),
                                  (beam_s2,  model.solv_dim)])
    beam_cat    = beam_cat.gather(1, par)
    beam_s1     = beam_s1.gather(1,  par)
    beam_s2     = beam_s2.gather(1,  par)
    beam_scores = new_sc

    # ── Step 5: Reagent 2 ────────────────────────────────────────────────────
    new_sc, par, beam_r2 = step(model.reag2_head, model.reag_dim,
                                 [(beam_cat, model.cat_dim),
                                  (beam_s1,  model.solv_dim),
                                  (beam_s2,  model.solv_dim),
                                  (beam_r1,  model.reag_dim)])
    beam_cat    = beam_cat.gather(1, par)
    beam_s1     = beam_s1.gather(1,  par)
    beam_s2     = beam_s2.gather(1,  par)
    beam_r1     = beam_r1.gather(1,  par)
    beam_scores = new_sc
    beam_r2     = beam_r2

    # ── Move to CPU and build output list ────────────────────────────────────
    sc = beam_scores.cpu().numpy()
    ct = beam_cat.cpu().numpy()
    s1 = beam_s1.cpu().numpy()
    s2 = beam_s2.cpu().numpy()
    r1 = beam_r1.cpu().numpy()
    r2 = beam_r2.cpu().numpy()

    return [
        [(float(sc[i, j]), int(ct[i, j]), int(s1[i, j]),
          int(s2[i, j]), int(r1[i, j]), int(r2[i, j]))
         for j in range(bw)]
        for i in range(N)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Inference over test set → CSV
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(model, fp_size, device, beam_width,
                  gcn_csv=None, out_csv=None, chunk=16384):
    """
    Runs beam search on the given GCN data CSV (default: GCN_data_test.csv).
    Returns (results_df, gcn_df) where results_df has the standard
    cand{j} / score{j} / gt columns.
    """
    if gcn_csv is None:
        gcn_csv = GCN_TEST_CSV
    if out_csv is None:
        out_csv = os.path.join(SAVE_DIR, f'inference_results_beam{beam_width}.csv')
    # ── Label index → SMILES ──────────────────────────────────────────────────
    cat_df   = pd.read_csv(os.path.join(LABEL_DIR, 'cat_labels.csv'))
    solv_df  = pd.read_csv(os.path.join(LABEL_DIR, 'solv_labels.csv'))
    reag_df  = pd.read_csv(os.path.join(LABEL_DIR, 'reag_labels.csv'))

    cat_id2label  = cat_df['cat'].fillna(NULL).tolist()
    solv_id2label = solv_df['solv'].fillna(NULL).tolist()
    reag_id2label = reag_df['reag'].fillna(NULL).tolist()

    def idx2label(idx, mapping):
        if idx < 0 or idx >= len(mapping):
            return NULL
        v = mapping[idx]
        return NULL if (v == NULL or v != v) else v   # nan check

    # ── Load data CSV ─────────────────────────────────────────────────────────
    print(f'Loading data from {gcn_csv} ...')
    gcn_df = pd.read_csv(gcn_csv)
    print(f'  {len(gcn_df):,} samples')

    # ── Compute fingerprints (parallel) ──────────────────────────────────────
    n_workers = min(multiprocessing.cpu_count(), 96)
    print(f'Computing fingerprints (fp_size={fp_size}, radius={FP_RADIUS}, workers={n_workers}) ...')
    smiles_list = gcn_df['reaction'].astype(str).tolist()
    with multiprocessing.Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(fp_size, FP_RADIUS),
    ) as pool:
        fp_results = pool.map(_worker_fn, smiles_list, chunksize=1024)

    fps = []
    for i, fp in enumerate(fp_results):
        if fp is None:
            fps.append(np.zeros(2 * fp_size, dtype=np.float32))
            print(f'  [WARN] FP failed for row {i}')
        else:
            fps.append(fp)

    X = torch.from_numpy(np.stack(fps))   # [N, 2*fp_size]  kept on CPU
    N = len(X)
    print(f'  {N:,} samples ready')

    # ── Beam search (chunked: backbone + beam search + CSV write per chunk) ──────
    CHUNK = chunk
    print(f'Running beam search (beam_width={beam_width}, chunk={CHUNK}) ...')
    model.eval()
    first_chunk = True

    for chunk_start in range(0, N, CHUNK):
        chunk_end = min(chunk_start + CHUNK, N)
        X_chunk = X[chunk_start:chunk_end].to(device)   # [C, 2*fp_size]  moved to GPU here

        with torch.no_grad():
            dense_chunk = model.backbone(X_chunk)        # [C, D]
            all_beams   = batch_beam_search_vectorized(
                              model, dense_chunk, beam_width, device)

        records = []
        for ci, beams in enumerate(all_beams):
            i = chunk_start + ci

            row = gcn_df.iloc[i]
            rec = {}

            # Ground truth (compact: cat;s1;s2;r1;r2)
            rec['gt'] = ";".join([
                idx2label(int(row['cat']),   cat_id2label),
                idx2label(int(row['solv0']), solv_id2label),
                idx2label(int(row['solv1']), solv_id2label),
                idx2label(int(row['reag0']), reag_id2label),
                idx2label(int(row['reag1']), reag_id2label),
            ])

            # Top-k candidates (compact: cat;s1;s2;r1;r2) + score
            for j, (score, cat_id, s1_id, s2_id, r1_id, r2_id) in enumerate(beams, 1):
                rec[f'cand{j}'] = ";".join([
                    idx2label(cat_id, cat_id2label),
                    idx2label(s1_id,  solv_id2label),
                    idx2label(s2_id,  solv_id2label),
                    idx2label(r1_id,  reag_id2label),
                    idx2label(r2_id,  reag_id2label),
                ])
                rec[f'score{j}'] = round(score, 6)

            records.append(rec)

        # Flush chunk to CSV
        pd.DataFrame(records).to_csv(
            out_csv,
            mode='w' if first_chunk else 'a',
            header=first_chunk,
            index=False,
        )
        first_chunk = False
        print(f'  {chunk_end:,}/{N:,}', end='\r')

    print()
    print(f'Saved -> {out_csv}')
    results_df = pd.read_csv(out_csv)
    return results_df, gcn_df


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Candidate helpers  (compact format: 'cat;s1;s2;r1;r2')
# ─────────────────────────────────────────────────────────────────────────────

def parse_cand(s):
    """Parse 'cat;s1;s2;r1;r2' into a list of 5 values."""
    if pd.isna(s) or str(s).strip() == "":
        return [NULL] * 5
    parts = str(s).split(";")
    while len(parts) < 5:
        parts.append(NULL)
    return [p.strip() or NULL for p in parts[:5]]


def candidate_set(row, j):
    return {v for v in parse_cand(row.get(f"cand{j}", "")) if v != NULL}


def gt_set_full(row):
    return {v for v in parse_cand(row.get("gt", "")) if v != NULL}


def gt_set_group(row, pos_indices):
    parts = parse_cand(row.get("gt", ""))
    return {parts[i] for i in pos_indices if parts[i] != NULL}


def cand_set_group(row, pos_indices, j):
    parts = parse_cand(row.get(f"cand{j}", ""))
    return {parts[i] for i in pos_indices if parts[i] != NULL}


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Fast metrics — pre-parse once, then vectorize with numpy
# ─────────────────────────────────────────────────────────────────────────────

def set_metrics(gt, pred):
    if len(gt) == 0 and len(pred) == 0:
        return 1.0, 1.0, 1.0, 1.0
    tp = len(gt & pred); fp = len(pred - gt); fn = len(gt - pred)
    p = tp / len(pred) if len(pred) > 0 else (1.0 if len(gt) == 0 else 0.0)
    r = tp / len(gt)   if len(gt)  > 0 else (1.0 if len(pred) == 0 else 0.0)
    f = 2*p*r / (p+r)  if (p+r) > 0 else 0.0
    j = tp / (tp+fp+fn) if (tp+fp+fn) > 0 else 1.0
    return p, r, f, j


def preparse_all(rows_list, n_cands):
    """
    Parse all GT and candidate strings into frozensets once.
    Returns:
        gt_full   : list[N] of frozenset
        cand_full : list[N] of list[n_cands] of frozenset
        gt_group  : dict[gname] -> list[N] of frozenset
        cand_group: dict[gname] -> list[N] of list[n_cands] of frozenset
    """
    N = len(rows_list)
    gt_full   = []
    cand_full = []
    gt_group  = {g: [] for g in GROUPS}
    cand_group= {g: [] for g in GROUPS}

    for row in rows_list:
        gt_parts = parse_cand(row.get("gt", ""))
        gt_full.append(frozenset(v for v in gt_parts if v != NULL))
        for gname, groles in GROUPS.items():
            gt_group[gname].append(frozenset(gt_parts[idx] for idx in groles if gt_parts[idx] != NULL))

        row_cands = []
        row_cands_g = {g: [] for g in GROUPS}
        for j in range(1, n_cands + 1):
            cparts = parse_cand(row.get(f"cand{j}", ""))
            row_cands.append(frozenset(v for v in cparts if v != NULL))
            for gname, groles in GROUPS.items():
                row_cands_g[gname].append(frozenset(cparts[idx] for idx in groles if cparts[idx] != NULL))

        cand_full.append(row_cands)
        for gname in GROUPS:
            cand_group[gname].append(row_cands_g[gname])

    return gt_full, cand_full, gt_group, cand_group


def precompute_pairwise(gt_full, cand_full, gt_group, cand_group):
    """
    Compute tp / pred_size / exact-match for every (sample i, candidate j).
    Returns numpy arrays of shape [N, n_cands].
    All set ops are on small frozensets (≤5 elements) — fast Python.
    """
    N       = len(gt_full)
    n_cands = len(cand_full[0])

    gt_size = np.array([len(g) for g in gt_full], dtype=np.int32)
    tp_full = np.empty((N, n_cands), dtype=np.int32)
    ps_full = np.empty((N, n_cands), dtype=np.int32)
    em_full = np.empty((N, n_cands), dtype=bool)

    for i in range(N):
        gt = gt_full[i]
        for j in range(n_cands):
            pred = cand_full[i][j]
            tp_full[i, j] = len(gt & pred)
            ps_full[i, j] = len(pred)
            em_full[i, j] = (gt == pred)

    # Group arrays
    tp_group  = {}; ps_group  = {}; em_group  = {}
    gt_size_g = {}; g_pos     = {}

    for gname, groles in GROUPS.items():
        gg_size = np.array([len(gt_group[gname][i]) for i in range(N)], dtype=np.int32)
        g_tp = np.empty((N, n_cands), dtype=np.int32)
        g_ps = np.empty((N, n_cands), dtype=np.int32)
        g_em = np.empty((N, n_cands), dtype=bool)
        for i in range(N):
            gt_g = gt_group[gname][i]
            for j in range(n_cands):
                pred_g = cand_group[gname][i][j]
                g_tp[i, j] = len(gt_g & pred_g)
                g_ps[i, j] = len(pred_g)
                g_em[i, j] = (gt_g == pred_g)
        tp_group[gname]  = g_tp
        ps_group[gname]  = g_ps
        em_group[gname]  = g_em
        gt_size_g[gname] = gg_size
        g_pos[gname]     = gg_size > 0

    return gt_size, tp_full, ps_full, em_full, tp_group, ps_group, em_group, gt_size_g, g_pos


def _build_rank_idx(ordered_ranks, N, n_positions):
    """Build [N, n_positions] 0-indexed rank index array + valid mask."""
    rank_idx   = np.zeros((N, n_positions), dtype=np.int32)
    valid_mask = np.zeros((N, n_positions), dtype=bool)
    for i, ranks in enumerate(ordered_ranks):
        for pos, j in enumerate(ranks[:n_positions]):
            rank_idx[i, pos]   = j - 1   # 0-indexed
            valid_mask[i, pos] = True
    return rank_idx, valid_mask


def compute_metrics_vectorized(gt_size, tp_full, ps_full, em_full,
                                tp_group, ps_group, em_group, gt_size_g, g_pos,
                                ordered_ranks):
    """
    Fully vectorized metrics using numpy.
    All loops are O(N × n_cands) Python during pre-computation, then O(1) numpy.
    """
    N       = len(gt_size)
    n_cands = tp_full.shape[1]
    n_pos   = max(len(r) for r in ordered_ranks)

    rank_idx, valid_mask = _build_rank_idx(ordered_ranks, N, n_pos)
    row_idx = np.arange(N)[:, None]   # [N, 1]

    # Gather [N, n_pos]
    tp_r = tp_full[row_idx, rank_idx]
    ps_r = ps_full[row_idx, rank_idx]
    em_r = em_full[row_idx, rank_idx] & valid_mask
    gt_a = gt_size[:, None]           # [N, 1]

    # Vectorized P / R / F / J  [N, n_pos]
    with np.errstate(divide='ignore', invalid='ignore'):
        P_mat = np.where(ps_r > 0, tp_r / ps_r,  np.where(gt_a == 0, 1.0, 0.0))
        R_mat = np.where(gt_a  > 0, tp_r / gt_a, np.where(ps_r == 0, 1.0, 0.0))
        dJ    = gt_a + ps_r - tp_r
        J_mat = np.where(dJ > 0, tp_r / dJ, 1.0)
        dF    = P_mat + R_mat
        F_mat = np.where(dF > 0, 2*P_mat*R_mat / dF, 0.0)

    # Mask invalid positions to 0 (so they don't inflate max)
    P_mat = np.where(valid_mask, P_mat, 0.0)
    R_mat = np.where(valid_mask, R_mat, 0.0)
    F_mat = np.where(valid_mask, F_mat, 0.0)
    J_mat = np.where(valid_mask, J_mat, 0.0)

    # Top-1
    P1 = P_mat[:, 0]; R1 = R_mat[:, 0]; F1_ = F_mat[:, 0]; J1 = J_mat[:, 0]
    EM1 = em_r[:, 0]

    # Top-k (cumulative max / any over first k positions)
    best_P_k = {}; best_R_k = {}; best_F_k = {}; best_J_k = {}; hit_EM_k = {}
    for k in TOPK:
        kp = min(k, n_pos)
        best_P_k[k] = P_mat[:, :kp].max(axis=1)
        best_R_k[k] = R_mat[:, :kp].max(axis=1)
        best_F_k[k] = F_mat[:, :kp].max(axis=1)
        best_J_k[k] = J_mat[:, :kp].max(axis=1)
        hit_EM_k[k] = em_r[:, :kp].any(axis=1)

    # Group-level
    group_metrics = {}
    for gname, groles in GROUPS.items():
        g_gt_a  = gt_size_g[gname][:, None]
        g_tp_r  = tp_group[gname][row_idx, rank_idx]
        g_ps_r  = ps_group[gname][row_idx, rank_idx]
        g_em_r  = em_group[gname][row_idx, rank_idx] & valid_mask

        with np.errstate(divide='ignore', invalid='ignore'):
            g_R_mat = np.where(g_gt_a > 0, g_tp_r / g_gt_a,
                               np.where(g_ps_r == 0, 1.0, 0.0))
        g_R_mat = np.where(valid_mask, g_R_mat, 0.0)

        g_best_R = {}; g_hit_EM = {}
        for k in TOPK:
            kp = min(k, n_pos)
            g_best_R[k] = g_R_mat[:, :kp].max(axis=1)
            g_hit_EM[k] = g_em_r[:, :kp].any(axis=1)

        group_metrics[gname] = {
            "best_R":   g_best_R,
            "hit_EM":   g_hit_EM,
            "pos_mask": g_pos[gname],
        }

    return {
        "P1": P1, "R1": R1, "F1": F1_, "J1": J1, "EM1": EM1,
        "gt_sizes":  gt_size,
        "pred_sizes": ps_full[:, 0],
        "best_P_k": best_P_k, "best_R_k": best_R_k,
        "best_F_k": best_F_k, "best_J_k": best_J_k,
        "hit_EM_k": hit_EM_k,
        "group": group_metrics,
    }


# Keep the original compute_metrics as an alias (not used in main, but kept for API compat)
def compute_metrics(df, ordered_ranks):
    N    = len(df)
    rows = df.to_dict("records")
    n_cands = max(len(r) for r in ordered_ranks)
    gt_full, cand_full, gt_group, cand_group = preparse_all(rows, n_cands)
    gt_size, tp_full, ps_full, em_full, tp_group, ps_group, em_group, gt_size_g, g_pos = \
        precompute_pairwise(gt_full, cand_full, gt_group, cand_group)
    return compute_metrics_vectorized(
        gt_size, tp_full, ps_full, em_full,
        tp_group, ps_group, em_group, gt_size_g, g_pos,
        ordered_ranks)


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Report  (beam-only metrics)
# ─────────────────────────────────────────────────────────────────────────────

def print_report(m_beam, N):
    lines = []

    def h(t):
        lines.append("\n" + "=" * 72)
        lines.append(f"  {t}")
        lines.append("=" * 72)

    h(f"Gao 2018 — Beam Search Metrics  |  N={N}")

    h("1. Full Set Top-k EM and Recall (max over candidates 1..k)")
    lines.append(f"  {'k':<6}  {'EM':>10}  {'Recall':>10}")
    lines.append("  " + "─" * 32)
    for k in TOPK:
        em = m_beam["hit_EM_k"][k].mean()
        r  = m_beam["best_R_k"][k].mean()
        mk = "  <- top-1" if k == 1 else ""
        lines.append(f"  top-{k:<3}  {em:>10.4f}  {r:>10.4f}{mk}")

    h("2. Full Set Top-k Precision and F1")
    lines.append(f"  {'k':<6}  {'Precision':>10}  {'F1':>10}")
    lines.append("  " + "─" * 32)
    for k in TOPK:
        p = m_beam["best_P_k"][k].mean()
        f = m_beam["best_F_k"][k].mean()
        lines.append(f"  top-{k:<3}  {p:>10.4f}  {f:>10.4f}")

    h("3. Group-level Recall@k (positive samples only)")
    for gname in GROUPS:
        gb = m_beam["group"][gname]
        n_pos = gb["pos_mask"].sum()
        lines.append(f"\n  Group: {gname}  (n_pos={n_pos})")
        lines.append(f"  {'k':<6}  {'Recall@k':>10}  {'EM@k':>10}")
        lines.append("  " + "─" * 32)
        for k in TOPK:
            pm = gb["pos_mask"]
            r  = gb["best_R"][k][pm].mean() if pm.any() else float("nan")
            em = gb["hit_EM"][k].mean()
            lines.append(f"  top-{k:<3}  {r:>10.4f}  {em:>10.4f}")

    h("4. Top-1 Full Set Metrics")
    lines.append(f"  {'Metric':<14}  {'Value':>10}")
    lines.append("  " + "─" * 28)
    for name, arr in [
        ("Precision",   m_beam["P1"]),
        ("Recall",      m_beam["R1"]),
        ("F1",          m_beam["F1"]),
        ("Jaccard",     m_beam["J1"]),
        ("Exact Match", m_beam["EM1"].astype(float)),
    ]:
        lines.append(f"  {name:<14}  {arr.mean():>10.4f}")

    report = "\n".join(lines)
    print(report)
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Beam search inference for Gao 2018 model")
    parser.add_argument("--fp_size",    type=int,   default=16384,
                        help="Morgan FP bit size (must match training, default: 16384)")
    parser.add_argument("--beam_width", type=int,   default=30,
                        help="Beam width (default: 30)")
    parser.add_argument("--split",      choices=["test", "val"], default="test",
                        help="Dataset split to run (default: test)")
    parser.add_argument("--chunk",      type=int,   default=16384,
                        help="Samples per beam-search chunk (reduce if OOM, default: 16384)")
    parser.add_argument("--skip_inference", action="store_true",
                        help="Skip inference if inference_results_beam{N}.csv already exists")
    args = parser.parse_args()

    os.makedirs(SAVE_DIR, exist_ok=True)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")

    # ── Check model exists ────────────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        print(f"\nERROR: model not found at {MODEL_PATH}")
        print("  Run train_baseline_gao2018.py first to train the model.")
        return

    # ── Load vocab sizes ──────────────────────────────────────────────────────
    cat_df  = pd.read_csv(os.path.join(LABEL_DIR, 'cat_labels.csv'))
    solv_df = pd.read_csv(os.path.join(LABEL_DIR, 'solv_labels.csv'))
    reag_df = pd.read_csv(os.path.join(LABEL_DIR, 'reag_labels.csv'))
    cat_dim, solv_dim, reag_dim = len(cat_df), len(solv_df), len(reag_df)
    print(f"Vocab — cat:{cat_dim}  solv:{solv_dim}  reag:{reag_dim}")

    # ── Load model ────────────────────────────────────────────────────────────
    input_dim = 2 * args.fp_size
    model = Gao2018Model(input_dim, cat_dim, solv_dim, reag_dim).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded {MODEL_PATH}  ({n_params:,} params)")

    # ── Split-dependent paths ─────────────────────────────────────────────────
    split_suffix = f"_beam{args.beam_width}" if args.split == "test" else f"_beam{args.beam_width}_{args.split}"
    inf_csv  = os.path.join(SAVE_DIR, f"inference_results{split_suffix}.csv")
    gcn_csv  = os.path.join(DATA_DIR, f"MPNN_data/GCN_data_{args.split}.csv")
    print(f"Split: {args.split}  |  Data: {gcn_csv}")

    # mirror inference CSV to InferenceResults/{split}/ with canonical name
    IR_DIR = os.path.join(ROOT_DIR, "tar", "InferenceResults", args.split)
    ir_csv = os.path.join(IR_DIR, f"gao_beam{args.beam_width}.csv")

    # ── Inference ─────────────────────────────────────────────────────────────
    if args.skip_inference and os.path.exists(inf_csv):
        print(f"\nSkipping inference — loading existing {inf_csv}")
        results_df = pd.read_csv(inf_csv)
    else:
        print(f"\n--- Beam Search Inference ---")
        results_df, _ = run_inference(model, args.fp_size, DEVICE, args.beam_width,
                                      gcn_csv=gcn_csv, out_csv=inf_csv,
                                      chunk=args.chunk)

    # copy inference CSV to InferenceResults/{split}/
    import shutil
    os.makedirs(IR_DIR, exist_ok=True)
    shutil.copy2(inf_csv, ir_csv)
    print(f"  Copied → {ir_csv}")

    # Fill nulls
    for col in results_df.columns:
        if results_df[col].dtype == object:
            results_df[col] = results_df[col].fillna(NULL).astype(str)

    N = len(results_df)
    n_cands = sum(1 for j in range(1, 200) if f"cand{j}" in results_df.columns)
    print(f"\n{N:,} test samples  |  {n_cands} candidates per sample")

    # ── Pre-parse all candidates once ─────────────────────────────────────────
    print("Pre-parsing candidates ...")
    rows_list = results_df.to_dict("records")
    gt_full, cand_full, gt_group, cand_group = preparse_all(rows_list, n_cands)
    print("Pre-computing pairwise set stats ...")
    gt_size, tp_full, ps_full, em_full, tp_group, ps_group, em_group, gt_size_g, g_pos = \
        precompute_pairwise(gt_full, cand_full, gt_group, cand_group)

    # Beam-only ordering (1..n_cands)
    beam_ranks = [[j for j in range(1, n_cands + 1)] for _ in range(N)]

    print("\n--- Beam Search Metrics ---")
    m_beam = compute_metrics_vectorized(
        gt_size, tp_full, ps_full, em_full,
        tp_group, ps_group, em_group, gt_size_g, g_pos,
        beam_ranks)

    report = print_report(m_beam, N)

    # ── Save report ───────────────────────────────────────────────────────────
    split_tag = f"_{args.split}" if args.split != "test" else ""
    out_txt = os.path.join(SAVE_DIR, f"beam_metrics{split_tag}_beam{args.beam_width}.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved -> {out_txt}")
    print("\nDone.")


if __name__ == "__main__":
    main()
