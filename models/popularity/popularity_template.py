"""
popularity.py
=============
Popularity (frequency) baseline for reaction condition prediction.

Concept
-------
No model inference. For each test reaction, look up its extracted template in
the condition library built from train+val. Return the most frequent conditions
seen with that template during training, ranked by count.

  r5 → r4 → r3 → r2 → r1 → r0  (most specific first, first match wins)
  Fallback: global most-popular conditions when no template matches.

Usage
-----
  cd models/popularity
  python popularity_template.py
  python popularity_template.py --top_k 10 --no_global_fallback
"""

import os
import sys
import math
import pickle
import argparse
import numpy as np
import pandas as pd
from collections import Counter

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.normpath(os.path.join(BASE_DIR, "../../Data"))
TAR_DIR = os.path.normpath(os.path.join(BASE_DIR, "../../tar"))
if TAR_DIR not in sys.path:
    sys.path.insert(0, TAR_DIR)

LIB_PKL  = os.path.join(DATA_DIR, "standard_template_library.pkl")
TEST_CSV  = os.path.join(DATA_DIR, "data_test_template.csv")

RADII  = ("r5", "r4", "r3", "r2", "r1", "r0")   # most specific first
KS     = [1, 3, 5, 10]
NULL   = "none"
NONE_SET = {"None", "nan", "", "none", "NaN"}


# ── helpers ──────────────────────────────────────────────────────────────────

def parse_field(val):
    """Semicolon-separated SMILES field → sorted tuple."""
    if pd.isna(val) or str(val).strip() == "":
        return ()
    return tuple(sorted(s.strip() for s in str(val).split(";") if s.strip()))


def flat_set(cat, solv, reag):
    """(cat_tuple, solv_tuple, reag_tuple) → frozenset of all SMILES."""
    return frozenset(x for group in (cat, solv, reag) for x in group)


def set_metrics(gt, pred):
    if not gt and not pred:
        return 1.0, 1.0, 1.0, 1.0
    tp = len(gt & pred); fp = len(pred - gt); fn = len(gt - pred)
    p  = tp / len(pred) if pred else (1.0 if not gt else 0.0)
    r  = tp / len(gt)   if gt   else (1.0 if not pred else 0.0)
    f  = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    j  = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 1.0
    return p, r, f, j


# ── load library ─────────────────────────────────────────────────────────────

def load_library(pkl_path):
    """
    Load pre-built standard_template_library.pkl.

    Structure inside the pkl (TemplateReranker object):
      ._libs[radius][template_str] = Counter{ (cat_tuple, solv_tuple, reag_tuple): count }

    Also builds a global_counter by summing all condition counts across all
    templates at r0 (most general — covers all training conditions).
    """
    print(f"  Loading {pkl_path}")
    with open(pkl_path, "rb") as f:
        reranker = pickle.load(f)
    lib = reranker._libs   # dict[r] -> dict[tmpl] -> Counter

    for r in RADII:
        print(f"  {r}: {len(lib[r]):>8,} unique templates")

    # Build global counter from all conditions in the library (using r0 = most general)
    global_counter = Counter()
    for bucket in lib["r0"].values():
        for key, cnt in bucket.items():
            global_counter[key] += cnt
    print(f"  global: {len(global_counter):,} unique condition sets")

    return lib, global_counter


# ── prediction ────────────────────────────────────────────────────────────────

def predict_one(row, lib, global_topk, top_k, no_global_fallback):
    """
    Accumulate candidates across radii r5→r0 until top_k is reached.
    radius_filled: the last radius that was needed to reach top_k (or None for global).
    """
    candidates  = []
    seen_keys   = set()
    radius_filled = None   # deepest radius needed to fill top_k

    for r in RADII:
        if len(candidates) >= top_k:
            break
        tmpl = row.get(f"template_{r}")
        if pd.isna(tmpl) or not str(tmpl).strip():
            continue
        bucket = lib[r].get(str(tmpl).strip())
        if bucket is None:
            continue
        for key, cnt in bucket.most_common():
            fset = flat_set(*key)
            if fset not in seen_keys:
                candidates.append((fset, cnt))
                seen_keys.add(fset)
                if len(candidates) >= top_k:
                    break
        radius_filled = r   # updated each time a radius contributes

    # Global fallback if still not enough
    if len(candidates) < top_k and not no_global_fallback:
        for key, cnt in global_topk:
            if len(candidates) >= top_k:
                break
            fset = flat_set(*key)
            if fset not in seen_keys:
                candidates.append((fset, cnt))
                seen_keys.add(fset)
        radius_filled = None   # None = needed global fallback

    return candidates, radius_filled


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate(test_df, all_cands, all_radius):
    N = len(test_df)

    hit_EM = {k: np.zeros(N, dtype=bool) for k in KS}
    best_R = {k: np.zeros(N) for k in KS}
    best_P = {k: np.zeros(N) for k in KS}
    best_F = {k: np.zeros(N) for k in KS}
    best_J = {k: np.zeros(N) for k in KS}
    em1 = np.zeros(N, dtype=bool)
    p1  = np.zeros(N); r1 = np.zeros(N)
    f1  = np.zeros(N); j1 = np.zeros(N)

    prev_k = 0
    for k_idx, k in enumerate(KS):
        if k_idx > 0:
            pk = KS[k_idx - 1]
            hit_EM[k] = hit_EM[pk].copy()
            best_R[k]  = best_R[pk].copy()
            best_P[k]  = best_P[pk].copy()
            best_F[k]  = best_F[pk].copy()
            best_J[k]  = best_J[pk].copy()
        for pos in range(prev_k, k):
            for i, row in enumerate(test_df.itertuples(index=False)):
                if pos >= len(all_cands[i]):
                    continue
                pred_s = all_cands[i][pos][0]
                gt_s   = flat_set(parse_field(row.Catalyst),
                                  parse_field(row.Solvent),
                                  parse_field(row.Agent))
                p, r, f, jac = set_metrics(gt_s, pred_s)
                if p   > best_P[k][i]: best_P[k][i] = p
                if r   > best_R[k][i]: best_R[k][i] = r
                if f   > best_F[k][i]: best_F[k][i] = f
                if jac > best_J[k][i]: best_J[k][i] = jac
                if pred_s == gt_s:     hit_EM[k][i]  = True
        prev_k = k

    for i, row in enumerate(test_df.itertuples(index=False)):
        gt_s = flat_set(parse_field(row.Catalyst),
                        parse_field(row.Solvent),
                        parse_field(row.Agent))
        if all_cands[i]:
            pred_s = all_cands[i][0][0]
            p, r, f, jac = set_metrics(gt_s, pred_s)
            em1[i] = (pred_s == gt_s)
            p1[i] = p; r1[i] = r; f1[i] = f; j1[i] = jac

    return dict(hit_EM=hit_EM, best_R=best_R, best_P=best_P,
                best_F=best_F, best_J=best_J,
                EM1=em1, P1=p1, R1=r1, F1=f1, J1=j1)


# ── report ────────────────────────────────────────────────────────────────────

def build_report(m, N, radius_stats):
    lines = []
    def h(t):
        lines.append("\n" + "═" * 74)
        lines.append(f"  {t}")
        lines.append("═" * 74)

    h(f"Popularity Baseline  |  N={N}")

    h("0. Template Radius Usage  (deepest radius needed to fill top-k)")
    lines.append(f"  {'Radius':<16} {'Count':>8}  {'Rate':>8}  {'Cumulative':>12}")
    cumul = 0
    for r in list(RADII) + [None]:
        c = radius_stats.get(r, 0)
        cumul += c
        label = f"until {r}" if r is not None else "global fallback"
        lines.append(f"  {label:<16} {c:>8}  {c/N:>8.3f}  {cumul:>8} ({100*cumul/N:.1f}%)")
    lines.append(f"\n  Global fallback needed: {radius_stats.get(None,0)} "
                 f"({100*radius_stats.get(None,0)/N:.1f}%)")

    h("1. Full-Set Top-k EM and Recall")
    lines.append(f"  {'k':<6}  {'EM':>10}  {'Recall':>10}  {'Precision':>10}  {'F1':>10}")
    lines.append("  " + "─" * 54)
    for k in KS:
        lines.append(f"  top-{k:<3}  "
                     f"{m['hit_EM'][k].mean():>10.4f}  "
                     f"{m['best_R'][k].mean():>10.4f}  "
                     f"{m['best_P'][k].mean():>10.4f}  "
                     f"{m['best_F'][k].mean():>10.4f}")

    h("2. Top-1 Full-Set Metrics")
    lines.append(f"  {'Metric':<14}  {'Value':>10}")
    lines.append("  " + "─" * 28)
    for name, arr in [("Precision",   m["P1"]),
                      ("Recall",      m["R1"]),
                      ("F1",          m["F1"]),
                      ("Jaccard",     m["J1"]),
                      ("Exact Match", m["EM1"].astype(float))]:
        lines.append(f"  {name:<14}  {arr.mean():>10.4f}")

    return "\n".join(lines)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top_k",              type=int, default=10)
    ap.add_argument("--no_global_fallback", action="store_true",
                    help="Do not fill with global popular conditions when no template matches")
    ap.add_argument("--suffix",             default="",
                    help="Suffix appended to output filenames (e.g. '_top30')")
    args = ap.parse_args()

    out_csv    = os.path.join(BASE_DIR, f"predictions_popularity_test{args.suffix}.csv")
    out_report = os.path.join(BASE_DIR, f"result_popularity{args.suffix}.txt")

    print("Loading template library ...")
    lib, global_counter = load_library(LIB_PKL)
    global_topk = global_counter.most_common(args.top_k)

    print(f"\nLoading test set: {TEST_CSV}")
    test_df = pd.read_csv(TEST_CSV, usecols=["row_id", "SMILES", "Catalyst",
                                              "Solvent", "Agent"]
                                            + [f"template_{r}" for r in RADII],
                          low_memory=False)
    N = len(test_df)
    print(f"  {N:,} test samples")

    print("\nPredicting ...")
    all_cands  = []
    all_radius = []
    radius_stats = Counter()

    for _, row in test_df.iterrows():
        cands, r_used = predict_one(row, lib, global_topk,
                                    args.top_k, args.no_global_fallback)
        all_cands.append(cands)
        all_radius.append(r_used)
        radius_stats[r_used] += 1

    print("Evaluating ...")
    m = evaluate(test_df, all_cands, all_radius)

    report = build_report(m, N, radius_stats)
    print(report)
    with open(out_report, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport → {out_report}")

    # Save CSV
    records = []
    for i, (row, cands) in enumerate(zip(test_df.itertuples(index=False), all_cands)):
        rec = {"row_id": row.row_id, "SMILES": row.SMILES,
               "radius_used": str(all_radius[i])}
        for k, (fset, score) in enumerate(cands, 1):
            rec[f"cand{k}"]  = ";".join(sorted(fset)) or NULL
            rec[f"score{k}"] = score
        records.append(rec)
    pd.DataFrame(records).to_csv(out_csv, index=False)
    print(f"Predictions → {out_csv}")


if __name__ == "__main__":
    main()
