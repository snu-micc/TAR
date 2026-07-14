"""
popularity_rxnclass.py
======================
Reaction-class-aware popularity baseline.

Library (built from train + val):
  lib_l3["3.2.1"] = Counter{ frozenset: count }   most specific
  lib_l2["3.2"]   = Counter{ frozenset: count }
  lib_l1["3"]     = Counter{ frozenset: count }   coarsest

Prediction for a test sample with reaction class "3.2.1":
  1. lib_l3["3.2.1"] → fill candidates by frequency
  2. lib_l2["3.2"]   → fill remaining slots (deduplicated)
  3. lib_l1["3"]     → fill remaining slots
  4. global fallback → fill remaining slots up to top_k

Saves:
  rxnclass_libraries.pkl           — all 3 level libs + global counter
  predictions_rxnclass_test.csv    — top-k candidates per test sample
  result_rxnclass.txt              — EM / Recall / Precision / F1 report
"""

import os
import sys
import pickle
from collections import Counter

import numpy as np
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(BASE_DIR, "../../Data"))

TRAIN_CSV  = os.path.join(DATA_DIR, "data_train.csv")
VAL_CSV    = os.path.join(DATA_DIR, "data_val.csv")
TEST_CSV   = os.path.join(DATA_DIR, "data_test.csv")

LIB_PKL = os.path.join(BASE_DIR, "rxnclass_libraries.pkl")

KS    = [1, 3, 5, 10]
NULL  = "none"


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_field(val):
    if pd.isna(val) or str(val).strip() == "":
        return ()
    return tuple(sorted(s.strip() for s in str(val).split(";") if s.strip()))


def flat_set(cat, solv, agent):
    return frozenset(x for grp in (cat, solv, agent) for x in grp)


def rxnclass_levels(rxn_class):
    """'3.2.1' → (l1='3', l2='3.2', l3='3.2.1')"""
    parts = str(rxn_class).strip().split(".")
    l1 = parts[0]
    l2 = ".".join(parts[:2]) if len(parts) >= 2 else l1
    l3 = ".".join(parts[:3]) if len(parts) >= 3 else l2
    return l1, l2, l3


# ── library building ──────────────────────────────────────────────────────────

def build_libraries():
    lib_l1 = {}
    lib_l2 = {}
    lib_l3 = {}
    global_counter = Counter()

    for csv_path in [TRAIN_CSV, VAL_CSV]:
        df = pd.read_csv(csv_path,
                         usecols=["Reaction Class", "Catalyst", "Agent", "Solvent"],
                         low_memory=False)
        df = df.rename(columns={"Reaction Class": "rxnclass"})
        print(f"  {os.path.basename(csv_path)}: {len(df):,} rows")

        for row in df.itertuples(index=False):
            rc = row.rxnclass
            if pd.isna(rc) or not str(rc).strip():
                continue

            l1, l2, l3 = rxnclass_levels(rc)
            fset = flat_set(parse_field(row.Catalyst),
                            parse_field(row.Solvent),
                            parse_field(row.Agent))

            for lib, key in [(lib_l1, l1), (lib_l2, l2), (lib_l3, l3)]:
                if key not in lib:
                    lib[key] = Counter()
                lib[key][fset] += 1

            global_counter[fset] += 1

    print(f"\n  lib_l1 (coarse):  {len(lib_l1):>6,} reaction classes")
    print(f"  lib_l2 (mid):     {len(lib_l2):>6,} reaction classes")
    print(f"  lib_l3 (fine):    {len(lib_l3):>6,} reaction classes")
    print(f"  global:           {len(global_counter):>6,} unique condition sets")

    return lib_l1, lib_l2, lib_l3, global_counter


# ── prediction ────────────────────────────────────────────────────────────────

def predict_one(rxn_class, lib_l3, lib_l2, lib_l1, global_topk, top_k):
    if pd.isna(rxn_class) or not str(rxn_class).strip():
        l1 = l2 = l3 = None
    else:
        l1, l2, l3 = rxnclass_levels(rxn_class)

    candidates = []
    seen       = set()
    level_used = None

    for lib, key, lvl in [(lib_l3, l3, "l3"),
                          (lib_l2, l2, "l2"),
                          (lib_l1, l1, "l1")]:
        if len(candidates) >= top_k:
            break
        if key is None or key not in lib:
            continue
        for fset, cnt in lib[key].most_common():
            if fset not in seen:
                candidates.append((fset, cnt))
                seen.add(fset)
                if len(candidates) >= top_k:
                    break
        if candidates and level_used is None:
            level_used = lvl

    # global fallback
    if len(candidates) < top_k:
        for fset, cnt in global_topk:
            if len(candidates) >= top_k:
                break
            if fset not in seen:
                candidates.append((fset, cnt))
                seen.add(fset)
        if level_used is None:
            level_used = "global"

    return candidates, level_used


# ── evaluation ────────────────────────────────────────────────────────────────

def set_metrics(gt, pred):
    if not gt and not pred:
        return 1.0, 1.0, 1.0, 1.0
    tp = len(gt & pred); fp = len(pred - gt); fn = len(gt - pred)
    p  = tp / len(pred) if pred else (1.0 if not gt else 0.0)
    r  = tp / len(gt)   if gt   else (1.0 if not pred else 0.0)
    f  = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    j  = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 1.0
    return p, r, f, j


def evaluate(test_df, all_cands):
    N = len(test_df)
    hit_EM = {k: np.zeros(N, dtype=bool) for k in KS}
    best_R = {k: np.zeros(N) for k in KS}
    best_P = {k: np.zeros(N) for k in KS}
    best_F = {k: np.zeros(N) for k in KS}

    prev_k = 0
    for k_idx, k in enumerate(KS):
        if k_idx > 0:
            pk = KS[k_idx - 1]
            hit_EM[k] = hit_EM[pk].copy()
            best_R[k]  = best_R[pk].copy()
            best_P[k]  = best_P[pk].copy()
            best_F[k]  = best_F[pk].copy()
        for pos in range(prev_k, k):
            for i, row in enumerate(test_df.itertuples(index=False)):
                if pos >= len(all_cands[i]):
                    continue
                pred_s = all_cands[i][pos][0]
                gt_s   = flat_set(parse_field(row.Catalyst),
                                  parse_field(row.Solvent),
                                  parse_field(row.Agent))
                p, r, f, _ = set_metrics(gt_s, pred_s)
                if p > best_P[k][i]: best_P[k][i] = p
                if r > best_R[k][i]: best_R[k][i] = r
                if f > best_F[k][i]: best_F[k][i] = f
                if pred_s == gt_s:   hit_EM[k][i]  = True
        prev_k = k

    return dict(hit_EM=hit_EM, best_R=best_R, best_P=best_P, best_F=best_F)


# ── report ────────────────────────────────────────────────────────────────────

def build_report(m, N, level_stats):
    lines = []
    def h(t):
        lines.append("\n" + "═" * 74)
        lines.append(f"  {t}")
        lines.append("═" * 74)

    h(f"Popularity Reaction Class Baseline  |  N={N:,}")

    h("0. Reaction Class Level Usage")
    lines.append(f"  {'Level':<10} {'Count':>8}  {'Rate':>8}  {'Cumulative':>14}")
    cumul = 0
    for lvl in ["l3", "l2", "l1", "global"]:
        c = level_stats.get(lvl, 0)
        cumul += c
        label = {"l3": "fine  (l3)", "l2": "mid   (l2)",
                 "l1": "coarse(l1)", "global": "global fallback"}[lvl]
        lines.append(f"  {label:<16} {c:>8}  {c/N:>8.3f}  {cumul:>8} ({100*cumul/N:.1f}%)")

    h("1. Full-Set Top-k EM and Recall")
    lines.append(f"  {'k':<6}  {'EM':>10}  {'Recall':>10}  {'Precision':>10}  {'F1':>10}")
    lines.append("  " + "─" * 54)
    for k in KS:
        lines.append(f"  top-{k:<3}  "
                     f"{m['hit_EM'][k].mean():>10.4f}  "
                     f"{m['best_R'][k].mean():>10.4f}  "
                     f"{m['best_P'][k].mean():>10.4f}  "
                     f"{m['best_F'][k].mean():>10.4f}")

    return "\n".join(lines)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--top_k",  type=int, default=10)
    ap.add_argument("--suffix", default="",
                    help="Suffix appended to output filenames (e.g. '_top30')")
    ap.add_argument("--no_rebuild_lib", action="store_true",
                    help="Skip rebuilding the library (use existing rxnclass_libraries.pkl)")
    args = ap.parse_args()
    top_k = args.top_k

    out_csv    = os.path.join(BASE_DIR, f"predictions_rxnclass_test{args.suffix}.csv")
    out_report = os.path.join(BASE_DIR, f"result_rxnclass{args.suffix}.txt")

    if args.no_rebuild_lib and os.path.exists(LIB_PKL):
        print(f"Loading existing library from {LIB_PKL} ...")
        with open(LIB_PKL, "rb") as f:
            libs = pickle.load(f)
        lib_l1, lib_l2, lib_l3 = libs["l1"], libs["l2"], libs["l3"]
        global_counter = libs["global"]
    else:
        print("Building reaction class libraries from train + val ...")
        lib_l1, lib_l2, lib_l3, global_counter = build_libraries()
        libs = {"l1": lib_l1, "l2": lib_l2, "l3": lib_l3, "global": global_counter}
        with open(LIB_PKL, "wb") as f:
            pickle.dump(libs, f)
        print(f"\nLibraries saved → {LIB_PKL}")

    global_topk = global_counter.most_common(top_k)

    print(f"\nLoading test set: {TEST_CSV}")
    test_df = pd.read_csv(TEST_CSV,
                          usecols=["row_id", "SMILES", "Reaction Class",
                                   "Catalyst", "Agent", "Solvent"],
                          low_memory=False)
    test_df = test_df.rename(columns={"Reaction Class": "rxnclass"})
    N = len(test_df)
    print(f"  {N:,} test samples")

    print("\nPredicting ...")
    all_cands  = []
    all_levels = []
    level_stats = Counter()

    for _, row in test_df.iterrows():
        rc = row["rxnclass"]
        cands, lvl = predict_one(rc, lib_l3, lib_l2, lib_l1, global_topk, top_k)
        all_cands.append(cands)
        all_levels.append(lvl)
        level_stats[lvl] += 1

    print("Evaluating ...")
    m = evaluate(test_df, all_cands)

    report = build_report(m, N, level_stats)
    print(report)
    with open(out_report, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport → {out_report}")

    # Save predictions CSV
    records = []
    for i, (row, cands) in enumerate(zip(test_df.itertuples(index=False), all_cands)):
        gt_fset = flat_set(parse_field(row.Catalyst),
                           parse_field(row.Solvent),
                           parse_field(row.Agent))
        rec = {
            "row_id":     row.row_id,
            "SMILES":     row.SMILES,
            "rxnclass":   row.rxnclass,
            "level_used": str(all_levels[i]),
            "gt":         ";".join(sorted(gt_fset)) or NULL,
        }
        for k, (fset, score) in enumerate(cands, 1):
            rec[f"cand{k}"]  = ";".join(sorted(fset)) or NULL
            rec[f"score{k}"] = score
        records.append(rec)

    pd.DataFrame(records).to_csv(out_csv, index=False)
    print(f"Predictions → {out_csv}")


if __name__ == "__main__":
    main()
