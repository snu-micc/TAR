"""
Converts raw Pistachio CSV data into the format expected by Chem-DETR.

Reads:
  data_total.csv              – full dataset (output of data_filtering.py);
                                randomly split 8:1:1 into the files below
                                if they don't already exist

Writes:
  data_train.csv, data_val.csv, data_test.csv – random 8:1:1 split
  labels/cat_labels.csv       – catalyst vocabulary  (cat, count)
  labels/solv_labels.csv      – solvent vocabulary    (solv, count)
  labels/reag_labels.csv      – reagent vocabulary    (reag, count)
  MPNN_data/GCN_data_train.csv
  MPNN_data/GCN_data_val.csv
  MPNN_data/GCN_data_test.csv
"""

import os
import pandas as pd
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "../../Data"))
LABEL_DIR = os.path.join(DATA_DIR, "labels")
MPNN_DIR = os.path.join(DATA_DIR, "MPNN_data")

SPLIT_SEED = 42          # random seed for the 8:1:1 split (reproducibility)
VAL_FRAC = 0.1
TEST_FRAC = 0.1


def split_field(value):
    """Split a semicolon-separated field into a list of stripped SMILES."""
    if pd.isna(value) or str(value).strip() == "":
        return []
    return [s.strip() for s in str(value).split(";") if s.strip()]


def build_labels(df_total):
    """Build label files from the full dataset.

    Returns:
        cat_map, solv_map, reag_map: dicts of {SMILES: 1-based index}
    """
    cat_counter = Counter()
    solv_counter = Counter()
    reag_counter = Counter()

    for _, row in df_total.iterrows():
        # Catalyst – single value
        cat_vals = split_field(row.get("Catalyst"))
        for v in cat_vals:
            cat_counter[v] += 1

        # Solvent – semicolon-separated
        solv_vals = split_field(row.get("Solvent"))
        for v in solv_vals:
            solv_counter[v] += 1

        # Agent → reagent – semicolon-separated
        reag_vals = split_field(row.get("Agent"))
        for v in reag_vals:
            reag_counter[v] += 1

    def save_label_file(counter, col_name, filename, total_rows):
        """Save a label CSV with None as row 0, then entries sorted by count desc."""
        # None의 개수 = 전체 행 - 채워진 항목의 총 개수
        none_count = total_rows - counter.total()
        rows = [("None", none_count)]
        for smiles, count in counter.most_common():
            rows.append((smiles, count))
        df = pd.DataFrame(rows, columns=[col_name, "count"])
        path = os.path.join(LABEL_DIR, filename)
        df.to_csv(path, index=False)
        print(f"  {filename}: {len(df)} entries (1 None + {len(df)-1} unique)")
        return df

    print("Building label files...")
    total_rows = len(df_total)
    cat_df = save_label_file(cat_counter, "cat", "cat_labels.csv", total_rows)
    solv_df = save_label_file(solv_counter, "solv", "solv_labels.csv", total_rows)
    reag_df = save_label_file(reag_counter, "reag", "reag_labels.csv", total_rows)

    # Build SMILES → index mappings (index 0 = None)
    cat_map = {smiles: idx for idx, smiles in enumerate(cat_df["cat"]) if smiles != "None"}
    solv_map = {smiles: idx for idx, smiles in enumerate(solv_df["solv"]) if smiles != "None"}
    reag_map = {smiles: idx for idx, smiles in enumerate(reag_df["reag"]) if smiles != "None"}

    return cat_map, solv_map, reag_map


def lookup(value, mapping):
    """Look up a single SMILES in a mapping, returning 0 for missing/NaN."""
    if pd.isna(value) or str(value).strip() == "":
        return 0
    return mapping.get(str(value).strip(), 0)


def convert_split(split_name, cat_map, solv_map, reag_map):
    """Convert one data split (train/val/test) to GCN format."""
    in_path = os.path.join(DATA_DIR, f"data_{split_name}.csv")
    out_path = os.path.join(MPNN_DIR, f"GCN_data_{split_name}.csv")

    df = pd.read_csv(in_path)
    n = len(df)

    # reaction = SMILES column (reaction SMILES with atom maps)
    reaction = df["SMILES"]

    # _id = row_id
    _id = df["row_id"]

    # cat: single catalyst value → index
    cat = df["Catalyst"].apply(lambda x: lookup(x, cat_map))

    # solv0, solv1: split Solvent by ';', look up indices
    solv0 = []
    solv1 = []
    for val in df["Solvent"]:
        parts = split_field(val)
        solv0.append(solv_map.get(parts[0], 0) if len(parts) > 0 else 0)
        solv1.append(solv_map.get(parts[1], 0) if len(parts) > 1 else 0)

    # reag0, reag1: split Agent by ';', look up indices
    reag0 = []
    reag1 = []
    for val in df["Agent"]:
        parts = split_field(val)
        reag0.append(reag_map.get(parts[0], 0) if len(parts) > 0 else 0)
        reag1.append(reag_map.get(parts[1], 0) if len(parts) > 1 else 0)

    out_df = pd.DataFrame({
        "reaction": reaction,
        "cat": cat,
        "solv0": solv0,
        "solv1": solv1,
        "reag0": reag0,
        "reag1": reag1,
        "_id": _id,
    })

    out_df.to_csv(out_path, index=False)
    print(f"  {os.path.basename(out_path)}: {n} rows")
    return n


def verify(cat_map, solv_map, reag_map):
    """Run basic verification checks on the output files."""
    print("\nVerification:")

    # Check label files have None as first row
    for fname, col in [("cat_labels.csv", "cat"), ("solv_labels.csv", "solv"), ("reag_labels.csv", "reag")]:
        df = pd.read_csv(os.path.join(LABEL_DIR, fname), keep_default_na=False)
        first = str(df.iloc[0, 0])
        status = "OK" if first == "None" else f"FAIL (got '{first}')"
        print(f"  {fname} first row = '{first}' ... {status}")

    # Check GCN files
    for split in ["train", "val", "test"]:
        gcn_path = os.path.join(MPNN_DIR, f"GCN_data_{split}.csv")
        raw_path = os.path.join(DATA_DIR, f"data_{split}.csv")
        gcn_df = pd.read_csv(gcn_path)
        raw_df = pd.read_csv(raw_path)

        expected_cols = ["reaction", "cat", "solv0", "solv1", "reag0", "reag1", "_id"]
        cols_ok = list(gcn_df.columns) == expected_cols
        rows_ok = len(gcn_df) == len(raw_df)
        # Check integer types for target columns
        int_cols = ["cat", "solv0", "solv1", "reag0", "reag1"]
        ints_ok = all(gcn_df[c].dtype in ["int64", "int32"] for c in int_cols)

        print(f"  GCN_data_{split}.csv: cols={cols_ok}, rows={rows_ok} ({len(gcn_df)}), int_targets={ints_ok}")


def split_data():
    """Randomly split data_total.csv into data_{train,val,test}.csv (8:1:1).

    Skips splitting if all three split files already exist. Returns the full
    DataFrame (data_total.csv) for downstream vocabulary building.
    """
    total_path = os.path.join(DATA_DIR, "data_total.csv")
    split_paths = {s: os.path.join(DATA_DIR, f"data_{s}.csv")
                   for s in ["train", "val", "test"]}

    if all(os.path.exists(p) for p in split_paths.values()):
        print("Split files already exist — skipping split.")
        return pd.read_csv(total_path) if os.path.exists(total_path) else None

    print(f"Reading {total_path} for 8:1:1 split (seed={SPLIT_SEED}) ...")
    df_total = pd.read_csv(total_path)
    n = len(df_total)

    # Shuffle once, then slice into test / val / train.
    shuffled = df_total.sample(frac=1.0, random_state=SPLIT_SEED).reset_index(drop=True)
    n_test = int(round(n * TEST_FRAC))
    n_val = int(round(n * VAL_FRAC))
    df_test = shuffled.iloc[:n_test]
    df_val = shuffled.iloc[n_test:n_test + n_val]
    df_train = shuffled.iloc[n_test + n_val:]

    for split, part in [("train", df_train), ("val", df_val), ("test", df_test)]:
        part.to_csv(split_paths[split], index=False)
        print(f"  data_{split}.csv: {len(part):,} rows")

    return df_total


def main():
    os.makedirs(LABEL_DIR, exist_ok=True)
    os.makedirs(MPNN_DIR, exist_ok=True)

    # Step 0: Random 8:1:1 split of data_total.csv into data_{train,val,test}.csv
    df_total = split_data()

    # Step 1 & 2: Build vocabulary from the full dataset (data_total.csv).
    if df_total is None:
        print("data_total.csv not found — concatenating data_train/val/test.csv ...")
        df_total = pd.concat(
            [pd.read_csv(os.path.join(DATA_DIR, f"data_{split}.csv"))
             for split in ["train", "val", "test"]],
            ignore_index=True,
        )
    print(f"  Total rows: {len(df_total)}")

    cat_map, solv_map, reag_map = build_labels(df_total)
    print(f"\nVocab sizes: cat={len(cat_map)}, solv={len(solv_map)}, reag={len(reag_map)}")

    # Step 3: Convert each split
    print("\nConverting splits...")
    for split in ["train", "val", "test"]:
        convert_split(split, cat_map, solv_map, reag_map)

    # Verify
    verify(cat_map, solv_map, reag_map)
    print("\nDone.")


if __name__ == "__main__":
    main()
