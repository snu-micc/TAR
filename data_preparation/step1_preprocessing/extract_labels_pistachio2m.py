"""
extract_labels_pistachio2m.py
==============================
Extracts unique reagent/catalyst/solvent labels with counts
from P2026_Final.csv (the final output of data_filtering.py).

Output:
  Data/labels/cat_labels.csv   — row 0 = NaN count, rest sorted by count desc
  Data/labels/solv_labels.csv
  Data/labels/reag_labels.csv

Usage:
    python extract_labels_pistachio2m.py
"""
import os
import pandas as pd
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IN_PATH   = os.path.join(SCRIPT_DIR, 'P2026_Final.csv')
LABEL_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, '../../Data/labels'))
CHUNK     = 100_000

cat_counter  = Counter()
solv_counter = Counter()
reag_counter = Counter()
n_total = 0

print(f'Processing {IN_PATH} ...')
for chunk in pd.read_csv(IN_PATH, usecols=['Catalyst','Agent', 'Solvent'],
                         dtype=str, chunksize=CHUNK):
    chunk = chunk.fillna('')
    n_total += len(chunk)

    # Catalyst — single value per row
    for val in chunk['Catalyst']:
        cat_counter[val if val else None] += 1

    # Solvent — semicolon-separated
    for val in chunk['Solvent']:
        if not val:
            solv_counter[None] += 1
        else:
            for smi in val.split(';'):
                smi = smi.strip()
                if smi:
                    solv_counter[smi] += 1

    # Reagents — semicolon-separated
    for val in chunk['Agent']:
        if not val:
            reag_counter[None] += 1
        else:
            for smi in val.split(';'):
                smi = smi.strip()
                if smi:
                    reag_counter[smi] += 1

    print(f'  {n_total:,} rows processed ...', end='\r')

print(f'\nTotal rows: {n_total:,}')

def save_labels(counter, col_name, out_path):
    nan_count = counter.pop(None, 0)
    sorted_items = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    rows = [{'col': None, 'count': nan_count}] + \
           [{'col': smi, 'count': cnt} for smi, cnt in sorted_items]
    df = pd.DataFrame(rows)
    df.columns = [col_name, 'count']
    df.to_csv(out_path, index=False)
    print(f'Saved {len(df):,} labels -> {out_path}')

os.makedirs(LABEL_DIR, exist_ok=True)

save_labels(cat_counter,  'cat',  f'{LABEL_DIR}/cat_labels.csv')
save_labels(solv_counter, 'solv', f'{LABEL_DIR}/solv_labels.csv')
save_labels(reag_counter, 'reag', f'{LABEL_DIR}/reag_labels.csv')

print('Done.')
