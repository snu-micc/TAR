"""
build_template_libraries.py
============================
Build and cache both template library formats from train+val CSVs.

  standard_template_library.pkl  — TemplateReranker
      key: (cat_tuple, solv_tuple, reag_tuple)
      used by: DETR, Reacon

  flat_template_library.pkl      — RoleAgnosticTemplateReranker
      key: sorted_tuple(all_agents_union)
      used by: roleagnostic_gnn

Run once; both files are saved to --out_dir (default: ../Data/).
All evaluation scripts check --lib_path / --flat_lib_path first before
rebuilding from scratch.

Usage
-----
  cd data_preparation
  python step3_build_template_library.py
  python step3_build_template_library.py --out_dir ../Data/
"""

import os
import sys
import argparse
import pickle

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.normpath(os.path.join(SCRIPT_DIR, '../Data'))

sys.path.insert(0, os.path.join(SCRIPT_DIR, '../tar'))
from template_reranker import TemplateReranker, RoleAgnosticTemplateReranker


def main():
    parser = argparse.ArgumentParser(
        description='Build and cache standard + flat template libraries')
    parser.add_argument('--data_dir', default=DATA_DIR,
                        help='Directory containing data_train/val_template.csv')
    parser.add_argument('--out_dir', default=DATA_DIR,
                        help='Directory to write .pkl files (default: same as data_dir)')
    parser.add_argument('--force', action='store_true',
                        help='Rebuild even if .pkl files already exist')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    std_path  = os.path.join(args.out_dir, 'standard_template_library.pkl')
    flat_path = os.path.join(args.out_dir, 'flat_template_library.pkl')

    # ── Standard library (cat/solv/reag key) ─────────────────────────────────
    if not args.force and os.path.exists(std_path):
        print(f'Standard library already exists: {std_path}  (use --force to rebuild)')
    else:
        print('Building standard template library (key: cat/solv/reag) ...')
        reranker = TemplateReranker(data_dir=args.data_dir)
        with open(std_path, 'wb') as f:
            pickle.dump(reranker, f)
        print(f'Saved → {std_path}')

    # ── Flat library (agent-union key) ────────────────────────────────────────
    if not args.force and os.path.exists(flat_path):
        print(f'Flat library already exists:     {flat_path}  (use --force to rebuild)')
    else:
        print('Building flat template library   (key: agent union) ...')
        flat_reranker = RoleAgnosticTemplateReranker(data_dir=args.data_dir)
        with open(flat_path, 'wb') as f:
            pickle.dump(flat_reranker, f)
        print(f'Saved → {flat_path}')

    print('\nDone. Pass these paths via --lib_path / --flat_lib_path to skip rebuilding.')


if __name__ == '__main__':
    main()
