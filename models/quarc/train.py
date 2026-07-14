"""
train_roleagnostic_quarc_gnn_rxnclass.py
=========================================
Role-agnostic autoregressive agent predictor — QUARC GNN variant
with Level-3 reaction class (A.B.C) as an additional one-hot input.

Key change vs train_roleagnostic_quarc_gnn.py:
  • Loads reaction class from data_{split}_template.csv (joined via _id = row_id)
  • Adds class_projector: Linear(n_classes, class_emb_dim=256) + LN + ReLU
  • FFN input: [graph_fp(1024) | class_emb(256) | agent_emb(512)] = 1792
  • Class vocab saved to roleagnostic_gnn_rxnclass/class_vocab.json

All other design decisions (subset augmentation, pascal weights, beam search,
early stopping on greedy EM) are identical to the original.

Usage:
    cd Baseline/role_augnostic_autoregressive
    python train_roleagnostic_quarc_gnn_rxnclass.py
    python train_roleagnostic_quarc_gnn_rxnclass.py --batch 256 --max_epochs 50
"""

import os
import sys
import math
import json
import time
import argparse
import re
from itertools import combinations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.normpath(os.path.join(BASE_DIR, '../..'))
LABEL_DIR = os.path.normpath(os.path.join(ROOT_DIR, 'Data/labels'))
FILT_DIR  = os.path.normpath(os.path.join(ROOT_DIR, 'Data/MPNN_data'))
DATA_DIR  = os.path.normpath(os.path.join(ROOT_DIR, 'Data'))
SAVE_DIR  = os.path.join(BASE_DIR, 'roleagnostic_gnn_rxnclass')
os.makedirs(SAVE_DIR, exist_ok=True)

CHEMPROP_DIR = os.path.normpath(os.path.join(ROOT_DIR, 'models/reacon'))
if CHEMPROP_DIR not in sys.path:
    sys.path.insert(0, CHEMPROP_DIR)

from chemprop.features import (
    set_reaction, get_atom_fdim, get_bond_fdim,
    mol2graph, BatchMolGraph,
)
from chemprop.models.mpn import MPNEncoder

EOS_IDX  = 0
UNK_CLASS = 0   # index 0 = unknown class


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Unified agent vocabulary  (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

def build_unified_vocab(label_dir):
    cat_df  = pd.read_csv(os.path.join(label_dir, 'cat_labels.csv'))
    solv_df = pd.read_csv(os.path.join(label_dir, 'solv_labels.csv'))
    reag_df = pd.read_csv(os.path.join(label_dir, 'reag_labels.csv'))

    all_smi = set()
    for df, col in [(cat_df, 'cat'), (solv_df, 'solv'), (reag_df, 'reag')]:
        for v in df[col]:
            if pd.notna(v) and str(v).strip():
                all_smi.add(str(v).strip())

    uid2smi = ['EOS'] + sorted(all_smi)
    smi2uid = {s: i for i, s in enumerate(uid2smi)}

    def make_map(df, col):
        m = [EOS_IDX]
        for v in df[col].iloc[1:]:
            s = str(v).strip() if pd.notna(v) else ''
            m.append(smi2uid.get(s, EOS_IDX))
        return m

    cat_id2uid  = make_map(cat_df,  'cat')
    solv_id2uid = make_map(solv_df, 'solv')
    reag_id2uid = make_map(reag_df, 'reag')
    return uid2smi, smi2uid, cat_id2uid, solv_id2uid, reag_id2uid


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Reaction class vocabulary  (Level 3: A.B.C)
# ─────────────────────────────────────────────────────────────────────────────

def build_class_vocab(data_dir):
    """
    Reads data_train_template.csv to collect all Level-3 reaction classes.
    Returns:
      class2idx : dict  str -> int  (UNK=0, classes start at 1)
      n_classes : int   total vocab size including UNK
    """
    tmpl_csv = os.path.join(data_dir, 'data_train_template.csv')
    df = pd.read_csv(tmpl_csv, usecols=['Reaction Class'])
    classes = sorted(df['Reaction Class'].dropna().astype(str).unique())
    class2idx = {c: i + 1 for i, c in enumerate(classes)}  # 0 = UNK
    n_classes = len(class2idx) + 1
    return class2idx, n_classes


def build_class_map(gcn_df, split, data_dir, class2idx):
    """
    Joins GCN_data_{split}.csv (_id) with data_{split}_template.csv (row_id)
    to get the Level-3 class index for each reaction.
    Returns list[int] of length len(gcn_df), UNK_CLASS for missing.
    """
    tmpl_csv = os.path.join(data_dir, f'data_{split}_template.csv')
    tmpl_df  = pd.read_csv(tmpl_csv, usecols=['row_id', 'Reaction Class'],
                           dtype={'row_id': str}).set_index('row_id')
    ids = gcn_df['_id'].astype(str).tolist()
    result = []
    for rid in ids:
        if rid in tmpl_df.index:
            rc = str(tmpl_df.at[rid, 'Reaction Class'])
            result.append(class2idx.get(rc, UNK_CLASS))
        else:
            result.append(UNK_CLASS)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 3.  SMILES validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_rxn_smiles(rxn_smiles):
    try:
        parts = rxn_smiles.strip().split('>>')
        r_smi, p_smi = parts[0].strip(), parts[-1].strip()
        return (Chem.MolFromSmiles(r_smi) is not None and
                Chem.MolFromSmiles(p_smi) is not None)
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Dataset
# ─────────────────────────────────────────────────────────────────────────────

def get_agent_uid_set(row, cat_id2uid, solv_id2uid, reag_id2uid):
    ids = set()
    for col, mapping in [('cat',   cat_id2uid),
                         ('solv0', solv_id2uid),
                         ('solv1', solv_id2uid),
                         ('reag0', reag_id2uid),
                         ('reag1', reag_id2uid)]:
        raw = int(row[col])
        uid = mapping[raw] if raw < len(mapping) else EOS_IDX
        if uid != EOS_IDX:
            ids.add(uid)
    return sorted(ids)


def pascal_weights(n):
    return [1.0 / max(math.comb(n, k), 1) for k in range(n + 1)]


class SubsetAugmentedDatasetGNN(Dataset):
    """
    Same as original but also stores and returns per-sample class index.

    __getitem__ returns:
      rxn_smiles : str
      class_idx  : int          — reaction class index (UNK_CLASS if unknown)
      a_input    : Tensor [V]
      a_target   : Tensor [V]
      weight     : Tensor []
    """

    def __init__(self, df, valid_mask, cat_id2uid, solv_id2uid, reag_id2uid,
                 vocab_size, class_indices):
        self.df           = df.reset_index(drop=True)
        self.V            = vocab_size
        self.cat_id2uid   = cat_id2uid
        self.solv_id2uid  = solv_id2uid
        self.reag_id2uid  = reag_id2uid
        self.class_indices = class_indices   # list[int], len == len(df)

        self.agent_sets = []
        for i in range(len(self.df)):
            uids = get_agent_uid_set(self.df.iloc[i], cat_id2uid, solv_id2uid, reag_id2uid)
            self.agent_sets.append(uids)

        self.index_mapping = []
        for si, uids in enumerate(self.agent_sets):
            if not valid_mask[si]:
                continue
            n = len(uids)
            self.index_mapping.append((si, -1))
            for k in range(1, n + 1):
                for ci, _ in enumerate(combinations(range(n), k)):
                    self.index_mapping.append((si, (k, ci)))

    def __len__(self):
        return len(self.index_mapping)

    def __getitem__(self, idx):
        si, aug_spec = self.index_mapping[idx]
        uids = self.agent_sets[si]
        n    = len(uids)
        V    = self.V

        rxn_smiles = str(self.df.iloc[si]['reaction'])
        class_idx  = self.class_indices[si]

        a_orig = torch.zeros(V)
        if n > 0:
            a_orig[torch.tensor(uids)] = 1.0

        ws = pascal_weights(n)

        if aug_spec == -1:
            a_input  = a_orig
            a_target = torch.zeros(V); a_target[EOS_IDX] = 1.0
            weight   = torch.tensor(1.0, dtype=torch.float32)
        else:
            k, ci = aug_spec
            combos    = list(combinations(range(n), k))
            held_out  = list(combos[ci])
            held_uids = [uids[j] for j in held_out]

            a_input  = a_orig.clone()
            a_input[torch.tensor(held_uids)] = 0.0

            a_target = torch.zeros(V)
            a_target[torch.tensor(held_uids)] = 1.0

            weight = torch.tensor(ws[k], dtype=torch.float32)

        return rxn_smiles, class_idx, a_input, a_target, weight


def gnn_collate_fn(batch):
    smiles_list, class_idx_list, a_in_list, a_tgt_list, w_list = zip(*batch)
    bmg       = mol2graph(list(smiles_list))
    class_idx = torch.tensor(class_idx_list, dtype=torch.long)
    a_in      = torch.stack(a_in_list,  dim=0)
    a_tgt     = torch.stack(a_tgt_list, dim=0)
    w         = torch.stack(w_list,     dim=0)
    return bmg, class_idx, a_in, a_tgt, w


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Model
# ─────────────────────────────────────────────────────────────────────────────

class ShortAddBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.ln     = nn.LayerNorm(dim)
        self.act    = nn.ReLU()

    def forward(self, x):
        return self.act(self.ln(self.linear(x))) + x


class FFNBase(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_blocks):
        super().__init__()
        self.input_block = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )
        self.blocks  = nn.ModuleList([ShortAddBlock(hidden_size) for _ in range(n_blocks - 1)])
        self.readout = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.input_block(x)
        for blk in self.blocks:
            x = blk(x)
        return self.readout(x)


def _make_mpn_args(graph_dim, depth, device):
    return argparse.Namespace(
        atom_messages=False, hidden_size=graph_dim, bias=False,
        depth=depth, undirected=False, device=device,
        aggregation='mean', aggregation_norm=100,
        is_atom_bond_targets=False, dropout=0.0, activation='ReLU',
        atom_descriptors=None, bond_descriptors=None, layers_per_message=1,
    )


class RoleAgnosticGNNModelRxnClass(nn.Module):
    """
    QUARC GNN + Level-3 reaction class input.

    Architecture:
      MPNEncoder (reac_diff)               → graph_fp  [B, graph_dim=1024]
      class_projector: Embedding(n_classes→class_emb_dim=256) + LN + ReLU
                                           → class_emb [B, 256]
      agent_projector: Linear(V→512) + LN + ReLU
                                           → agent_emb [B, 512]
      FFNBase([graph_fp | class_emb | agent_emb] → 2048 × n_blocks → V)
    """

    def __init__(self, graph_dim, vocab_size, n_classes, device,
                 mpn_depth=2, class_emb_dim=256, agent_emb_dim=512,
                 hidden_dim=2048, n_blocks=3):
        super().__init__()
        self.vocab_size    = vocab_size
        self.graph_dim     = graph_dim
        self.class_emb_dim = class_emb_dim
        self.agent_emb_dim = agent_emb_dim
        self.n_classes     = n_classes

        mpn_args  = _make_mpn_args(graph_dim, mpn_depth, device)
        atom_fdim = get_atom_fdim(is_reaction=True)
        bond_fdim = get_bond_fdim(is_reaction=True, atom_messages=False)
        self.encoder = MPNEncoder(mpn_args, atom_fdim, bond_fdim)
        self.bn      = nn.BatchNorm1d(graph_dim)

        # Reaction class: Embedding (learned, handles UNK gracefully)
        self.class_projector = nn.Sequential(
            nn.Embedding(n_classes, class_emb_dim),
            nn.Flatten(start_dim=-2),   # [B, 1, C] → [B, C] via squeeze below
        )
        # Use a simpler learned embedding + projection
        self.class_embed = nn.Embedding(n_classes, class_emb_dim, padding_idx=UNK_CLASS)
        self.class_ln    = nn.LayerNorm(class_emb_dim)
        self.class_act   = nn.ReLU()

        self.agent_projector = nn.Sequential(
            nn.Linear(vocab_size, agent_emb_dim),
            nn.LayerNorm(agent_emb_dim),
            nn.ReLU(),
        )

        self.ffn = FFNBase(
            input_size  = graph_dim + class_emb_dim + agent_emb_dim,
            hidden_size = hidden_dim,
            output_size = vocab_size,
            n_blocks    = n_blocks,
        )

    def encode_reaction(self, bmg):
        return self.bn(self.encoder(bmg))

    def encode_class(self, class_idx):
        """class_idx: [B] long → class_emb [B, class_emb_dim]"""
        emb = self.class_embed(class_idx)          # [B, class_emb_dim]
        return self.class_act(self.class_ln(emb))  # [B, class_emb_dim]

    def forward(self, bmg, class_idx, agent_ctx):
        """
        bmg       : BatchMolGraph
        class_idx : Tensor [B] long — reaction class index
        agent_ctx : Tensor [B, V]  — multi-hot already-selected agents
        Returns logits [B, V]
        """
        graph_fp  = self.encode_reaction(bmg)           # [B, graph_dim]
        class_emb = self.encode_class(class_idx)        # [B, class_emb_dim]
        agent_emb = self.agent_projector(agent_ctx)     # [B, agent_emb_dim]
        combined  = torch.cat([graph_fp, class_emb, agent_emb], dim=-1)
        return self.ffn(combined)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Greedy evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def greedy_eval(model, df, valid_mask, cat_id2uid, solv_id2uid, reag_id2uid,
                class_indices, vocab_size, device, max_steps=6):
    model.eval()
    n_correct = 0
    n_total   = 0

    for i in range(len(df)):
        if not valid_mask[i]:
            continue
        gt_uids   = set(get_agent_uid_set(df.iloc[i], cat_id2uid, solv_id2uid, reag_id2uid))
        rxn_smi   = str(df.iloc[i]['reaction'])
        class_idx = class_indices[i]

        try:
            bmg = mol2graph([rxn_smi])
        except Exception:
            continue

        cidx = torch.tensor([class_idx], dtype=torch.long, device=device)
        ctx  = torch.zeros(1, vocab_size, device=device)
        pred = set()

        for _ in range(max_steps):
            logits = model(bmg, cidx, ctx)[0]
            logits[ctx[0] == 1] = -1e6
            nxt = logits.argmax().item()
            if nxt == EOS_IDX:
                break
            pred.add(nxt)
            ctx[0, nxt] = 1.0

        n_correct += int(pred == gt_uids)
        n_total   += 1

    return n_correct / n_total if n_total > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Data loading helper
# ─────────────────────────────────────────────────────────────────────────────

def _validate_worker(smi):
    return validate_rxn_smiles(smi)


def load_and_validate(split, filt_dir):
    import multiprocessing

    path = os.path.join(filt_dir, f'GCN_data_{split}.csv')
    df   = pd.read_csv(path)
    n_workers = min(multiprocessing.cpu_count() // 2, 12)
    print(f'  {split}: {len(df):,} rows — validating SMILES ...', flush=True)
    with multiprocessing.Pool(processes=n_workers) as pool:
        valid_mask = pool.map(_validate_worker,
                              df['reaction'].astype(str).tolist(),
                              chunksize=1000)
    print(f'    valid: {sum(valid_mask):,}/{len(df):,}', flush=True)
    return df, valid_mask


def parse_train_log(log_path):
    if not os.path.exists(log_path):
        return None
    pat  = re.compile(r'Epoch (\d+)\s+train_loss=([\d.]+)\s+val_EM=([\d.]+)\s+lr=([\d.e+\-]+)')
    rows = []
    with open(log_path, encoding='utf-8') as f:
        for line in f:
            m = pat.search(line)
            if m:
                rows.append({'epoch': int(m.group(1)),
                             'train_loss': float(m.group(2)),
                             'val_em': float(m.group(3)),
                             'lr': float(m.group(4))})
    if not rows:
        return None
    last      = rows[-1]
    best_val  = max(r['val_em'] for r in rows)
    best_ep   = max(r['epoch'] for r in rows if r['val_em'] == best_val)
    patience  = last['epoch'] - best_ep
    log_rows  = [{'epoch': r['epoch'], 'train_loss': r['train_loss'],
                  'val_em': r['val_em']} for r in rows]
    return last['epoch'], best_val, patience, last['lr'], log_rows


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_dim',    type=int,   default=1024)
    parser.add_argument('--mpn_depth',   type=int,   default=2)
    parser.add_argument('--class_emb',   type=int,   default=256,
                        help='Reaction class embedding dim')
    parser.add_argument('--agent_emb',   type=int,   default=512)
    parser.add_argument('--hidden',      type=int,   default=2048)
    parser.add_argument('--n_blocks',    type=int,   default=3)
    parser.add_argument('--batch',       type=int,   default=512)
    parser.add_argument('--max_epochs',  type=int,   default=100)
    parser.add_argument('--patience',    type=int,   default=5)
    parser.add_argument('--lr_max',      type=float, default=2.3e-4)
    parser.add_argument('--lr_init',     type=float, default=1e-5)
    args = parser.parse_args()

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {DEVICE}', flush=True)

    set_reaction(True, 'reac_diff')
    print(f'Featurization: atom_fdim={get_atom_fdim(is_reaction=True)}, '
          f'bond_fdim={get_bond_fdim(is_reaction=True)}', flush=True)

    # ── Agent vocab ────────────────────────────────────────────────────────────
    print('\nBuilding unified agent vocabulary ...', flush=True)
    uid2smi, smi2uid, cat_id2uid, solv_id2uid, reag_id2uid = build_unified_vocab(LABEL_DIR)
    V = len(uid2smi)
    print(f'  Agent vocab size: {V}  (index 0 = EOS)', flush=True)

    vocab_path = os.path.join(SAVE_DIR, 'unified_vocab.json')
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(uid2smi, f, ensure_ascii=False, indent=2)

    # ── Reaction class vocab ───────────────────────────────────────────────────
    print('\nBuilding reaction class vocabulary (Level 3: A.B.C) ...', flush=True)
    class2idx, n_classes = build_class_vocab(DATA_DIR)
    print(f'  Class vocab size: {n_classes}  (index 0 = UNK, 1..{n_classes-1} = classes)',
          flush=True)

    class_vocab_path = os.path.join(SAVE_DIR, 'class_vocab.json')
    with open(class_vocab_path, 'w', encoding='utf-8') as f:
        json.dump(class2idx, f, ensure_ascii=False, indent=2)
    print(f'  Saved -> {class_vocab_path}', flush=True)

    # ── Data ───────────────────────────────────────────────────────────────────
    print('\nLoading data ...', flush=True)
    train_df, train_valid = load_and_validate('train', FILT_DIR)
    val_df,   val_valid   = load_and_validate('val',   FILT_DIR)
    test_df,  test_valid  = load_and_validate('test',  FILT_DIR)

    print('\nBuilding reaction class indices ...', flush=True)
    train_class = build_class_map(train_df, 'train', DATA_DIR, class2idx)
    val_class   = build_class_map(val_df,   'val',   DATA_DIR, class2idx)
    test_class  = build_class_map(test_df,  'test',  DATA_DIR, class2idx)
    unk_train   = sum(1 for c in train_class if c == UNK_CLASS)
    print(f'  Train UNK class: {unk_train}/{len(train_class)} '
          f'({100*unk_train/len(train_class):.1f}%)', flush=True)

    # ── Dataset & loader ───────────────────────────────────────────────────────
    print('\nBuilding augmented training dataset ...', flush=True)
    train_ds = SubsetAugmentedDatasetGNN(
        train_df, train_valid, cat_id2uid, solv_id2uid, reag_id2uid, V, train_class)
    print(f'  Augmented training samples: {len(train_ds):,}  '
          f'(from {len(train_df):,} reactions)', flush=True)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=0, pin_memory=False,
        collate_fn=gnn_collate_fn,
    )

    # ── Model ──────────────────────────────────────────────────────────────────
    model = RoleAgnosticGNNModelRxnClass(
        graph_dim     = args.graph_dim,
        vocab_size    = V,
        n_classes     = n_classes,
        device        = DEVICE,
        mpn_depth     = args.mpn_depth,
        class_emb_dim = args.class_emb,
        agent_emb_dim = args.agent_emb,
        hidden_dim    = args.hidden,
        n_blocks      = args.n_blocks,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f'\nModel parameters: {n_params:,}')
    print(f'  Encoder  : D-MPNN (reac_diff, depth={args.mpn_depth}, hidden={args.graph_dim})')
    print(f'  ClassEmb : Embedding({n_classes} → {args.class_emb}) + LN + ReLU')
    print(f'  AgentProj: Linear({V} → {args.agent_emb}) + LN + ReLU')
    print(f'  FFN input: {args.graph_dim + args.class_emb + args.agent_emb} '
          f'→ {args.hidden} × {args.n_blocks} blocks → {V}')

    criterion = nn.CrossEntropyLoss(reduction='none')
    model_path = os.path.join(SAVE_DIR, 'best_model.pt')
    ckpt_path  = os.path.join(SAVE_DIR, 'checkpoint.pt')
    log_path   = os.path.join(SAVE_DIR, 'train_log.txt')

    # ── Optimizer / scheduler ──────────────────────────────────────────────────
    best_val_em    = -1.0
    patience_count = 0
    log_rows       = []
    start_epoch    = 1

    if os.path.exists(ckpt_path):
        print(f'Resuming from checkpoint {ckpt_path} ...', flush=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_init)
        total_steps = len(train_loader) * args.max_epochs
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr_max, total_steps=total_steps,
            pct_start=0.05, final_div_factor=args.lr_max / args.lr_init)
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        start_epoch    = ckpt['epoch'] + 1
        best_val_em    = ckpt['best_val_em']
        patience_count = ckpt['patience_count']
        log_rows       = ckpt['log_rows']
        print(f'  Resumed from epoch {start_epoch}  '
              f'(best_val_EM={best_val_em:.4f})', flush=True)
    elif os.path.exists(model_path) and os.path.exists(log_path):
        parsed = parse_train_log(log_path)
        if parsed is not None:
            last_epoch, best_val_em, patience_count, last_lr, log_rows = parsed
            start_epoch      = last_epoch + 1
            remaining_epochs = args.max_epochs - last_epoch
            print(f'Warm-starting from {model_path} '
                  f'(epoch={last_epoch}, best_val_EM={best_val_em:.4f})', flush=True)
            ckpt = torch.load(model_path, map_location=DEVICE)
            model.load_state_dict(ckpt['model_state'])
            optimizer = torch.optim.Adam(model.parameters(), lr=last_lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=len(train_loader) * remaining_epochs,
                eta_min=args.lr_init)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_init)
            total_steps = len(train_loader) * args.max_epochs
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=args.lr_max, total_steps=total_steps,
                pct_start=0.05, final_div_factor=args.lr_max / args.lr_init)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_init)
        total_steps = len(train_loader) * args.max_epochs
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr_max, total_steps=total_steps,
            pct_start=0.05, final_div_factor=args.lr_max / args.lr_init)

    # ── Training loop ──────────────────────────────────────────────────────────
    print(f'\nTraining for up to {args.max_epochs} epochs '
          f'(patience={args.patience}, batch={args.batch}) ...\n', flush=True)

    for epoch in range(start_epoch, args.max_epochs + 1):
        model.train()
        train_loss = 0.0
        n_samples  = 0
        t0 = time.time()

        for bmg, class_idx, a_in, a_tgt, w in train_loader:
            class_idx = class_idx.to(DEVICE)
            a_in      = a_in.to(DEVICE)
            a_tgt     = a_tgt.to(DEVICE)
            w         = w.to(DEVICE)

            optimizer.zero_grad()
            logits   = model(bmg, class_idx, a_in)   # [B, V]
            loss_raw = criterion(logits, a_tgt)       # [B]
            loss     = (loss_raw * w).mean()
            loss.backward()
            optimizer.step()
            scheduler.step()

            bs = a_in.size(0)
            train_loss += loss.item() * bs
            n_samples  += bs

        train_loss /= max(n_samples, 1)

        val_em  = greedy_eval(model, val_df, val_valid,
                              cat_id2uid, solv_id2uid, reag_id2uid,
                              val_class, V, DEVICE)
        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]

        print(f'Epoch {epoch:03d}  train_loss={train_loss:.4f}  '
              f'val_EM={val_em:.4f}  lr={lr_now:.2e}  [{elapsed:.1f}s]', flush=True)

        log_rows.append({'epoch': epoch,
                         'train_loss': round(train_loss, 6),
                         'val_em':     round(val_em, 6)})
        pd.DataFrame(log_rows).to_csv(os.path.join(SAVE_DIR, 'train_log.csv'), index=False)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f'Epoch {epoch:03d}  train_loss={train_loss:.4f}  '
                    f'val_EM={val_em:.4f}  lr={lr_now:.2e}  [{elapsed:.1f}s]\n')

        if val_em > best_val_em:
            best_val_em    = val_em
            patience_count = 0
            torch.save({
                'model_state':  model.state_dict(),
                'args':         vars(args),
                'vocab_size':   V,
                'n_classes':    n_classes,
                'graph_dim':    args.graph_dim,
                'class_emb':    args.class_emb,
                'agent_emb':    args.agent_emb,
            }, model_path)
            print(f'  -> Saved best model (val_EM={val_em:.4f})', flush=True)
        else:
            patience_count += 1

        torch.save({
            'epoch':           epoch,
            'model_state':     model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'best_val_em':     best_val_em,
            'patience_count':  patience_count,
            'log_rows':        log_rows,
        }, ckpt_path)

        if patience_count >= args.patience:
            print(f'Early stopping at epoch {epoch}.', flush=True)
            break

    # ── Final test evaluation ──────────────────────────────────────────────────
    print('\nLoading best model for test evaluation ...')
    ckpt = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state'])

    test_em = greedy_eval(model, test_df, test_valid,
                          cat_id2uid, solv_id2uid, reag_id2uid,
                          test_class, V, DEVICE)
    print(f'Test greedy Exact Match: {test_em:.4f}')
    print(f'\nSummary:')
    print(f'  Best val greedy EM : {best_val_em:.4f}')
    print(f'  Test greedy EM     : {test_em:.4f}')
    print('Done.')


if __name__ == '__main__':
    main()
