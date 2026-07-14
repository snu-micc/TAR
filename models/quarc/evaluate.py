"""
evaluate_roleagnostic_gnn_rxnclass_template.py
================================================
Beam search inference for the roleagnostic GNN trained WITH Level-3
reaction class input (train_roleagnostic_quarc_gnn_rxnclass.py).

Key difference vs evaluate_roleagnostic_gnn_template.py:
  • Loads class_vocab.json from the model directory
  • Joins test reactions to data_test_template.csv to get Level-3 class index
  • Passes class embedding into beam search at every step

Template re-ranking (including entropy-adaptive λ) lives in tar/rerank_all.py,
which consumes the beam CSV mirrored to tar/InferenceResults/{split}/ by this
script. This script only runs Phase 1 (beam search inference).

Phase 1: beam search → roleagnostic_gnn_rxnclass/inference_results_beam{N}_{split}.csv
          (also copied to tar/InferenceResults/{split}/quarc_rxnclass_beam{N}.csv)

Usage:
    cd models/quarc

    python evaluate.py --split test
    python evaluate.py --skip_inference   # reuse existing inference CSV, just re-copy
"""

import os
import sys
import json
import argparse
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore')

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.normpath(os.path.join(SCRIPT_DIR, '../..'))
LABEL_DIR  = os.path.join(ROOT_DIR, 'Data/labels')
MPNN_DIR   = os.path.join(ROOT_DIR, 'Data/MPNN_data')
DATA_DIR   = os.path.join(ROOT_DIR, 'Data')
SAVE_DIR   = os.path.join(SCRIPT_DIR, 'roleagnostic_gnn_rxnclass')

REACON_CP  = os.path.join(ROOT_DIR, 'models/reacon')
if REACON_CP not in sys.path:
    sys.path.insert(0, REACON_CP)

from chemprop.features import set_reaction, get_atom_fdim, get_bond_fdim, mol2graph
from chemprop.models.mpn import MPNEncoder

EOS_IDX  = 0
UNK_CLASS = 0
NULL     = 'none'
NONE_SET = {'None', 'nan', '', 'none', 'NaN'}


# ══════════════════════════════════════════════════════════════════════════════
# Model  (must match train_roleagnostic_quarc_gnn_rxnclass.py exactly)
# ══════════════════════════════════════════════════════════════════════════════

class ShortAddBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.ln     = nn.LayerNorm(dim)
        self.act    = nn.ReLU()
    def forward(self, x): return self.act(self.ln(self.linear(x))) + x


class FFNBase(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_blocks):
        super().__init__()
        self.input_block = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU())
        self.blocks  = nn.ModuleList([ShortAddBlock(hidden_size) for _ in range(n_blocks - 1)])
        self.readout = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x = self.input_block(x)
        for blk in self.blocks: x = blk(x)
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

        # Keep unused class_projector to avoid state_dict mismatch
        self.class_projector = nn.Sequential(
            nn.Embedding(n_classes, class_emb_dim),
            nn.Flatten(start_dim=-2),
        )
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
        emb = self.class_embed(class_idx)
        return self.class_act(self.class_ln(emb))

    def forward(self, bmg, class_idx, agent_ctx):
        graph_fp  = self.encode_reaction(bmg)
        class_emb = self.encode_class(class_idx)
        agent_emb = self.agent_projector(agent_ctx)
        combined  = torch.cat([graph_fp, class_emb, agent_emb], dim=-1)
        return self.ffn(combined)


# ══════════════════════════════════════════════════════════════════════════════
# Vocabulary helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_unified_vocab(vocab_path):
    with open(vocab_path, encoding='utf-8') as f:
        uid2smi = json.load(f)
    smi2uid = {s: i for i, s in enumerate(uid2smi)}
    return uid2smi, smi2uid


def load_class_vocab(class_vocab_path):
    with open(class_vocab_path, encoding='utf-8') as f:
        class2idx = json.load(f)
    return class2idx


def build_vocab_maps(label_dir, smi2uid):
    def make_map(df, col):
        m = [EOS_IDX]
        for v in df[col].iloc[1:]:
            s = str(v).strip() if pd.notna(v) else ''
            m.append(smi2uid.get(s, EOS_IDX))
        return m
    cat_df   = pd.read_csv(os.path.join(label_dir, 'cat_labels.csv'))
    solv_df  = pd.read_csv(os.path.join(label_dir, 'solv_labels.csv'))
    reag_df  = pd.read_csv(os.path.join(label_dir, 'reag_labels.csv'))
    return (make_map(cat_df, 'cat'),
            make_map(solv_df, 'solv'),
            make_map(reag_df, 'reag'))


def load_label_lists(label_dir):
    def _load(path, col):
        df = pd.read_csv(path)
        return df[col].fillna('').astype(str).tolist()
    return (_load(os.path.join(label_dir, 'cat_labels.csv'),  'cat'),
            _load(os.path.join(label_dir, 'solv_labels.csv'), 'solv'),
            _load(os.path.join(label_dir, 'reag_labels.csv'), 'reag'))


def build_class_indices(gcn_df, split, data_dir, class2idx):
    """Join GCN_data with data_{split}_template.csv to get class index per row."""
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


# ══════════════════════════════════════════════════════════════════════════════
# GT decoding
# ══════════════════════════════════════════════════════════════════════════════

def norm(s):
    return '' if str(s) in NONE_SET else str(s)


def decode_gt_flat(row, cat_list, solv_list, reag_list):
    return frozenset(filter(None, [
        norm(cat_list[int(row['cat'])]),
        norm(solv_list[int(row['solv0'])]),
        norm(solv_list[int(row['solv1'])]),
        norm(reag_list[int(row['reag0'])]),
        norm(reag_list[int(row['reag1'])]),
    ]))


def encode_flat(smi_set):
    return ';'.join(sorted(s for s in smi_set if s not in NONE_SET)) or NULL


def validate_rxn_smiles(s):
    try:
        parts = s.strip().split('>>')
        r, p  = parts[0].strip(), parts[-1].strip()
        return Chem.MolFromSmiles(r) is not None and Chem.MolFromSmiles(p) is not None
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════════════════
# Beam search  (class-aware)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def batch_beam_search(model, graph_fps, class_fps, vocab_size, device,
                      beam_size=20, max_steps=6):
    """
    Vectorized beam search for R reactions in parallel.

    graph_fps : [R, graph_dim]   — pre-computed reaction encoder output
    class_fps : [R, class_emb_dim] — pre-computed class embedding (fixed per reaction)
    Returns   : list of R lists, each [(frozenset_uid, log_score), ...]
    """
    R, D  = graph_fps.shape
    K     = beam_size
    V     = vocab_size
    K_exp = min(K, V - 1)

    ctx    = torch.zeros(R, K, V, device=device)
    scores = torch.full((R, K), -1e9, device=device)
    scores[:, 0] = 0.0

    # Pre-expand class_fps for all beams: [R, K, class_emb_dim]
    C = class_fps.shape[1]
    class_fps_expanded = class_fps.unsqueeze(1).expand(R, K, C)  # [R, K, C]

    comp_data = []

    for _ in range(max_steps):
        alive = scores > -1e8
        if not alive.any():
            break

        flat_ctx   = ctx.view(R * K, V)
        flat_alive = alive.view(R * K)
        flat_gfp   = (graph_fps.unsqueeze(1).expand(R, K, D).reshape(R * K, D))
        flat_cfp   = class_fps_expanded.reshape(R * K, C)

        # Agent embedding for live beams only
        ae_live = model.agent_projector(flat_ctx[flat_alive])               # [B_live, agent_emb]

        # Concatenate: [graph_fp | class_fp | agent_emb]
        combined_live = torch.cat([
            flat_gfp[flat_alive],
            flat_cfp[flat_alive],
            ae_live,
        ], dim=-1)
        log_live = model.ffn(combined_live)                                  # [B_live, V]

        logits = torch.full((R * K, V), -1e9, device=device)
        logits[flat_alive]   = log_live
        logits[flat_ctx > 0] = -1e9    # mask already-chosen

        log_probs = torch.log_softmax(logits.clamp(min=-1e8), dim=-1)
        log_probs[~flat_alive] = -1e9

        # EOS completions
        eos_s = (scores.view(R * K) + log_probs[:, EOS_IDX]).view(R, K)
        eos_s[~alive] = -1e9
        comp_data.append((eos_s.cpu(), ctx.cpu()))

        # Expand: top-K_exp non-EOS per beam
        lp_no_eos = log_probs.clone()
        lp_no_eos[:, EOS_IDX] = -1e9
        vals, idxs = lp_no_eos.topk(K_exp, dim=-1)

        child_sc = (scores.view(R * K, 1) + vals).view(R, K * K_exp)
        dead     = (~alive).unsqueeze(2).expand(R, K, K_exp).reshape(R, K * K_exp)
        child_sc[dead] = -1e9

        parent_exp = flat_ctx.unsqueeze(1).expand(R * K, K_exp, V).clone()
        parent_exp.scatter_(2, idxs.unsqueeze(2), 1.0)
        child_ctx = parent_exp.view(R, K * K_exp, V)
        del parent_exp

        topk_sc, topk_rank = child_sc.topk(K, dim=-1)
        ctx    = child_ctx.gather(1, topk_rank.unsqueeze(-1).expand(R, K, V))
        scores = topk_sc
        del child_ctx

    comp_data.append((scores.cpu(), ctx.cpu()))

    results = []
    for r in range(R):
        best: dict = {}
        for step_sc, step_ctx in comp_data:
            sc_r = step_sc[r]
            cx_r = step_ctx[r]
            for b in range(K):
                s = sc_r[b].item()
                if s <= -1e8:
                    continue
                uid_set = frozenset(cx_r[b].nonzero(as_tuple=True)[0].tolist())
                if uid_set not in best or best[uid_set] < s:
                    best[uid_set] = s
        results.append(sorted(best.items(), key=lambda x: x[1], reverse=True))
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1 — Inference
# ══════════════════════════════════════════════════════════════════════════════

def run_inference(model, gcn_df, class_indices,
                  uid2smi, cat_id2uid, solv_id2uid, reag_id2uid,
                  cat_list, solv_list, reag_list,
                  device, vocab_size, beam_size, out_csv, rxn_batch=512):

    N       = len(gcn_df)
    n_cands = beam_size

    header = ['_id', 'reaction', 'gt_flat']
    for j in range(1, n_cands + 1):
        header += [f'cand{j}', f'score{j}']

    model.eval()
    first_chunk = True

    for chunk_start in tqdm(range(0, N, rxn_batch), desc='Beam search batches'):
        chunk_end = min(chunk_start + rxn_batch, N)
        batch_df  = gcn_df.iloc[chunk_start:chunk_end]
        batch_cls = class_indices[chunk_start:chunk_end]

        ids      = batch_df['_id'].astype(str).tolist()
        rxn_smis = batch_df['reaction'].astype(str).tolist()

        meta         = []
        valid_smiles = []
        valid_idx    = []
        valid_cls    = []

        for local_i in range(len(batch_df)):
            row     = batch_df.iloc[local_i]
            rxn_smi = rxn_smis[local_i]
            gt_smi  = decode_gt_flat(row, cat_list, solv_list, reag_list)
            meta.append({
                '_id':      ids[local_i],
                'reaction': rxn_smi,
                'gt_flat':  encode_flat(gt_smi),
            })
            if validate_rxn_smiles(rxn_smi):
                valid_smiles.append(rxn_smi)
                valid_idx.append(local_i)
                valid_cls.append(batch_cls[local_i])

        beam_results = {}
        if valid_smiles:
            try:
                bmg = mol2graph(valid_smiles)
                with torch.no_grad():
                    graph_fps = model.encode_reaction(bmg)            # [R, graph_dim]
                    cls_t     = torch.tensor(valid_cls, dtype=torch.long, device=device)
                    class_fps = model.encode_class(cls_t)             # [R, class_emb_dim]

                beams = batch_beam_search(
                    model, graph_fps, class_fps, vocab_size, device, beam_size)
                for local_i, beam in zip(valid_idx, beams):
                    beam_results[local_i] = beam
            except Exception as e:
                import traceback
                print(f'  [warn] batch failed ({e}), skipping chunk')
                traceback.print_exc()

        records = []
        for local_i, rec_base in enumerate(meta):
            rec  = dict(rec_base)
            beam = beam_results.get(local_i, [])
            for j, (uid_set, log_score) in enumerate(beam, 1):
                smi_set = frozenset(
                    uid2smi[u] for u in uid_set
                    if u != EOS_IDX and uid2smi[u] not in NONE_SET)
                rec[f'cand{j}']  = encode_flat(smi_set)
                rec[f'score{j}'] = round(log_score, 6)
            records.append(rec)

        pd.DataFrame(records, columns=header).to_csv(
            out_csv, mode='w' if first_chunk else 'a',
            header=first_chunk, index=False)
        first_chunk = False
        print(f'  [{chunk_end}/{N}] saved', flush=True)

    print(f'Inference CSV → {out_csv}')


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split',          choices=['test', 'val'], default='test')
    parser.add_argument('--beam_size',      type=int,   default=20)
    parser.add_argument('--rxn_batch',      type=int,   default=32768)
    parser.add_argument('--device',         default=None)
    parser.add_argument('--skip_inference', action='store_true')
    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(SAVE_DIR, exist_ok=True)

    inf_csv = os.path.join(SAVE_DIR, f'inference_results_beam{args.beam_size}_{args.split}.csv')

    # mirror inference CSV to InferenceResults/{split}/ with canonical name
    IR_DIR  = os.path.join(ROOT_DIR, 'tar', 'InferenceResults', args.split)
    ir_csv  = os.path.join(IR_DIR, f'quarc_rxnclass_beam{args.beam_size}.csv')
    gcn_csv = os.path.join(MPNN_DIR, f'GCN_data_{args.split}.csv')

    print(f'Device    : {device}')
    print(f'Split     : {args.split}')
    print(f'beam_size : {args.beam_size}')

    # ── Phase 1: Inference ───────────────────────────────────────────────────
    if args.skip_inference and os.path.exists(inf_csv):
        print(f'\nSkipping inference — loading {inf_csv}')
    else:
        print('\n--- Phase 1: Inference (beam search) ---')

        vocab_path       = os.path.join(SAVE_DIR, 'unified_vocab.json')
        class_vocab_path = os.path.join(SAVE_DIR, 'class_vocab.json')

        uid2smi, smi2uid = load_unified_vocab(vocab_path)
        class2idx        = load_class_vocab(class_vocab_path)
        V = len(uid2smi)
        print(f'  Agent vocab : {V}')
        print(f'  Class vocab : {len(class2idx)+1} (incl. UNK)')

        cat_id2uid, solv_id2uid, reag_id2uid = build_vocab_maps(LABEL_DIR, smi2uid)
        cat_list, solv_list, reag_list        = load_label_lists(LABEL_DIR)

        model_path = os.path.join(SAVE_DIR, 'best_model.pt')
        print(f'Loading model: {model_path}')
        ckpt = torch.load(model_path, map_location=device)

        set_reaction(True, 'reac_diff')

        model = RoleAgnosticGNNModelRxnClass(
            graph_dim     = ckpt.get('graph_dim', 1024),
            vocab_size    = ckpt.get('vocab_size', V),
            n_classes     = ckpt.get('n_classes', len(class2idx) + 1),
            device        = device,
            class_emb_dim = ckpt.get('class_emb', 256),
            agent_emb_dim = ckpt.get('agent_emb', 512),
        ).to(device)
        model.load_state_dict(ckpt['model_state'])
        model.eval()

        gcn_df        = pd.read_csv(gcn_csv)
        class_indices = build_class_indices(gcn_df, args.split, DATA_DIR, class2idx)
        unk_count     = sum(1 for c in class_indices if c == UNK_CLASS)
        print(f'  Class UNK: {unk_count}/{len(class_indices)} ({100*unk_count/len(class_indices):.1f}%)')

        run_inference(model, gcn_df, class_indices,
                      uid2smi, cat_id2uid, solv_id2uid, reag_id2uid,
                      cat_list, solv_list, reag_list,
                      device, ckpt.get('vocab_size', V), args.beam_size, inf_csv,
                      rxn_batch=args.rxn_batch)

    # copy inference CSV to InferenceResults/{split}/
    import shutil
    os.makedirs(IR_DIR, exist_ok=True)
    shutil.copy2(inf_csv, ir_csv)
    print(f'  Copied → {ir_csv}')
    print('\nPhase 1 complete.')


if __name__ == '__main__':
    main()
