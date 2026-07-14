"""
train_baseline_gao2018.py
=========================
Faithful re-implementation of the autoregressive reaction condition
recommendation model from:

  Gao et al., "Using Machine Learning To Predict Suitable Conditions
  for Organic Reactions", ACS Cent. Sci. 2018, 4, 1465-1476.

Architecture (from Methods / Figure 7):
  Input  : product_FP (16384) | reaction_FP (prod - reac, 16384) → 32768
  Backbone (Dense FP):
      FC(32768 → 1000, ReLU)
      Dropout(0.5)
      FC(1000 → 1000, ReLU)
      → Dense FP [B, 1000]
  Sequential heads — each is:
      FC(context_dim → 300, ReLU) → FC(300 → vocab_size)
  Prediction order (teacher forcing during training):
      1. Catalyst  : Dense FP               → cat logits
      2. Solvent1  : Dense FP + cat_onehot  → solv1 logits
      3. Solvent2  : Dense FP + cat + s1    → solv2 logits
      4. Reagent1  : Dense FP + cat + s1+s2 → reag1 logits
      5. Reagent2  : Dense FP + cat+s1+s2+r1→ reag2 logits
  Loss: sum of cross-entropy losses for all 5 heads
  Inference: hard selection (argmax → one-hot) for next step context

Data:
  Uses existing GCN_data_*.csv from Data/MPNN_data/
  Labels (row index = class id, row 0 = NULL/none):
      Data/labels/cat_labels.csv   → cat_dim
      Data/labels/solv_labels.csv  → solv_dim
      Data/labels/reag_labels.csv  → reag_dim

Usage:
    python train_baseline_gao2018.py
    python train_baseline_gao2018.py --fp_size 2048   # faster for debugging

● train_baseline_gao2018.py — Detailed Explanation vs. Paper

  ---
  1. Input Representation

  Paper:
  ▎ "Morgan circular fingerprints, radius 2, bit vectors with length 16,384"
  ▎ "A reaction fingerprint is calculated as the difference between product fingerprint and reactant fingerprint"
  ▎ "The neural network takes the product fingerprint and reaction fingerprint as two inputs"

  Code (rxn_to_fp):
  fp_r = Morgan(reactant, radius=2, nBits=16384)
  fp_p = Morgan(product,  radius=2, nBits=16384)
  return np.concatenate([fp_p, fp_p - fp_r])   # [product | diff]  → size 32768
  - fp_p = product fingerprint (16384)
  - fp_p - fp_r = reaction fingerprint = substructures that change (16384)
  - Total input: 32768

  ---
  2. Backbone (Dense FP)

  Paper:
  ▎ "Reaction and product fingerprints are concatenated and passed through two fully connected layers (ReLU activation, size 1000;
  ReLU activation, size 1000, with a 0.5 dropout) to generate a dense representation (Dense FP)"

  Code (corrected to match Figure 7):
  self.backbone = nn.Sequential(
      nn.Linear(32768, 1000),  # FC layer 1
      nn.ReLU(),
      nn.Linear(1000, 1000),   # FC layer 2
      nn.ReLU(),
      nn.Dropout(0.5),         # dropout AFTER both 1000-dim layers (Figure 7)
      nn.Linear(1000, 300),    # FC layer 3 (Figure 7)
      nn.ReLU(),
      nn.Linear(300, 300),     # FC layer 4 (Figure 7)
      nn.Tanh(),               # Dense FP [B, 300]  ← paper uses Tanh here
  )

  NOTE: The paper text says "two FC layers of 1000" but Figure 7 shows 4 layers
  total with a final Tanh(300). The code follows Figure 7 as the architecture diagram.

  ---
  3. Prediction Heads

  Paper:
  ▎ "Dense FP is passed through two fully connected layers (ReLU activation, size 300; Softmax activation, size N) to predict the
  catalyst"

  Each head is implemented as PredictionHead:
  class PredictionHead(nn.Module):
      def __init__(self, in_dim, vocab_size, hidden=300):
          self.net = nn.Sequential(
              nn.Linear(in_dim, 300),   # FC → 300, ReLU
              nn.ReLU(),
              nn.Linear(300, vocab_size) # FC → vocab (softmax applied in loss)
          )

  ---
  4. Sequential (Autoregressive) Prediction

  Paper:
  ▎ "Predictions are made sequentially so that information from precedent elements can be incorporated"

  Order: Catalyst → Solvent1 → Solvent2 → Reagent1 → Reagent2

  Each step's head receives Dense FP concatenated with one-hot vectors of all previous predictions:

  ┌──────────┬────────────────────────────────────────────────────────────────────────┐
  │   Step   │                            Head input size                             │
  ├──────────┼────────────────────────────────────────────────────────────────────────┤
  │ Catalyst │ Dense FP (300)                                                         │
  ├──────────┼────────────────────────────────────────────────────────────────────────┤
  │ Solvent1 │ Dense FP + cat_onehot (300 + cat_dim)                                  │
  ├──────────┼────────────────────────────────────────────────────────────────────────┤
  │ Solvent2 │ Dense FP + cat + s1 (300 + cat_dim + solv_dim)                         │
  ├──────────┼────────────────────────────────────────────────────────────────────────┤
  │ Reagent1 │ Dense FP + cat + s1 + s2 (300 + cat_dim + 2×solv_dim)                  │
  ├──────────┼────────────────────────────────────────────────────────────────────────┤
  │ Reagent2 │ Dense FP + cat + s1 + s2 + r1 (300 + cat_dim + 2×solv_dim + reag_dim)  │
  └──────────┴────────────────────────────────────────────────────────────────────────┘

  Code:
  self.cat_head   = PredictionHead(1000,                                   cat_dim)
  self.solv1_head = PredictionHead(1000 + cat_dim,                         solv_dim)
  self.solv2_head = PredictionHead(1000 + cat_dim + solv_dim,              solv_dim)
  self.reag1_head = PredictionHead(1000 + cat_dim + 2*solv_dim,            reag_dim)
  self.reag2_head = PredictionHead(1000 + cat_dim + 2*solv_dim + reag_dim, reag_dim)

  ---
  5. Teacher Forcing

  Paper:
  ▎ "We apply the teacher forcing technique during training. It takes the ground truth output, instead of the predicted ones, as the
   input for the next prediction task."

  Code (during training — gt_* are ground truth indices):
  logits = model(X,
                 gt_cat=labels['cat'],    # ground truth → one-hot context
                 gt_s1=labels['solv0'],
                 gt_s2=labels['solv1'],
                 gt_r1=labels['reag0'])

  During inference — gt_* are omitted, so argmax of previous prediction is used:
  # inside forward(), when gt_cat is None:
  cat_ctx = F.one_hot(cat_logits.argmax(dim=-1), self.cat_dim).float()

  ---
  6. Loss Function

  Paper:
  ▎ "Categorical cross-entropy is used as the loss function for the classification problems"
  ▎ Temperature prediction (MSE) is not included in this script since the data doesn't have temperature labels.

  Code:
  def compute_loss(logits_dict, labels, device):
      total = 0
      for key, label_key in [('cat','cat'), ('solv1','solv0'), ('solv2','solv1'),
                              ('reag1','reag0'), ('reag2','reag1')]:
          total += F.cross_entropy(logits_dict[key], labels[label_key])
      return total  # sum of 5 cross-entropy losses

  ---
  7. Training Loop & Early Stopping

  Paper:
  ▎ "Training continues until the validation loss does not improve over five epochs"
  ▎ "Data split: 80/10/10 train/val/test"

  Code:
  - patience=5 (matches paper)
  - Best model saved when val_loss improves
  - Training log saved to gao2018/train_log.csv

  ---

"""

import os, argparse, time, hashlib, multiprocessing
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, '..', '..', 'Data', 'MPNN_data')
LABEL_DIR  = os.path.join(BASE_DIR, '..', '..', 'Data', 'labels')
SAVE_DIR   = os.path.join(BASE_DIR, 'gao2018')
os.makedirs(SAVE_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters (paper defaults)
# ─────────────────────────────────────────────────────────────────────────────
FP_SIZE      = 16384   # paper uses 16384; set --fp_size 2048 to run faster
FP_RADIUS    = 2
BACKBONE_DIM = 1000    # two FC layers of size 1000 (paper text)
DENSE_DIM    = 300     # Figure 7: FC(300,ReLU)→Tanh(300) after the two 1000-dim layers
HEAD_HIDDEN  = 300     # hidden size inside each prediction head (paper)
DROPOUT      = 0.5
BATCH_SIZE   = 512
LR           = 1e-3
EPOCHS       = 100
PATIENCE     = 5       # early stopping patience (paper: "until validation loss does not improve over five epochs")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Fingerprint computation
# ─────────────────────────────────────────────────────────────────────────────

def rxn_to_fp(rxn_smiles: str, nbits: int, radius: int):
    """
    Reaction fingerprint: [product_FP | (product_FP - reactant_FP)]
    Returns float32 numpy array of size 2*nbits, or None on failure.
    """
    try:
        parts = rxn_smiles.strip().split('>>')
        r_smi = parts[0].strip()
        p_smi = parts[-1].strip()
        r_mol = Chem.MolFromSmiles(r_smi)
        p_mol = Chem.MolFromSmiles(p_smi)
        if r_mol is None or p_mol is None:
            return None
        fp_r = np.array(AllChem.GetMorganFingerprintAsBitVect(r_mol, radius, nBits=nbits), dtype=np.float32)
        fp_p = np.array(AllChem.GetMorganFingerprintAsBitVect(p_mol, radius, nBits=nbits), dtype=np.float32)
        return np.concatenate([fp_p, fp_p - fp_r])   # [product | diff]
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


def build_features(df: pd.DataFrame, nbits: int, radius: int, cache_dir: str = None):
    """
    Returns X [N, 2*nbits] float32 (memory-mapped) and Y dict of int64 arrays.
    Rows where FP computation fails are dropped.

    FPs are written incrementally via imap+memmap so the full matrix is never
    held in RAM. Cache format: <key>_X.npy (mmap-loadable) + <key>_meta.npz.
    """
    # ── Cache lookup ──────────────────────────────────────────────────────────
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        key = hashlib.md5(
            ("|".join(df['reaction'].astype(str)) + f"|{nbits}|{radius}").encode()
        ).hexdigest()
        x_path   = os.path.join(cache_dir, f'fp_{key}_X.npy')
        meta_path = os.path.join(cache_dir, f'fp_{key}_meta.npz')
        if os.path.exists(x_path) and os.path.exists(meta_path):
            print(f'  Loading cached fingerprints (mmap) ...')
            meta    = np.load(meta_path)
            n_valid = int(meta['n_valid'])
            X       = np.load(x_path, mmap_mode='r')[:n_valid]
            Y       = {k: meta[k] for k in ['cat', 'solv0', 'solv1', 'reag0', 'reag1']}
            print(f'  {n_valid:,} valid samples loaded from cache, input_dim={X.shape[1]}')
            return X, Y
    else:
        x_path = meta_path = None

    # ── Parallel fingerprint computation (imap → memmap, low RAM) ─────────────
    smiles_list = df['reaction'].astype(str).tolist()
    label_keys  = ['cat', 'solv0', 'solv1', 'reag0', 'reag1']
    labels_arr  = {k: df[k].tolist() for k in label_keys}
    N        = len(smiles_list)
    feat_dim = 2 * nbits

    n_workers = min(multiprocessing.cpu_count() // 2, 12)
    print(f'  Computing fingerprints (nbits={nbits}, radius={radius}, workers={n_workers}) ...')
    t0 = time.time()

    # Pre-allocate memmap (shape N; only first n_valid rows will be filled)
    mm_path = x_path if x_path else f'_fp_tmp_{os.getpid()}.npy'
    mm      = np.lib.format.open_memmap(mm_path, mode='w+', dtype=np.float32, shape=(N, feat_dim))

    valid_src = []   # original row indices of valid reactions
    write_idx = 0
    with multiprocessing.Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(nbits, radius),
    ) as pool:
        for i, fp in enumerate(pool.imap(_worker_fn, smiles_list, chunksize=512)):
            if fp is not None:
                mm[write_idx] = fp
                valid_src.append(i)
                write_idx += 1
            if (i + 1) % 50_000 == 0:
                print(f'    {i+1:,}/{N:,} processed ...')

    mm.flush()
    del mm
    n_valid = write_idx
    print(f'  FP computation done in {time.time()-t0:.1f}s')

    Y = {k: np.array([int(labels_arr[k][i]) for i in valid_src], dtype=np.int64)
         for k in label_keys}
    print(f'  {n_valid:,} valid samples, input_dim={feat_dim}')

    # ── Save cache & return mmap ──────────────────────────────────────────────
    if x_path:
        np.savez_compressed(meta_path, n_valid=np.array(n_valid), **Y)
        print(f'  Fingerprints cached -> {x_path}')
        X = np.load(x_path, mmap_mode='r')[:n_valid]
    else:
        # No cache dir: keep temp file as mmap-backed array
        X = np.lib.format.open_memmap(mm_path, mode='r', shape=(N, feat_dim))[:n_valid]

    return X, Y


# ─────────────────────────────────────────────────────────────────────────────
# 2. Dataset
# ─────────────────────────────────────────────────────────────────────────────

class ReactionDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X)
        self.Y = {k: torch.from_numpy(v) for k, v in Y.items()}

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], {k: v[idx] for k, v in self.Y.items()}


# ─────────────────────────────────────────────────────────────────────────────
# 3. Model  (Section "Model Structure" in the paper)
# ─────────────────────────────────────────────────────────────────────────────

class PredictionHead(nn.Module):
    """Two FC layers: FC(in→300, ReLU) → FC(300→vocab_size)."""
    def __init__(self, in_dim: int, vocab_size: int, hidden: int = HEAD_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, vocab_size),
        )

    def forward(self, x):
        return self.net(x)


class Gao2018Model(nn.Module):
    """
    Autoregressive reaction condition predictor (Gao et al. 2018).

    cat_dim   : number of catalyst classes (incl. NULL)
    solv_dim  : number of solvent classes   (shared for s1 and s2)
    reag_dim  : number of reagent classes   (shared for r1 and r2)
    input_dim : 2 * fp_size
    """
    def __init__(self, input_dim: int, cat_dim: int, solv_dim: int, reag_dim: int,
                 backbone_dim: int = BACKBONE_DIM, dense_dim: int = DENSE_DIM,
                 dropout: float = DROPOUT):
        super().__init__()
        self.cat_dim  = cat_dim
        self.solv_dim = solv_dim
        self.reag_dim = reag_dim

        # ── Backbone (Figure 7) ───────────────────────────────────────────────
        # FC(1000,ReLU) → FC(1000,ReLU) → Dropout(0.5) → FC(300,ReLU) → Tanh(300)
        # Dropout comes AFTER both 1000-dim layers (Figure 7, not between them).
        # The final Tanh(300) output is the Dense FP used by all prediction heads.
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, backbone_dim),    # FC1: 32768→1000
            nn.ReLU(),
            nn.Linear(backbone_dim, backbone_dim), # FC2: 1000→1000
            nn.ReLU(),
            nn.Dropout(dropout),                   # dropout after both FC layers
            nn.Linear(backbone_dim, dense_dim),    # FC3: 1000→300
            nn.ReLU(),
            nn.Linear(dense_dim, dense_dim),       # FC4: 300→300
            nn.Tanh(),                             # Dense FP [B, 300]
        )

        D = dense_dim
        # ── Prediction heads ─────────────────────────────────────────────────
        # Each head sees Dense FP concatenated with one-hot context from
        # all previously predicted roles (teacher forcing during training).
        self.cat_head   = PredictionHead(D,                                 cat_dim)
        self.solv1_head = PredictionHead(D + cat_dim,                       solv_dim)
        self.solv2_head = PredictionHead(D + cat_dim + solv_dim,            solv_dim)
        self.reag1_head = PredictionHead(D + cat_dim + 2*solv_dim,          reag_dim)
        self.reag2_head = PredictionHead(D + cat_dim + 2*solv_dim + reag_dim, reag_dim)

    def forward(self, x, gt_cat=None, gt_s1=None, gt_s2=None, gt_r1=None):
        """
        Training (teacher forcing): pass all gt_* tensors [B] (ground-truth indices).
        Inference: omit gt_* — previous argmax prediction used as context.

        Returns dict of logits for each role.
        """
        dense = self.backbone(x)   # [B, D]

        # ── Catalyst ─────────────────────────────────────────────────────────
        cat_logits = self.cat_head(dense)           # [B, cat_dim]

        if gt_cat is not None:
            cat_ctx = F.one_hot(gt_cat, self.cat_dim).float()
        else:
            cat_ctx = F.one_hot(cat_logits.argmax(dim=-1), self.cat_dim).float()

        # ── Solvent 1 ─────────────────────────────────────────────────────────
        s1_logits = self.solv1_head(torch.cat([dense, cat_ctx], dim=-1))

        if gt_s1 is not None:
            s1_ctx = F.one_hot(gt_s1, self.solv_dim).float()
        else:
            s1_ctx = F.one_hot(s1_logits.argmax(dim=-1), self.solv_dim).float()

        # ── Solvent 2 ─────────────────────────────────────────────────────────
        s2_logits = self.solv2_head(torch.cat([dense, cat_ctx, s1_ctx], dim=-1))

        if gt_s2 is not None:
            s2_ctx = F.one_hot(gt_s2, self.solv_dim).float()
        else:
            s2_ctx = F.one_hot(s2_logits.argmax(dim=-1), self.solv_dim).float()

        # ── Reagent 1 ─────────────────────────────────────────────────────────
        r1_logits = self.reag1_head(torch.cat([dense, cat_ctx, s1_ctx, s2_ctx], dim=-1))

        if gt_r1 is not None:
            r1_ctx = F.one_hot(gt_r1, self.reag_dim).float()
        else:
            r1_ctx = F.one_hot(r1_logits.argmax(dim=-1), self.reag_dim).float()

        # ── Reagent 2 ─────────────────────────────────────────────────────────
        r2_logits = self.reag2_head(torch.cat([dense, cat_ctx, s1_ctx, s2_ctx, r1_ctx], dim=-1))

        return {
            'cat':   cat_logits,
            'solv1': s1_logits,
            'solv2': s2_logits,
            'reag1': r1_logits,
            'reag2': r2_logits,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Training helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_loss(logits_dict, labels, device):
    """Sum of cross-entropy losses for all 5 roles."""
    total = torch.tensor(0.0, device=device)
    for key, label_key in [('cat','cat'), ('solv1','solv0'), ('solv2','solv1'),
                            ('reag1','reag0'), ('reag2','reag1')]:
        total += F.cross_entropy(logits_dict[key], labels[label_key].to(device))
    return total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss   = 0.0
    correct      = {k: 0 for k in ['cat','solv1','solv2','reag1','reag2']}
    n_samples    = 0

    for X, labels in loader:
        X = X.to(device)
        gt = {k: labels[k].to(device) for k in labels}

        logits = model(X,
                       gt_cat=gt['cat'],
                       gt_s1=gt['solv0'],
                       gt_s2=gt['solv1'],
                       gt_r1=gt['reag0'])
        loss   = compute_loss(logits, labels, device)
        B      = X.size(0)
        total_loss += loss.item() * B
        n_samples  += B

        for key, lk in [('cat','cat'), ('solv1','solv0'), ('solv2','solv1'),
                        ('reag1','reag0'), ('reag2','reag1')]:
            preds = logits[key].argmax(dim=-1)
            correct[key] += (preds == gt[lk]).sum().item()

    avg_loss = total_loss / n_samples
    acc = {k: correct[k] / n_samples for k in correct}
    return avg_loss, acc


# ─────────────────────────────────────────────────────────────────────────────
# 5. Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp_size', type=int, default=FP_SIZE,
                        help='Morgan FP bit vector size (paper: 16384)')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LR)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--patience', type=int, default=PATIENCE)
    args = parser.parse_args()

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {DEVICE}')

    # ── Load label vocab sizes ────────────────────────────────────────────────
    cat_df   = pd.read_csv(os.path.join(LABEL_DIR, 'cat_labels.csv'))
    solv_df  = pd.read_csv(os.path.join(LABEL_DIR, 'solv_labels.csv'))
    reag_df  = pd.read_csv(os.path.join(LABEL_DIR, 'reag_labels.csv'))
    cat_dim  = len(cat_df)
    solv_dim = len(solv_df)
    reag_dim = len(reag_df)
    print(f'Vocab sizes — cat: {cat_dim}, solv: {solv_dim}, reag: {reag_dim}')

    # ── Load & featurize data ─────────────────────────────────────────────────
    for split in ['train', 'val', 'test']:
        path = os.path.join(DATA_DIR, f'GCN_data_{split}.csv')
        if not os.path.exists(path):
            print(f'ERROR: {path} not found'); return

    fp_cache_dir = os.path.join(SAVE_DIR, 'fp_cache')
    datasets = {}
    for split in ['train', 'val', 'test']:
        path = os.path.join(DATA_DIR, f'GCN_data_{split}.csv')
        print(f'\nLoading {split} ...')
        df = pd.read_csv(path)
        X, Y = build_features(df, args.fp_size, FP_RADIUS, cache_dir=fp_cache_dir)
        datasets[split] = ReactionDataset(X, Y)

    loaders = {
        split: DataLoader(datasets[split],
                          batch_size=args.batch_size,
                          shuffle=(split == 'train'),
                          num_workers=0,
                          pin_memory=True)
        for split in datasets
    }

    # ── Build model ───────────────────────────────────────────────────────────
    input_dim = 2 * args.fp_size
    model = Gao2018Model(input_dim, cat_dim, solv_dim, reag_dim).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'\nModel parameters: {n_params:,}')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float('inf')
    patience_counter = 0
    model_path = os.path.join(SAVE_DIR, 'best_model.pt')
    log_rows = []

    print(f'\nTraining for up to {args.epochs} epochs (patience={args.patience}) ...\n')
    for epoch in range(1, args.epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        t0 = time.time()
        for X, labels in loaders['train']:
            X = X.to(DEVICE)
            optimizer.zero_grad()
            logits = model(X,
                           gt_cat=labels['cat'].to(DEVICE),
                           gt_s1=labels['solv0'].to(DEVICE),
                           gt_s2=labels['solv1'].to(DEVICE),
                           gt_r1=labels['reag0'].to(DEVICE))
            loss = compute_loss(logits, labels, DEVICE)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X.size(0)

        train_loss /= len(datasets['train'])

        # ── Validate ──────────────────────────────────────────────────────────
        val_loss, val_acc = evaluate(model, loaders['val'], DEVICE)
        elapsed = time.time() - t0

        print(f'Epoch {epoch:03d}  '
              f'train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  '
              f'val_acc(cat={val_acc["cat"]:.3f} s1={val_acc["solv1"]:.3f} '
              f's2={val_acc["solv2"]:.3f} r1={val_acc["reag1"]:.3f} r2={val_acc["reag2"]:.3f})  '
              f'[{elapsed:.1f}s]')

        log_rows.append({
            'epoch': epoch,
            'train_loss': round(train_loss, 6),
            'val_loss': round(val_loss, 6),
            **{f'val_acc_{k}': round(v, 6) for k, v in val_acc.items()},
        })

        # ── Early stopping ─────────────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
            print(f'  -> Saved best model (val_loss={val_loss:.4f})')
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f'Early stopping at epoch {epoch}.')
                break

    # ── Test ──────────────────────────────────────────────────────────────────
    print('\nLoading best model for test evaluation ...')
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    test_loss, test_acc = evaluate(model, loaders['test'], DEVICE)
    print(f'Test  loss={test_loss:.4f}  '
          f'acc(cat={test_acc["cat"]:.4f} s1={test_acc["solv1"]:.4f} '
          f's2={test_acc["solv2"]:.4f} r1={test_acc["reag1"]:.4f} r2={test_acc["reag2"]:.4f})')

    # ── Save training log ──────────────────────────────────────────────────────
    log_df = pd.DataFrame(log_rows)
    log_path = os.path.join(SAVE_DIR, 'train_log.csv')
    log_df.to_csv(log_path, index=False)
    print(f'Training log saved -> {log_path}')


if __name__ == '__main__':
    main()
