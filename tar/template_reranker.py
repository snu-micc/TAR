"""
template_reranker.py
====================
Template Prior × DETR Score Re-ranker.

Hierarchy (most specific → least):
  r2 → r1 → r0 → DETR-only fallback

combined_score = detr_log_score + lam * log P(set | template)

P(set | template) is estimated by building the library at runtime from
train+val CSVs (same approach as single_autoregressive/template_reranking.py),
so there is no risk of test-data leakage from pre-built library files.

Candidate key: (cat_tuple, solv_tuple, reag_tuple) — same as single_autoregressive.
Laplace smoothing (alpha=0.5) applied within matched template level.
"""

import os
import math
import pandas as pd
from collections import defaultdict, Counter


SMOOTH       = 0.5     # Laplace pseudocount
LOG_FALLBACK = -0.0   # used when no template level matched


# ─────────────────────────────────────────────────────────────────────────────
# Library building (identical logic to single_autoregressive)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_smiles_field(val):
    """Split a semicolon-separated SMILES field into a sorted tuple."""
    if pd.isna(val) or str(val).strip() == '':
        return ()
    return tuple(sorted(s.strip() for s in str(val).split(';') if s.strip()))


RADII = ('r0', 'r1', 'r2', 'r3', 'r4', 'r5')
RADII_DESC = tuple(reversed(RADII))   # r5 → r4 → r3 → r2 → r1 → r0


def build_library(csv_paths: list) -> dict:
    """
    Build template_lib[r][tmpl_str] = Counter{(cat_tuple, solv_tuple, reag_tuple): count}
    from one or more train/val CSV files.

    Args:
        csv_paths: list of paths to data_{split}_template.csv files
    Returns:
        {r: {template_str: Counter{key: count}}}  for r in RADII
    """
    template_lib = {r: defaultdict(Counter) for r in RADII}
    tmpl_cols    = [f'template_{r}' for r in RADII]
    for path in csv_paths:
        print(f'  [TemplateReranker] Loading {path}')
        df = pd.read_csv(path, usecols=['Catalyst', 'Solvent', 'Agent'] + tmpl_cols,
                         low_memory=False)
        for _, row in df.iterrows():
            key = (
                _parse_smiles_field(row.get('Catalyst')),
                _parse_smiles_field(row.get('Solvent')),
                _parse_smiles_field(row.get('Agent')),
            )
            for r in RADII:
                tmpl = row.get(f'template_{r}')
                if pd.notna(tmpl) and str(tmpl).strip():
                    template_lib[r][str(tmpl).strip()][key] += 1

    for r in RADII_DESC:
        print(f'  [TemplateReranker] {r}: {len(template_lib[r]):,} unique templates')
    return template_lib


# ─────────────────────────────────────────────────────────────────────────────
# Key conversion
# ─────────────────────────────────────────────────────────────────────────────

def candidate_to_key(candidate: dict) -> tuple:
    """
    Convert a DETR candidate dict → (cat_tuple, solv_tuple, reag_tuple).

    candidate: {'cat': frozenset of SMILES,
                'reag': frozenset of SMILES,
                'solv': frozenset of SMILES}
    """
    cat  = tuple(sorted(candidate.get('cat',  frozenset())))
    solv = tuple(sorted(candidate.get('solv', frozenset())))
    reag = tuple(sorted(candidate.get('reag', frozenset())))
    return (cat, solv, reag)


# ─────────────────────────────────────────────────────────────────────────────
# Reranker
# ─────────────────────────────────────────────────────────────────────────────

class TemplateReranker:
    """
    Template Prior × DETR Score Re-ranker.

    Hierarchy: r5 (most specific) → r4 → r3 → r2 → r1 → r0 → fallback.

    Usage
    -----
    reranker = TemplateReranker(data_dir='../Data')

    # per sample — pass a dict of all available template strings:
    templates = {'r0': '...', 'r1': '...', 'r2': '...', 'r3': '...', 'r4': '...', 'r5': '...'}
    reranked = reranker.rerank(candidates, templates=templates, lam=1.0)
    """

    RADII        = RADII_DESC   # ('r5', 'r4', 'r3', 'r2', 'r1', 'r0')
    SMOOTH       = SMOOTH
    LOG_FALLBACK = LOG_FALLBACK

    def __init__(self, csv_paths: list = None, data_dir: str = None):
        if csv_paths is None:
            if data_dir is None:
                data_dir = os.path.normpath(
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), '../Data'))
            csv_paths = [
                os.path.join(data_dir, 'data_train_template.csv'),
                os.path.join(data_dir, 'data_val_template.csv'),
            ]
        self._libs = build_library(csv_paths)

    def log_prior(self,
                  candidate_key: tuple,
                  templates: dict) -> tuple:
        """
        Compute log P(set | template) using the hierarchy r5 → r4 → ... → r0.

        Args:
            candidate_key : (cat_tuple, solv_tuple, reag_tuple)
            templates     : dict mapping radius name → template string
                            e.g. {'r0': '...', 'r1': '...', ..., 'r5': '...'}

        Returns (log_prob, radius_used).
          radius_used is None when no template level matched (fallback).
        """
        for radius in self.RADII:
            tmpl = templates.get(radius)
            if not tmpl or not isinstance(tmpl, str) or not tmpl.strip():
                continue
            bucket = self._libs[radius].get(tmpl.strip())
            if bucket is None:
                continue
            count  = bucket.get(candidate_key, 0)
            total  = sum(bucket.values())
            n_keys = len(bucket)
            log_p  = math.log(
                (count + self.SMOOTH) /
                (total + self.SMOOTH * (n_keys + 1))
            )
            return log_p, radius

        return self.LOG_FALLBACK, None

    def rerank(self,
               candidates: list,
               templates: dict,
               lam: float = 1.0) -> list:
        """
        Re-rank DETR candidates by:
            combined_score = detr_log_score + lam * log P(set | template)

        Args:
            candidates : list of dicts {'cat', 'reag', 'solv', 'log_score', ...}
            templates  : dict {'r0': tmpl_str, ..., 'r5': tmpl_str}
            lam        : weight on the template prior
        """
        scored = []
        for cand in candidates:
            key           = candidate_to_key(cand)
            log_p, radius = self.log_prior(key, templates)
            combined      = cand['log_score'] + lam * log_p
            scored.append({
                **cand,
                'template_log_prior': log_p,
                'radius_used':        radius,
                'combined_score':     combined,
            })

        return sorted(scored, key=lambda x: x['combined_score'], reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# Role-agnostic variant  (for models that predict a flat agent set)
# ─────────────────────────────────────────────────────────────────────────────

def build_library_flat(csv_paths: list) -> dict:
    """
    Build a flat template library for role-agnostic models.

    Key: sorted tuple of ALL agents (union of Catalyst + Solvent + Agent),
    so it can be matched against a role-agnostic predicted set.

    Returns:
        {r: {template_str: Counter{flat_tuple: count}}}  for r in ('r0','r1','r2')
    """
    template_lib = {r: defaultdict(Counter) for r in ('r0', 'r1', 'r2', 'r3', 'r4', 'r5')}
    for path in csv_paths:
        print(f'  [FlatTemplateReranker] Loading {path}')
        df = pd.read_csv(path, usecols=['Catalyst', 'Solvent', 'Agent',
                                        'template_r0', 'template_r1', 'template_r2', 'template_r3', 'template_r4', 'template_r5'],
                         low_memory=False)
        for _, row in df.iterrows():
            cat  = set(_parse_smiles_field(row.get('Catalyst')))
            solv = set(_parse_smiles_field(row.get('Solvent')))
            reag = set(_parse_smiles_field(row.get('Agent')))
            flat = tuple(sorted(cat | solv | reag))
            for r in ('r0', 'r1', 'r2',  'r3', 'r4', 'r5'):
                tmpl = row.get(f'template_{r}')
                if pd.notna(tmpl) and str(tmpl).strip():
                    template_lib[r][str(tmpl).strip()][flat] += 1

    for r in ( 'r5', 'r4', 'r3', 'r2', 'r1', 'r0'):
        print(f'  [FlatTemplateReranker] {r}: {len(template_lib[r]):,} unique templates')
    return template_lib


class RoleAgnosticTemplateReranker(TemplateReranker):
    """
    Template reranker for role-agnostic models (e.g. roleagnostic_gnn).

    Candidates supply a flat set of agent SMILES; the template library key
    is the sorted tuple of all agents regardless of role.

    Usage
    -----
    reranker = RoleAgnosticTemplateReranker(data_dir='../Data')

    templates = {'r0': '...', 'r1': '...', ..., 'r5': '...'}
    reranked = reranker.rerank(candidates, templates=templates, lam=1.0)
    """

    def __init__(self, csv_paths: list = None, data_dir: str = None):
        if csv_paths is None:
            if data_dir is None:
                data_dir = os.path.normpath(
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), '../Data'))
            csv_paths = [
                os.path.join(data_dir, 'data_train_template.csv'),
                os.path.join(data_dir, 'data_val_template.csv'),
            ]
        self._libs = build_library_flat(csv_paths)

    def log_prior_flat(self, flat_key: tuple, templates: dict) -> tuple:
        """
        Compute log P(agent_set | template) using flat agent-union key.

        Args:
            flat_key  : sorted tuple of all agent SMILES
            templates : dict {'r0': tmpl_str, ..., 'r5': tmpl_str}

        Returns (log_prob, radius_used).
        """
        for radius in self.RADII:
            tmpl = templates.get(radius)
            if not tmpl or not isinstance(tmpl, str) or not tmpl.strip():
                continue
            bucket = self._libs[radius].get(tmpl.strip())
            if bucket is None:
                continue
            count  = bucket.get(flat_key, 0)
            total  = sum(bucket.values())
            n_keys = len(bucket)
            log_p  = math.log(
                (count + self.SMOOTH) /
                (total + self.SMOOTH * (n_keys + 1))
            )
            return log_p, radius
        return self.LOG_FALLBACK, None

    def rerank(self,
               candidates: list,
               templates: dict,
               lam: float = 1.0) -> list:
        """
        Re-rank role-agnostic candidates by:
            combined_score = log_score + lam * log P(agent_set | template)

        Args:
            candidates : list of dicts with 'agents' (frozenset) and 'log_score'
            templates  : dict {'r0': tmpl_str, ..., 'r5': tmpl_str}
            lam        : weight on the template prior
        """
        scored = []
        for cand in candidates:
            flat_key      = tuple(sorted(cand.get('agents', frozenset())))
            log_p, radius = self.log_prior_flat(flat_key, templates)
            combined      = cand['log_score'] + lam * log_p
            scored.append({
                **cand,
                'template_log_prior': log_p,
                'radius_used':        radius,
                'combined_score':     combined,
            })
        return sorted(scored, key=lambda x: x['combined_score'], reverse=True)
