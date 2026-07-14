"""
analyze_template_entropy.py
===========================
Shannon Entropy analysis of the reaction template condition library.

For each template T at each radius r:
  H(T) = -Σ P(r_i | T) * log2 P(r_i | T)

where P(r_i | T) = count(r_i, T) / Σ_j count(r_j, T)

Output
------
  entropy_analysis.csv          — per-template stats
  entropy_hist.png              — entropy distribution by radius
  entropy_vs_freq.png           — scatter: log(total_count) vs entropy
  entropy_by_radius.png         — mean/median entropy per radius
  top10_certainty.txt           — lowest-entropy, high-count templates
  top10_ambiguity.txt           — highest-entropy, high-count templates

Usage
-----
  cd data_preparation
  python step4_compute_entropy.py
  python step4_compute_entropy.py --lib_path ../Data/flat_template_library.pkl
"""

import os
import sys
import math
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.normpath(os.path.join(SCRIPT_DIR, '../Data'))

RADII = ('r0', 'r1', 'r2', 'r3', 'r4', 'r5')

try:
    import seaborn as sns
    HAS_SEABORN = True
    sns.set_theme(style='whitegrid', font_scale=1.1)
except ImportError:
    HAS_SEABORN = False


# ── entropy ───────────────────────────────────────────────────────────────────

def shannon_entropy(counts):
    """Shannon entropy in bits (log2) from a list/array of counts."""
    counts = np.array(counts, dtype=float)
    total  = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


# ── build records ─────────────────────────────────────────────────────────────

def build_records(libs):
    records = []
    for r in RADII:
        lib = libs[r]
        for tmpl, bucket in lib.items():
            counts       = list(bucket.values())
            total_count  = sum(counts)
            unique_count = len(counts)
            H            = shannon_entropy(counts)
            records.append({
                'radius':              r,
                'template':            tmpl,
                'total_count':         total_count,
                'unique_reagent_sets': unique_count,
                'entropy':             H,
            })
    return pd.DataFrame(records)


# ── plots ─────────────────────────────────────────────────────────────────────

COLORS = {
    'r0': '#1f77b4', 'r1': '#ff7f0e', 'r2': '#2ca02c',
    'r3': '#d62728', 'r4': '#9467bd', 'r5': '#8c564b',
}


def plot_entropy_hist(df, out_path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=False)
    axes = axes.flatten()
    for ax, r in zip(axes, RADII):
        sub = df[df['radius'] == r]['entropy']
        ax.hist(sub, bins=50, color=COLORS[r], alpha=0.85, edgecolor='white', linewidth=0.4)
        ax.axvline(sub.mean(),   color='black',  linestyle='--', linewidth=1.2, label=f'mean={sub.mean():.2f}')
        ax.axvline(sub.median(), color='dimgray', linestyle=':',  linewidth=1.2, label=f'med={sub.median():.2f}')
        ax.set_title(f'{r}  (n={len(sub):,})', fontsize=12)
        ax.set_xlabel('Entropy (bits)')
        ax.set_ylabel('Template count')
        ax.legend(fontsize=9)
    fig.suptitle('Shannon Entropy Distribution by Template Radius', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path}')


def plot_entropy_vs_freq(df, out_path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    for ax, r in zip(axes, RADII):
        sub = df[df['radius'] == r].copy()
        sub['log_count'] = np.log10(sub['total_count'].clip(lower=1))
        # sample for readability if large
        plot_sub = sub.sample(min(len(sub), 20000), random_state=42)
        ax.scatter(plot_sub['log_count'], plot_sub['entropy'],
                   alpha=0.15, s=6, color=COLORS[r], rasterized=True)
        # trend line
        if len(sub) > 10:
            z = np.polyfit(sub['log_count'], sub['entropy'], 1)
            p = np.poly1d(z)
            xs = np.linspace(sub['log_count'].min(), sub['log_count'].max(), 200)
            ax.plot(xs, p(xs), color='black', linewidth=1.5, label=f'slope={z[0]:+.2f}')
            ax.legend(fontsize=9)
        ax.set_title(f'{r}  (n={len(sub):,})', fontsize=12)
        ax.set_xlabel('log₁₀(Total Count)')
        ax.set_ylabel('Entropy (bits)')
    fig.suptitle('Entropy vs. Template Frequency  (log scale)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path}')


def plot_entropy_by_radius(df, out_path):
    stats = df.groupby('radius')['entropy'].agg(
        mean='mean', median='median', q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75)
    ).reindex(RADII)

    x     = np.arange(len(RADII))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(x - width/2, stats['mean'],   width, label='Mean',   color='steelblue',  alpha=0.85)
    ax.bar(       x + width/2, stats['median'], width, label='Median', color='darkorange', alpha=0.85)
    # IQR error bars on mean
    yerr_lo = (stats['mean'] - stats['q25']).clip(lower=0)
    yerr_hi = (stats['q75'] - stats['mean']).clip(lower=0)
    ax.errorbar(x - width/2, stats['mean'],
                yerr=[yerr_lo, yerr_hi],
                fmt='none', color='black', capsize=4, linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(RADII)
    ax.set_xlabel('Template Radius')
    ax.set_ylabel('Entropy (bits)')
    ax.set_title('Mean & Median Entropy by Radius\n(error bars = IQR)', fontsize=13, fontweight='bold')
    ax.legend()
    for bar, val in zip(bars, stats['mean']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path}')


# ── insight extraction ────────────────────────────────────────────────────────

def extract_insights(df, libs, out_path):
    lines = []
    def h(t): lines.extend(['', '=' * 80, f'  {t}', '=' * 80])

    h('TOP-10 CHEMICAL RULES  (entropy=0, total_count > 5, per radius)')
    for r in RADII:
        sub = df[(df['radius'] == r) & (df['entropy'] == 0) & (df['total_count'] > 5)]
        sub = sub.sort_values('total_count', ascending=False).head(10)
        lines.append(f'\n  {r}  ({len(sub)} shown)')
        lines.append(f'  {"Count":>8}  {"Sets":>5}  Template')
        lines.append('  ' + '-' * 70)
        for _, row in sub.iterrows():
            tmpl = row['template']
            # show the single known reagent set
            bucket  = libs[r].get(tmpl, {})
            top_set = max(bucket, key=bucket.get) if bucket else ()
            lines.append(f'  {int(row["total_count"]):>8}  {int(row["unique_reagent_sets"]):>5}  '
                         f'{tmpl[:55]}')
            lines.append(f'  {"":>8}  {"":>5}  → {top_set}')

    h('TOP-10 FLEXIBLE CONDITIONS  (highest entropy, total_count > 20, per radius)')
    for r in RADII:
        sub = df[(df['radius'] == r) & (df['total_count'] > 20)]
        sub = sub.sort_values('entropy', ascending=False).head(10)
        lines.append(f'\n  {r}  ({len(sub)} shown)')
        lines.append(f'  {"Entropy":>8}  {"Count":>8}  {"Sets":>5}  Template')
        lines.append('  ' + '-' * 70)
        for _, row in sub.iterrows():
            lines.append(f'  {row["entropy"]:>8.3f}  {int(row["total_count"]):>8}'
                         f'  {int(row["unique_reagent_sets"]):>5}  {row["template"][:55]}')

    text = '\n'.join(lines)
    print(text)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f'\n  Saved: {out_path}')


# ── summary stats ─────────────────────────────────────────────────────────────

def print_summary(df):
    print('\n' + '═' * 70)
    print('  Shannon Entropy Summary by Radius')
    print('═' * 70)
    print(f'  {"Radius":>6}  {"Templates":>10}  {"Mean H":>8}  {"Med H":>8}'
          f'  {"H=0 (%)":>10}  {"H>2 (%)":>10}')
    print('  ' + '─' * 64)
    for r in RADII:
        sub   = df[df['radius'] == r]['entropy']
        n     = len(sub)
        zero  = (sub == 0).sum()
        hi    = (sub > 2).sum()
        print(f'  {r:>6}  {n:>10,}  {sub.mean():>8.3f}  {sub.median():>8.3f}'
              f'  {100*zero/n:>9.1f}%  {100*hi/n:>9.1f}%')
    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lib_path', default=None,
                    help='Path to the template library .pkl '
                         '(default: Data/flat_template_library.pkl — the flat '
                         'library used by tar/rerank_all.py, so entropy matches reranking)')
    ap.add_argument('--out_dir', default=DATA_DIR,
                    help='Directory to save outputs (default: ../Data/)')
    args = ap.parse_args()

    lib_path = args.lib_path or os.path.join(DATA_DIR, 'flat_template_library.pkl')
    out_dir  = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # ── load library ──────────────────────────────────────────────────────────
    print(f'Loading library: {lib_path}')
    tar_dir = os.path.join(SCRIPT_DIR, '../tar')
    if tar_dir not in sys.path:
        sys.path.insert(0, tar_dir)
    with open(lib_path, 'rb') as f:
        reranker = pickle.load(f)
    libs = reranker._libs
    for r in RADII:
        print(f'  {r}: {len(libs[r]):,} templates')

    # ── compute entropy ───────────────────────────────────────────────────────
    print('\nComputing Shannon entropy ...')
    df = build_records(libs)
    print(f'  Total records: {len(df):,}')

    csv_path = os.path.join(out_dir, 'entropy_analysis.csv')
    df.to_csv(csv_path, index=False)
    print(f'  Saved: {csv_path}')

    print_summary(df)

    # ── plots ─────────────────────────────────────────────────────────────────
    print('Generating plots ...')
    plot_entropy_hist(    df, os.path.join(out_dir, 'entropy_hist.png'))
    plot_entropy_vs_freq( df, os.path.join(out_dir, 'entropy_vs_freq.png'))
    plot_entropy_by_radius(df, os.path.join(out_dir, 'entropy_by_radius.png'))

    # ── insights ──────────────────────────────────────────────────────────────
    print('\nExtracting insights ...')
    extract_insights(df, libs, os.path.join(out_dir, 'template_insights.txt'))

    print('\nDone.')


if __name__ == '__main__':
    main()
