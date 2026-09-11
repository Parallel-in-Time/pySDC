"""Four diagnostic plots for the Dahlquist z-study.

Plots produced:
  1. training_curve.png  – train / val MSE vs epoch
  2. z_acceptance.png    – acceptance rate vs |z| scatter
  3. niter_comparison.png – avg niter learned vs baseline bubble plot (by |z|)
  4. residual_ratio.png   – learned / baseline residual ratio vs |z|

Usage::

    python -m pySDC.playgrounds.learned_qdelta.dahlquist_plot \\
        --checkpoint  <path/to/best.pt> \\
        --benchmark   <path/to/dahlquist_matrix.json> \\
        --output-dir  <path/to/plots>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    _HAVE_MPL = True
except ImportError:
    _HAVE_MPL = False


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load_checkpoint(path: str) -> dict:
    import torch
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location='cpu')
    return ckpt


def _load_benchmark(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# plot 1 – training curve
# ---------------------------------------------------------------------------

def plot_training_curve(history: list[dict], out_path: Path) -> None:
    epochs = [h['epoch'] for h in history]
    train_mse = [h['train_mse'] for h in history]
    val_mse = [h['val_mse'] for h in history]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(epochs, train_mse, label='train MSE', color='steelblue')
    ax.semilogy(epochs, val_mse, label='val MSE', color='tomato', linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE (log scale)')
    ax.set_title('Training curve – one-sweep correction loss')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'  saved {out_path}')


# ---------------------------------------------------------------------------
# plot 2 – acceptance rate vs |z|
# ---------------------------------------------------------------------------

def plot_z_acceptance(rows: list[dict], out_path: Path) -> None:
    z_abs = [abs(r['lam'] * r['dt']) for r in rows]
    acc = [r['learned_acceptance_rate'] for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(z_abs, acc, alpha=0.7, edgecolors='k', linewidths=0.4, s=50)
    ax.set_xscale('log')
    ax.set_xlabel('|z| = |λ·dt|')
    ax.set_ylabel('Acceptance rate')
    ax.set_title('Learned-proposal acceptance rate vs stiffness |z|')
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.5, color='gray', linestyle=':', linewidth=1)
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'  saved {out_path}')


# ---------------------------------------------------------------------------
# plot 3 – avg niter comparison
# ---------------------------------------------------------------------------

def plot_niter_comparison(rows: list[dict], out_path: Path) -> None:
    z_abs = np.array([abs(r['lam'] * r['dt']) for r in rows])
    b_niter = np.array([r['baseline_avg_niter'] for r in rows])
    l_niter = np.array([r['learned_avg_niter'] for r in rows])
    delta = b_niter - l_niter  # positive = learned better

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # scatter
    ax = axes[0]
    sc = ax.scatter(z_abs, l_niter, c=delta, cmap='RdYlGn', vmin=-2, vmax=2,
                    edgecolors='k', linewidths=0.4, s=60)
    ax.plot(z_abs, b_niter, 'k+', markersize=8, label='baseline')
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label('baseline − learned niter')
    ax.set_xscale('log')
    ax.set_xlabel('|z|')
    ax.set_ylabel('avg niter')
    ax.set_title('Avg SDC iterations: dots=learned, + =baseline')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)

    # histogram of delta
    ax2 = axes[1]
    ax2.hist(delta, bins=20, color='steelblue', edgecolor='k', linewidth=0.5)
    ax2.axvline(0, color='red', linewidth=1.5, label='no change')
    ax2.set_xlabel('baseline niter − learned niter')
    ax2.set_ylabel('count')
    ax2.set_title('Iteration savings distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'  saved {out_path}')


# ---------------------------------------------------------------------------
# plot 4 – residual ratio
# ---------------------------------------------------------------------------

def plot_residual_ratio(rows: list[dict], out_path: Path) -> None:
    z_abs = np.array([abs(r['lam'] * r['dt']) for r in rows])
    b_res = np.array([r['baseline_avg_residual_ratio'] for r in rows])
    l_res = np.array([r['learned_avg_residual_ratio'] for r in rows])

    # ratio: values < 1 mean learned converges faster
    ratio = np.where(b_res > 0, l_res / (b_res + 1e-30), np.nan)

    fig, ax = plt.subplots(figsize=(7, 4))
    sc = ax.scatter(z_abs, ratio, c=np.log10(z_abs + 1e-9), cmap='plasma',
                    edgecolors='k', linewidths=0.4, s=60)
    ax.axhline(1.0, color='red', linestyle='--', linewidth=1.5, label='parity')
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label('log₁₀|z|')
    ax.set_xscale('log')
    ax.set_xlabel('|z|')
    ax.set_ylabel('learned / baseline residual ratio')
    ax.set_title('Residual reduction: learned vs baseline (< 1 is better)')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'  saved {out_path}')


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Dahlquist z-study diagnostic plots.')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to best.pt checkpoint from train.py (used for benchmark plots)')
    parser.add_argument('--history-checkpoint', type=str, default=None,
                        help='Optional: separate checkpoint for training history '
                             '(e.g. last.pt). Falls back to --checkpoint if omitted. '
                             'Use this when best.pt was saved early and has incomplete history.')
    parser.add_argument('--benchmark', type=str, required=True,
                        help='Path to dahlquist_matrix.json from dahlquist_benchmark.py')
    parser.add_argument('--output-dir', type=str,
                        default='pySDC/playgrounds/learned_qdelta/results/plots')
    args = parser.parse_args()

    if not _HAVE_MPL:
        raise RuntimeError('matplotlib is required for this script.')

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load data
    ckpt = _load_checkpoint(args.checkpoint)
    bench = _load_benchmark(args.benchmark)

    # For history prefer the last.pt (full run) over best.pt (may be early snapshot)
    hist_ckpt_path = args.history_checkpoint or args.checkpoint
    if hist_ckpt_path != args.checkpoint:
        hist_ckpt = _load_checkpoint(hist_ckpt_path)
    else:
        hist_ckpt = ckpt

    history = hist_ckpt.get('history', [])
    # If best.pt history is shorter than last.pt, fall back automatically
    if hist_ckpt_path == args.checkpoint:
        last_pt = Path(args.checkpoint).parent / 'last.pt'
        if last_pt.exists():
            last_ckpt = _load_checkpoint(str(last_pt))
            last_hist = last_ckpt.get('history', [])
            if len(last_hist) > len(history):
                print(f'  NOTE: using last.pt history ({len(last_hist)} epochs) '
                      f'instead of best.pt ({len(history)} epochs)')
                history = last_hist
    rows = bench.get('rows', [])

    if not history:
        print('WARNING: no training history found – skipping training curve.')
    else:
        print(f'  plotting training curve ({len(history)} epochs)')
        plot_training_curve(history, out_dir / 'training_curve.png')

    if not rows:
        print('WARNING: benchmark file contains no rows – skipping benchmark plots.')
    else:
        plot_z_acceptance(rows, out_dir / 'z_acceptance.png')
        plot_niter_comparison(rows, out_dir / 'niter_comparison.png')
        plot_residual_ratio(rows, out_dir / 'residual_ratio.png')

    print(f'\nAll plots written to {out_dir}')


if __name__ == '__main__':
    main()

