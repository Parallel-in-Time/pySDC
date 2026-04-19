"""Benchmark FNO sweeper vs classical on 1D heat equation.

This benchmark is solver-oriented: it checks parity criteria on iterations,
residual reduction, runtime and acceptance, not just model MSE.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pySDC.helpers.stats_helper import get_sorted
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_unforced
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.playgrounds.learned_qdelta.hooks import LearnedQDeltaHook
from pySDC.playgrounds.learned_qdelta.learned_sweeper import FNOLearnedQDeltaSweeper
from pySDC.playgrounds.learned_qdelta.sweeper_utils import state_to_numpy

try:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    HAVE_MPL = True
except Exception:
    HAVE_MPL = False


def parse_int_list(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(',') if part.strip()]


def _safe_mean(values) -> float:
    return float(np.mean(values)) if values else float('nan')


def _safe_nanmean(values) -> float:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    return float(np.mean(finite)) if finite.size > 0 else float('nan')


def _safe_nanstd(values) -> float:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    return float(np.std(finite)) if finite.size > 0 else float('nan')


def _collect_stats(stats, stat_type: str):
    return [float(v) for _, v in get_sorted(stats, type=stat_type, sortby='time')]


def run_heat1d(sweeper_class, sweeper_params, *, nvars: int, nu: float, dt: float, Tend: float, maxiter: int):
    description = {
        'problem_class': heatNd_unforced,
        'problem_params': {
            'nvars': nvars,
            'nu': nu,
            'freq': 2,
            'bc': 'dirichlet-zero',
        },
        'sweeper_class': sweeper_class,
        'sweeper_params': sweeper_params,
        'level_params': {'dt': dt, 'restol': 1e-10},
        'step_params': {'maxiter': maxiter},
    }
    controller = controller_nonMPI(
        num_procs=1,
        controller_params={'logger_level': 30, 'hook_class': [LearnedQDeltaHook]},
        description=description,
    )

    P = controller.MS[0].levels[0].prob
    u_exact = getattr(P, 'u_exact')
    u0 = u_exact(0.0)

    t0 = time.perf_counter()
    uend, stats = controller.run(u0=u0, t0=0.0, Tend=Tend)
    runtime = time.perf_counter() - t0
    uend_np = state_to_numpy(uend)
    uref_np = state_to_numpy(u_exact(Tend))
    abs_err = float(np.linalg.norm(uend_np - uref_np))
    ref_norm = float(np.linalg.norm(uref_np))

    residuals = _collect_stats(stats, 'residual_post_sweep')
    niters = _collect_stats(stats, 'niter')
    accepts = _collect_stats(stats, 'learned_accept')
    old_res = _collect_stats(stats, 'learned_old_residual')
    tri_res = _collect_stats(stats, 'learned_trial_residual')
    eff_acc = _collect_stats(stats, 'learned_effective_accept_factor')
    conf_ratio = _collect_stats(stats, 'learned_confidence_ratio')
    gate_reason = [int(round(v)) for v in _collect_stats(stats, 'learned_gate_reason_code')]
    infer_times = _collect_stats(stats, 'learned_inference_time')
    trial_eval_times = _collect_stats(stats, 'learned_trial_eval_time')

    gate_hist = {str(code): int(np.sum(np.asarray(gate_reason) == code)) for code in np.unique(gate_reason)}
    pre_reject_frac = float(gate_hist.get('2', 0)) / max(len(gate_reason), 1)

    return {
        'runtime': runtime,
        'avg_niter': _safe_mean(niters),
        'num_steps': len(niters),
        'avg_residual_ratio': float(np.mean(np.array(residuals[1:]) / np.array(residuals[:-1]))) if len(residuals) > 1 else float('nan'),
        'final_abs_error': abs_err,
        'final_rel_error': abs_err / max(ref_norm, 1e-16),
        'acceptance_rate': _safe_mean(accepts),
        'n_proposals': len(accepts),
        'avg_old_residual': _safe_mean(old_res),
        'avg_trial_residual': _safe_mean(tri_res),
        'avg_effective_accept_factor': _safe_mean(eff_acc),
        'avg_confidence_ratio': _safe_mean(conf_ratio),
        'avg_inference_time': _safe_mean(infer_times),
        'avg_trial_eval_time': _safe_mean(trial_eval_times),
        'pre_reject_fraction': pre_reject_frac,
        'gate_reason_hist': gate_hist,
    }


def aggregate_runs(runs: list[dict]) -> dict:
    if not runs:
        return {}
    numeric_keys = [
        'runtime',
        'avg_niter',
        'avg_residual_ratio',
        'final_rel_error',
        'acceptance_rate',
        'avg_effective_accept_factor',
        'avg_confidence_ratio',
        'avg_inference_time',
        'avg_trial_eval_time',
        'pre_reject_fraction',
    ]
    out = {}
    for key in numeric_keys:
        vals = np.asarray([r.get(key, np.nan) for r in runs], dtype=np.float64)
        out[f'{key}_mean'] = _safe_nanmean(vals)
        out[f'{key}_std'] = _safe_nanstd(vals)

    n_proposals_vals = np.asarray([r.get('n_proposals', 0) for r in runs], dtype=np.float64)
    out['n_proposals_mean'] = float(np.mean(n_proposals_vals))
    out['runs'] = runs
    return out


def parity_check(baseline: dict, learned: dict, args) -> dict:
    pass_niter = learned['avg_niter_mean'] <= baseline['avg_niter_mean'] + args.parity_niter_tol
    pass_residual = learned['avg_residual_ratio_mean'] <= baseline['avg_residual_ratio_mean'] + args.parity_residual_tol
    pass_runtime = learned['runtime_mean'] <= args.parity_runtime_factor * baseline['runtime_mean']
    pass_acceptance = learned['acceptance_rate_mean'] >= args.min_acceptance

    niter_ratio = baseline['avg_niter_mean'] / max(learned['avg_niter_mean'], 1e-12)
    residual_ratio = baseline['avg_residual_ratio_mean'] / max(learned['avg_residual_ratio_mean'], 1e-12)
    runtime_ratio = baseline['runtime_mean'] / max(learned['runtime_mean'], 1e-12)
    composite_score = 0.4 * niter_ratio + 0.3 * residual_ratio + 0.2 * runtime_ratio + 0.1 * learned['acceptance_rate_mean']

    return {
        'pass_niter': bool(pass_niter),
        'pass_residual': bool(pass_residual),
        'pass_runtime': bool(pass_runtime),
        'pass_acceptance': bool(pass_acceptance),
        'parity_pass': bool(pass_niter and pass_residual and pass_runtime and pass_acceptance),
        'niter_ratio': float(niter_ratio),
        'residual_ratio': float(residual_ratio),
        'runtime_ratio': float(runtime_ratio),
        'composite_score': float(composite_score),
    }


def _plot_training_curve(checkpoint: str, out_dir: Path) -> None:
    if not HAVE_MPL:
        return
    try:
        import torch

        try:
            ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
        except TypeError:
            ckpt = torch.load(checkpoint, map_location='cpu')
    except Exception:
        return

    history = ckpt.get('history', [])
    if not history:
        return

    epochs = [h['epoch'] for h in history]
    train_mse = [h.get('train_mse', h.get('train_loss', np.nan)) for h in history]
    val_mse = [h.get('val_mse_phys', h.get('val_mse', np.nan)) for h in history]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(epochs, train_mse, label='train_mse')
    ax.semilogy(epochs, val_mse, label='val_mse_phys')
    ax.set_xlabel('epoch')
    ax.set_ylabel('MSE (log)')
    ax.set_title('FNO training curve')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / 'training_curve.png', dpi=150)
    plt.close(fig)


def _plot_benchmark(rows: list[dict], out_dir: Path) -> None:
    if not HAVE_MPL or not rows:
        return
    nvars = np.array([r['nvars'] for r in rows], dtype=np.int64)
    b_niter = np.array([r['baseline']['avg_niter_mean'] for r in rows], dtype=np.float64)
    l_niter = np.array([r['learned']['avg_niter_mean'] for r in rows], dtype=np.float64)
    b_runtime = np.array([r['baseline']['runtime_mean'] for r in rows], dtype=np.float64)
    l_runtime = np.array([r['learned']['runtime_mean'] for r in rows], dtype=np.float64)
    b_res = np.array([r['baseline']['avg_residual_ratio_mean'] for r in rows], dtype=np.float64)
    l_res = np.array([r['learned']['avg_residual_ratio_mean'] for r in rows], dtype=np.float64)
    acc = np.array([r['learned']['acceptance_rate_mean'] for r in rows], dtype=np.float64)
    pre = np.array([r['learned']['pre_reject_fraction_mean'] for r in rows], dtype=np.float64)
    infer = np.array([r['learned']['avg_inference_time_mean'] for r in rows], dtype=np.float64)
    trial_eval = np.array([r['learned']['avg_trial_eval_time_mean'] for r in rows], dtype=np.float64)
    score = np.array([r['parity']['composite_score'] for r in rows], dtype=np.float64)
    parity_ok = np.array([1.0 if r['parity']['parity_pass'] else 0.0 for r in rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(nvars, b_niter, 'o-', label='baseline')
    ax.plot(nvars, l_niter, 's--', label='learned')
    ax.set_xlabel('nvars')
    ax.set_ylabel('avg_niter')
    ax.set_title('Iteration parity by resolution')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / 'niter_vs_nvars.png', dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(nvars, b_runtime, 'o-', label='baseline')
    ax.plot(nvars, l_runtime, 's--', label='learned')
    ax.set_xlabel('nvars')
    ax.set_ylabel('runtime (s)')
    ax.set_title('Runtime parity by resolution')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / 'runtime_vs_nvars.png', dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(nvars, l_res / np.maximum(b_res, 1e-16), 'o-', label='learned / baseline')
    ax.axhline(1.0, color='k', linestyle='--', linewidth=1)
    ax.set_xlabel('nvars')
    ax.set_ylabel('residual ratio ratio')
    ax.set_title('Residual reduction parity (<1 better)')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / 'residual_ratio_parity.png', dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.35
    x = np.arange(len(nvars))
    ax.bar(x - width / 2, acc, width, label='acceptance')
    ax.bar(x + width / 2, pre, width, label='pre-reject')
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in nvars])
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel('nvars')
    ax.set_ylabel('fraction')
    ax.set_title('Gate behavior by resolution')
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / 'gate_behavior.png', dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ['tab:green' if ok > 0.5 else 'tab:red' for ok in parity_ok]
    ax.bar([str(v) for v in nvars], score, color=colors)
    ax.set_xlabel('nvars')
    ax.set_ylabel('composite parity score')
    ax.set_title('Composite score (green=parity pass)')
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / 'composite_score.png', dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(nvars, infer, 'o-', label='model inference')
    ax.plot(nvars, trial_eval, 's--', label='trial residual eval')
    ax.set_xlabel('nvars')
    ax.set_ylabel('time per proposal (s)')
    ax.set_title('Learned-sweeper overhead by resolution')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / 'inference_overhead.png', dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Benchmark FNO sweeper on 1D heat equation.')
    parser.add_argument('--checkpoint', type=str,
                        default='pySDC/playgrounds/learned_qdelta/checkpoints/heat1d_fno_v1/best.pt')
    parser.add_argument('--nvars', type=str, default='127,255', help='Comma-separated grid sizes to test')
    parser.add_argument('--nu', type=float, default=0.1)
    parser.add_argument('--dt', type=float, default=0.01)
    parser.add_argument('--Tend', type=float, default=0.05)
    parser.add_argument('--maxiter', type=int, default=4)
    parser.add_argument('--accept-factor', type=float, default=0.95)
    parser.add_argument('--accept-factor-min', type=float, default=0.92)
    parser.add_argument('--accept-factor-max', type=float, default=1.01)
    parser.add_argument('--accept-factor-slope', type=float, default=0.02)
    parser.add_argument('--accept-factor-center', type=float, default=-6.0)
    parser.add_argument('--confidence-ratio-max', type=float, default=4.0)
    parser.add_argument('--learned-max-sweeps-per-step', type=int, default=0,
                        help='Limit learned proposals to first K sweeps per step (0 = no limit)')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--parity-niter-tol', type=float, default=0.0)
    parser.add_argument('--parity-residual-tol', type=float, default=0.0)
    parser.add_argument('--parity-runtime-factor', type=float, default=1.10)
    parser.add_argument('--min-acceptance', type=float, default=0.05)
    parser.add_argument('--output', type=str,
                        default='pySDC/playgrounds/learned_qdelta/results/heat1d_fno_benchmark.json')
    parser.add_argument('--plot-dir', type=str, default='',
                        help='Optional directory for benchmark plots (default: <output>/plots)')
    args = parser.parse_args()

    classical_params = {'quad_type': 'RADAU-RIGHT', 'num_nodes': 3, 'QI': 'IE'}
    fno_params = {
        'quad_type': 'RADAU-RIGHT',
        'num_nodes': 3,
        'QI': 'IE',
        'model_checkpoint': args.checkpoint,
        'accept_factor': args.accept_factor,
        'accept_factor_min': args.accept_factor_min,
        'accept_factor_max': args.accept_factor_max,
        'accept_factor_slope': args.accept_factor_slope,
        'accept_factor_center': args.accept_factor_center,
        'confidence_ratio_max': args.confidence_ratio_max,
        'learned_max_sweeps_per_step': args.learned_max_sweeps_per_step,
        'model_device': 'cpu',
        'fallback_sweeper_class': generic_implicit,
    }

    rows = []
    print(
        f"{'nvars':>6}  {'base_niter':>10}  {'fno_niter':>10}  {'acc':>7}"
        f"  {'base_rt':>9}  {'fno_rt':>9}  {'score':>7}  {'parity':>7}"
    )
    print('-' * 88)

    for nvars in parse_int_list(args.nvars):
        baseline_runs = [
            run_heat1d(generic_implicit, classical_params, nvars=nvars, nu=args.nu, dt=args.dt, Tend=args.Tend, maxiter=args.maxiter)
            for _ in range(args.repeats)
        ]
        learned_runs = [
            run_heat1d(FNOLearnedQDeltaSweeper, fno_params, nvars=nvars, nu=args.nu, dt=args.dt, Tend=args.Tend, maxiter=args.maxiter)
            for _ in range(args.repeats)
        ]

        baseline = aggregate_runs(baseline_runs)
        learned = aggregate_runs(learned_runs)
        parity = parity_check(baseline, learned, args)
        row = {
            'nvars': nvars,
            'baseline': baseline,
            'learned': learned,
            'parity': parity,
        }
        rows.append(row)

        print(
            f"{nvars:6d}  {baseline['avg_niter_mean']:10.2f}  {learned['avg_niter_mean']:10.2f}"
            f"  {learned['acceptance_rate_mean']:7.3f}  {baseline['runtime_mean']:9.4f}"
            f"  {learned['runtime_mean']:9.4f}  {parity['composite_score']:7.3f}"
            f"  {str(parity['parity_pass']):>7}"
        )

    parity_pass_count = int(sum(1 for row in rows if row['parity']['parity_pass']))
    out = {
        'config': vars(args),
        'rows': rows,
        'parity_summary': {
            'num_rows': len(rows),
            'num_pass': parity_pass_count,
            'pass_rate': float(parity_pass_count / max(len(rows), 1)),
            'all_pass': bool(parity_pass_count == len(rows)),
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as fh:
        json.dump(out, fh, indent=2)
    print(f"\nSaved report to {out_path}")

    plot_dir = Path(args.plot_dir) if args.plot_dir else (out_path.parent / 'plots')
    plot_dir.mkdir(parents=True, exist_ok=True)
    if HAVE_MPL:
        _plot_training_curve(args.checkpoint, plot_dir)
        _plot_benchmark(rows, plot_dir)
        print(f"Saved plots to {plot_dir}")
    else:
        print('matplotlib not available; skipping plots')


if __name__ == '__main__':
    main()

