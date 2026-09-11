"""End-to-end heat1d FNO pipeline: generate data, train FNO, benchmark.

This is the spatial-PDE counterpart of ``dahlquist_fno_pipeline.py``.

The pipeline has three steps:

  1. **Data generation** – run pySDC on ``heatNd_unforced`` (1D heat equation,
     fixed ``nvars=127``) and collect one-sweep training samples.

  2. **FNO training** – train a Fourier Neural Operator (FNO1d) on the
     spatial-field samples.  The FNO is resolution-agnostic: after training on
     ``nvars=127`` grids it can be benchmarked on ``nvars=255``.

  3. **Benchmark** – run ``FNOLearnedQDeltaSweeper`` (live inference) against
     the classical ``generic_implicit`` sweeper on one or several grid sizes.

Usage
-----
::

    python -m pySDC.playgrounds.learned_qdelta.heat1d_fno_pipeline \\
        --num-cases 200 --epochs 80 --nvars 127,255 \\
        --output-root pySDC/playgrounds/learned_qdelta/results/heat1d_fno

Minimal smoke-test (fast, ~1 min)::

    python -m pySDC.playgrounds.learned_qdelta.heat1d_fno_pipeline \\
        --num-cases 20 --epochs 5 --nvars 127 \\
        --output-root /tmp/heat1d_fno_smoke
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from pySDC.helpers.stats_helper import get_sorted
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_unforced
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.playgrounds.learned_qdelta.hooks import LearnedQDeltaHook
from pySDC.playgrounds.learned_qdelta.learned_sweeper import FNOLearnedQDeltaSweeper
from pySDC.playgrounds.learned_qdelta.sweeper_utils import state_to_numpy
from pySDC.playgrounds.learned_qdelta.train_fno import train as train_fno


def _parse_float_list(raw: str) -> list[float]:
    return [float(part.strip()) for part in str(raw).split(',') if part.strip()]


# ---------------------------------------------------------------------------
# Benchmark helper
# ---------------------------------------------------------------------------

def run_heat1d(sweeper_class, sweeper_params, *, nvars: int, nu: float,
               dt: float, Tend: float, maxiter: int) -> dict:
    """Run 1D heat equation and return a metrics dict."""
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
    ctrl = controller_nonMPI(
        num_procs=1,
        controller_params={'logger_level': 30, 'hook_class': [LearnedQDeltaHook]},
        description=description,
    )
    P = ctrl.MS[0].levels[0].prob
    u0 = P.u_exact(0.0)

    t0 = time.perf_counter()
    uend, stats = ctrl.run(u0=u0, t0=0.0, Tend=Tend)
    runtime = time.perf_counter() - t0

    uend_np = state_to_numpy(uend)
    uref_np = state_to_numpy(P.u_exact(Tend))
    abs_err = float(np.linalg.norm(uend_np - uref_np))
    ref_norm = float(np.linalg.norm(uref_np))

    residuals = [v for _, v in get_sorted(stats, type='residual_post_sweep', sortby='time')]
    niters    = [v for _, v in get_sorted(stats, type='niter',               sortby='time')]
    accepts   = [v for _, v in get_sorted(stats, type='learned_accept',      sortby='time')]

    avg_res_ratio = (float(np.mean(np.array(residuals[1:]) / np.array(residuals[:-1])))
                     if len(residuals) > 1 else float('nan'))

    return {
        'runtime':           runtime,
        'avg_niter':         float(np.mean(niters)) if niters else float('nan'),
        'num_steps':         len(niters),
        'avg_residual_ratio': avg_res_ratio,
        'final_abs_error':   abs_err,
        'final_rel_error':   abs_err / max(ref_norm, 1e-16),
        'acceptance_rate':   float(np.mean(accepts)) if accepts else float('nan'),
    }


def run_benchmark(ckpt_path: str, args) -> list[dict]:
    """Benchmark FNO sweeper vs classical on one or several grid sizes."""
    nvars_list = [int(x.strip()) for x in str(args.nvars).split(',') if x.strip()]
    nu         = args.nu
    dt         = args.dt
    Tend       = args.Tend
    maxiter    = args.maxiter

    classical_params = {'quad_type': 'RADAU-RIGHT', 'num_nodes': 3, 'QI': 'IE'}
    fno_params = {
        'quad_type':             'RADAU-RIGHT',
        'num_nodes':             3,
        'QI':                    'IE',
        'model_checkpoint':      ckpt_path,
        'accept_factor':         args.accept_factor,
        'model_device':          'cpu',
        'fallback_sweeper_class': generic_implicit,
    }

    print(f"\n  {'nvars':>6}  {'base_niter':>10}  {'fno_niter':>9}  {'acc':>6}"
          f"  {'base_err':>10}  {'fno_err':>10}  {'base_rt':>8}  {'fno_rt':>8}")
    print('  ' + '-' * 76)

    rows = []
    for nvars in nvars_list:
        baseline = run_heat1d(generic_implicit, classical_params,
                              nvars=nvars, nu=nu, dt=dt, Tend=Tend, maxiter=maxiter)
        learned  = run_heat1d(FNOLearnedQDeltaSweeper, fno_params,
                              nvars=nvars, nu=nu, dt=dt, Tend=Tend, maxiter=maxiter)
        row = {
            'nvars':    nvars,
            'baseline': baseline,
            'learned':  learned,
            'comparison': {
                'niter_delta':         learned['avg_niter'] - baseline['avg_niter'],
                'runtime_delta':       learned['runtime']   - baseline['runtime'],
                'final_rel_err_delta': learned['final_rel_error'] - baseline['final_rel_error'],
            },
        }
        rows.append(row)
        print(
            f"  {nvars:6d}  {baseline['avg_niter']:10.2f}  {learned['avg_niter']:9.2f}"
            f"  {learned['acceptance_rate']:6.3f}"
            f"  {baseline['final_rel_error']:10.3e}  {learned['final_rel_error']:10.3e}"
            f"  {baseline['runtime']:8.3f}  {learned['runtime']:8.3f}"
        )
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Heat1d FNO pipeline: generate → train → benchmark'
    )

    # I/O
    parser.add_argument('--output-root', type=str,
                        default='pySDC/playgrounds/learned_qdelta/results/heat1d_fno')

    # Data generation
    parser.add_argument('--num-cases',    type=int,   default=200,
                        help='Number of randomised (nu, dt) parameter cases')
    parser.add_argument('--nsteps',       type=int,   default=4,
                        help='Time steps per case during data generation')
    parser.add_argument('--maxiter-data', type=int,   default=4,
                        help='Max SDC iterations per step during data generation')
    parser.add_argument('--train-nvars',  type=int,   default=127,
                        help='Grid size used for data generation / training')
    parser.add_argument('--seed',         type=int,   default=7)
    parser.add_argument('--data-cfl-edges', type=str, default='0.5,2.0,6.0',
                        help='Comma-separated diffusion-CFL regime edges for balanced data generation')
    parser.add_argument('--boundary-frac', type=float, default=0.2,
                        help='Fraction of data cases sampled near CFL regime boundaries')

    # FNO architecture & training
    parser.add_argument('--epochs',     type=int,   default=80)
    parser.add_argument('--width',      type=int,   default=64)
    parser.add_argument('--modes',      type=int,   default=16,
                        help='Number of Fourier modes kept in spectral layers')
    parser.add_argument('--depth',      type=int,   default=4)
    parser.add_argument('--batch-size', type=int,   default=32)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--balance-cfl', action='store_true')
    parser.add_argument('--cfl-edges', type=str, default='0.2,1.0,4.0')
    parser.add_argument('--residual-proj-weight', type=float, default=0.2)
    parser.add_argument('--cosine-gate-weight', type=float, default=0.05)
    parser.add_argument('--cosine-gate-margin', type=float, default=0.2)
    parser.add_argument('--checkpoint-metric', type=str, choices=['val_mse', 'composite'], default='composite')

    # Benchmark
    parser.add_argument('--nvars',         type=str,   default='127,255',
                        help='Comma-separated grid sizes to benchmark (may differ from training)')
    parser.add_argument('--nu',            type=float, default=0.1)
    parser.add_argument('--dt',            type=float, default=0.01)
    parser.add_argument('--Tend',          type=float, default=0.05)
    parser.add_argument('--maxiter',       type=int,   default=4)
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

    args = parser.parse_args()

    root      = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    data_path = root / 'heat1d_sweeps.npz'
    ckpt_dir  = root / 'checkpoints'

    # -------------------------------------------------------------------------
    # Step 1 — Data generation
    # -------------------------------------------------------------------------
    print('=' * 65)
    print(f'Step 1: generating heat1d training data  (nvars={args.train_nvars}) …')
    print('=' * 65)

    # Override the default nvars in generate_dataset by monkey-patching the
    # problem preset via the existing 'heat1d' preset (nvars=127 by default).
    # For non-default training grid sizes we temporarily patch the function.
    import pySDC.playgrounds.learned_qdelta.data_generation as _dg
    _orig_fn = _dg.generate_dataset

    train_nvars = args.train_nvars

    def _patched_generate_dataset(output_path, **kwargs):
        """Wrapper that injects --train-nvars and balanced CFL sampling."""
        import numpy as _np
        from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_unforced as _heat
        from pySDC.playgrounds.learned_qdelta.data_generation import _stack_samples, run_case
        from pySDC.playgrounds.learned_qdelta.dataset import DATA_CONTRACT_VERSION
        from pathlib import Path as _Path

        num_cases  = kwargs.get('num_cases', 200)
        nsteps     = kwargs.get('nsteps', 4)
        maxiter    = kwargs.get('maxiter', 4)
        num_nodes  = kwargs.get('num_nodes', 3)
        seed       = kwargs.get('seed', 7)
        rng        = _np.random.default_rng(seed)
        all_samples = []
        cfl_edges = _parse_float_list(args.data_cfl_edges)
        cfl_bins = [0.05] + cfl_edges + [15.0]
        boundary_frac = float(max(0.0, min(1.0, args.boundary_frac)))
        n_boundary = int(round(boundary_frac * num_cases))
        dx = 1.0 / float(train_nvars + 1)

        def _sample_dt_nu(regime_idx: int, near_boundary: bool = False):
            dt = float(rng.uniform(0.002, 0.02))
            if near_boundary and len(cfl_edges) > 0:
                edge = float(cfl_edges[regime_idx % len(cfl_edges)])
                cfl = float(edge * rng.uniform(0.9, 1.1))
            else:
                low = float(cfl_bins[regime_idx])
                high = float(cfl_bins[regime_idx + 1])
                cfl = float(rng.uniform(low, high))

            nu = cfl * (dx * dx) / max(dt, 1e-12)
            nu = float(np.clip(nu, 0.02, 0.2))
            return dt, nu

        total_regimes = max(1, len(cfl_bins) - 1)
        for i in range(num_cases):
            regime_idx = i % total_regimes
            near_boundary = i < n_boundary
            dt, nu = _sample_dt_nu(regime_idx=regime_idx, near_boundary=near_boundary)
            problem_params = {
                'nvars': train_nvars,
                'nu':    nu,
                'freq':  2,
                'bc':    'dirichlet-zero',
            }
            all_samples.extend(
                run_case(
                    problem_class=_heat,
                    problem_params=problem_params,
                    dt=dt,
                    maxiter=maxiter,
                    num_nodes=num_nodes,
                    nsteps=nsteps,
                )
            )

        if not all_samples:
            raise RuntimeError('No samples generated.')

        data = _stack_samples(all_samples)
        data['contract_version'] = _np.array([DATA_CONTRACT_VERSION], dtype=_np.int32)
        data['problem_name'] = _np.array(['heat1d'])
        data['seed'] = _np.array([seed], dtype=_np.int64)
        data['use_z_param'] = _np.array([0], dtype=_np.int32)
        out = _Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        _np.savez_compressed(out, **data)
        print(f'Saved {data["u0"].shape[0]} samples to {out}')

    _patched_generate_dataset(
        str(data_path),
        num_cases=args.num_cases,
        nsteps=args.nsteps,
        maxiter=args.maxiter_data,
        num_nodes=3,
        seed=args.seed,
    )

    # -------------------------------------------------------------------------
    # Step 2 — FNO training
    # -------------------------------------------------------------------------
    print()
    print('=' * 65)
    print('Step 2: training FNO …')
    print('=' * 65)

    torch.manual_seed(args.seed)
    train_fno(SimpleNamespace(
        data=str(data_path),
        output_dir=str(ckpt_dir),
        width=args.width,
        modes=args.modes,
        depth=args.depth,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_fraction=0.2,
        seed=args.seed,
        device='cpu',
        balance_cfl=bool(args.balance_cfl),
        cfl_edges=args.cfl_edges,
        residual_proj_weight=args.residual_proj_weight,
        cosine_gate_weight=args.cosine_gate_weight,
        cosine_gate_margin=args.cosine_gate_margin,
        checkpoint_metric=args.checkpoint_metric,
    ))
    best_ckpt = str(ckpt_dir / 'best.pt')
    print(f'Best checkpoint: {best_ckpt}')

    # -------------------------------------------------------------------------
    # Step 3 — Benchmark
    # -------------------------------------------------------------------------
    print()
    print('=' * 65)
    print('Step 3: benchmarking FNO vs classical sweeper …')
    print('=' * 65)

    benchmark_path = root / 'benchmark_report.json'
    plot_dir = root / 'plots'
    cmd = [
        sys.executable,
        '-m',
        'pySDC.playgrounds.learned_qdelta.heat1d_fno_benchmark',
        '--checkpoint', best_ckpt,
        '--nvars', str(args.nvars),
        '--nu', str(args.nu),
        '--dt', str(args.dt),
        '--Tend', str(args.Tend),
        '--maxiter', str(args.maxiter),
        '--accept-factor', str(args.accept_factor),
        '--accept-factor-min', str(args.accept_factor_min),
        '--accept-factor-max', str(args.accept_factor_max),
        '--accept-factor-slope', str(args.accept_factor_slope),
        '--accept-factor-center', str(args.accept_factor_center),
        '--confidence-ratio-max', str(args.confidence_ratio_max),
        '--learned-max-sweeps-per-step', str(args.learned_max_sweeps_per_step),
        '--repeats', str(args.repeats),
        '--parity-niter-tol', str(args.parity_niter_tol),
        '--parity-residual-tol', str(args.parity_residual_tol),
        '--parity-runtime-factor', str(args.parity_runtime_factor),
        '--min-acceptance', str(args.min_acceptance),
        '--output', str(benchmark_path),
        '--plot-dir', str(plot_dir),
    ]
    subprocess.run(cmd, check=True)

    # -------------------------------------------------------------------------
    # Step 4 — Save report
    # -------------------------------------------------------------------------
    with open(benchmark_path) as fh:
        report = json.load(fh)
    parity_summary = report.get('parity_summary', {})
    print(f"\nBenchmark report saved -> {benchmark_path}")
    print(
        'Parity pass rate: '
        f"{parity_summary.get('num_pass', 0)}/{parity_summary.get('num_rows', 0)} "
        f"({parity_summary.get('pass_rate', 0.0):.2%})"
    )
    print(f'Plots saved -> {plot_dir}')


if __name__ == '__main__':
    main()


