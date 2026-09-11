"""End-to-end Dahlquist FNO pipeline: generate data, train FNO, benchmark acceptance.

Two modes are supported via --use-z-param:

  Standard (--no-use-z-param, default):
    lambda and dt vary independently; benchmark sweeps (lambda, dt) pairs.

  Z-param (--use-z-param, recommended):
    z = lambda * dt is sampled log-uniformly (dt=1 fixed).
    Full stiffness range is covered uniformly in log space.
    Benchmark sweeps over z values with dt=1.

Dahlquist is a scalar ODE (state_dim=1), so the FNO is applied to
(B, 1+2M, 1) spatial fields. It degrades to pointwise operations —
equivalent to a small MLP — but validates the FNO inference path.

Usage (z-param, recommended)
-----
::

    python -m pySDC.playgrounds.learned_qdelta.dahlquist_fno_pipeline \\
        --use-z-param --num-cases 1000 --epochs 150 --width 64 \\
        --output-root results/dahlquist_fno_z
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace  # noqa: F401 (used for train_fno args)

import numpy as np
import torch

from pySDC.helpers.stats_helper import get_sorted
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.TestEquation_0D import testequation0d
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.playgrounds.learned_qdelta.data_generation import generate_dataset
from pySDC.playgrounds.learned_qdelta.dahlquist_benchmark import aggregate_by_regime
from pySDC.playgrounds.learned_qdelta.hooks import LearnedQDeltaHook
from pySDC.playgrounds.learned_qdelta.learned_sweeper import FNOLearnedQDeltaSweeper
from pySDC.playgrounds.learned_qdelta.train_fno import train as train_fno


# ---------------------------------------------------------------------------
def run_dahlquist(sweeper_class, sweeper_params, lam, dt, Tend, maxiter):
    """Run a single Dahlquist case and return stats dict (real inference)."""
    description = {
        'problem_class': testequation0d,
        'problem_params': {'lambdas': np.array([lam]), 'u0': 1.0},
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
    _, stats = ctrl.run(u0=u0, t0=0.0, Tend=Tend)
    runtime = time.perf_counter() - t0

    niters  = [v for _, v in get_sorted(stats, type='niter',         sortby='time')]
    accepts = [v for _, v in get_sorted(stats, type='learned_accept', sortby='time')]
    old_res = [v for _, v in get_sorted(stats, type='learned_old_residual',   sortby='time')]
    tri_res = [v for _, v in get_sorted(stats, type='learned_trial_residual', sortby='time')]

    return {
        'avg_niter':        float(np.mean(niters))   if niters   else float('nan'),
        'acceptance_rate':  float(np.mean(accepts))  if accepts  else float('nan'),
        'n_proposals':      len(accepts),
        'avg_old_residual': float(np.mean(old_res))  if old_res  else float('nan'),
        'avg_tri_residual': float(np.mean(tri_res))  if tri_res  else float('nan'),
        'runtime':          runtime,
    }


# ---------------------------------------------------------------------------
def run_benchmark(ckpt_path: str, args):
    """Run the full benchmark grid; returns (rows, regime_summary)."""
    regime_edges = [1.0, 5.0, 15.0]

    if args.use_z_param:
        # z-sweep: dt=1, lambda = z
        zvals = [float(x) for x in args.zvals.split(',')]
        lam_dt_pairs = [(z, 1.0) for z in zvals]
    else:
        lambdas = [float(x) for x in args.lambdas.split(',')]
        dts     = [float(x) for x in args.dts.split(',')]
        lam_dt_pairs = [(lam, dt) for lam in lambdas for dt in dts]

    classical_params = {'quad_type': 'RADAU-RIGHT', 'num_nodes': 3, 'QI': 'IE'}
    fno_params = {
        'quad_type':            'RADAU-RIGHT',
        'num_nodes':            3,
        'QI':                   'IE',
        'model_checkpoint':     ckpt_path,
        'accept_factor':        args.accept_factor,
        'model_device':         'cpu',
        'fallback_sweeper_class': generic_implicit,
    }

    rows = []
    print(f"\n  {'lam':>8}  {'dt':>5}  {'|z|':>6}  {'base_niter':>10}"
          f"  {'fno_niter':>9}  {'acc':>6}  {'proposals':>9}")

    for lam, dt in lam_dt_pairs:
        lamdt = abs(lam * dt)
        baseline = run_dahlquist(generic_implicit, classical_params,
                                 lam, dt, args.Tend, args.maxiter)
        fno      = run_dahlquist(FNOLearnedQDeltaSweeper, fno_params,
                                 lam, dt, args.Tend, args.maxiter)

        row = {
            'lam':                       lam,
            'dt':                        dt,
            'lamdt_abs':                 lamdt,
            'baseline_avg_niter':        baseline['avg_niter'],
            'learned_avg_niter':         fno['avg_niter'],
            'baseline_runtime':          baseline['runtime'],
            'learned_runtime':           fno['runtime'],
            'baseline_avg_residual_ratio': float('nan'),
            'learned_avg_residual_ratio':  float('nan'),
            'learned_acceptance_rate':   fno['acceptance_rate'],
            'n_proposals':               fno['n_proposals'],
            'avg_old_residual':          fno['avg_old_residual'],
            'avg_trial_residual':        fno['avg_tri_residual'],
        }
        rows.append(row)
        print(f"  {lam:>8.2f}  {dt:>5.3f}  {lamdt:>6.3f}"
              f"  {baseline['avg_niter']:>10.2f}  {fno['avg_niter']:>9.2f}"
              f"  {fno['acceptance_rate']:>6.3f}  {fno['n_proposals']:>9d}")

    regime_summary = aggregate_by_regime(rows, regime_edges)
    print('\n=== Regime summary (|z| = |lambda*dt| bands) ===')
    print(f"  {'regime':>8}  {'count':>6}  {'acc':>8}  {'fno_niter':>10}  {'base_niter':>10}")
    for label in sorted(regime_summary.keys(), key=int):
        info = regime_summary[label]
        print(f"  {label:>8}  {info['count']:>6}"
              f"  {info['learned_acceptance_rate']:>8.3f}"
              f"  {info['learned_avg_niter']:>10.2f}"
              f"  {info['baseline_avg_niter']:>10.2f}")

    all_acc = [r['learned_acceptance_rate'] for r in rows
               if not np.isnan(r['learned_acceptance_rate'])]
    overall  = float(np.mean(all_acc)) if all_acc else float('nan')
    print(f'\nOverall FNO acceptance rate: {overall:.3f}  (n={len(all_acc)} cases)')

    return rows, regime_summary


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Dahlquist FNO pipeline (z-param aware)')

    # ---- I/O ----
    parser.add_argument('--output-root', type=str,
                        default='pySDC/playgrounds/learned_qdelta/results/dahlquist_fno_z')

    # ---- Data generation ----
    parser.add_argument('--use-z-param', action='store_true', default=True,
                        help='Sample z=lambda*dt log-uniformly (dt=1). Recommended.')
    parser.add_argument('--no-use-z-param', dest='use_z_param', action='store_false')
    parser.add_argument('--num-cases',    type=int,   default=1000)
    parser.add_argument('--nsteps',       type=int,   default=8)
    parser.add_argument('--maxiter-data', type=int,   default=4)
    parser.add_argument('--z-min',        type=float, default=0.01)
    parser.add_argument('--z-max',        type=float, default=100.0)
    parser.add_argument('--lambda-min',   type=float, default=-25.0,
                        help='Used when --no-use-z-param')
    parser.add_argument('--lambda-max',   type=float, default=-0.5,
                        help='Used when --no-use-z-param')
    parser.add_argument('--dt-min',       type=float, default=0.02,
                        help='Used when --no-use-z-param')
    parser.add_argument('--dt-max',       type=float, default=0.2,
                        help='Used when --no-use-z-param')

    # ---- FNO training ----
    parser.add_argument('--epochs',     type=int,   default=150)
    parser.add_argument('--width',      type=int,   default=64)
    parser.add_argument('--modes',      type=int,   default=1,
                        help='Fourier modes; max=1 for scalar Dahlquist')
    parser.add_argument('--depth',      type=int,   default=4)
    parser.add_argument('--batch-size', type=int,   default=128)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--seed',       type=int,   default=7)

    # ---- Benchmark ----
    # z-param mode: sweep z values with dt=1
    parser.add_argument('--zvals', type=str,
                        default='-0.1,-0.5,-1,-2,-3,-5,-8,-10,-15,-20,-30,-50',
                        help='z = lambda*dt values for z-sweep (use -- before negative values)')
    # standard mode
    parser.add_argument('--lambdas', type=str, default='-2,-5,-10,-20')
    parser.add_argument('--dts',     type=str, default='0.05,0.1,0.2')

    parser.add_argument('--Tend',          type=float, default=1.0)
    parser.add_argument('--maxiter',       type=int,   default=6)
    parser.add_argument('--accept-factor', type=float, default=0.95)

    args = parser.parse_args()

    root     = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    data_path = root / 'dahlquist_sweeps.npz'
    ckpt_dir  = root / 'checkpoints'

    # ---- 1. Data generation ------------------------------------------------
    print('=' * 60)
    mode_label = 'z-param (dt=1, log-uniform |z|)' if args.use_z_param else 'standard (lambda,dt)'
    print(f'Step 1: generating Dahlquist training data [{mode_label}] …')

    if args.use_z_param:
        gen_kwargs = dict(problem='dahlquist_z', z_min=args.z_min, z_max=args.z_max)
    else:
        gen_kwargs = dict(
            problem='dahlquist',
            lambda_min=args.lambda_min, lambda_max=args.lambda_max,
            dt_min=args.dt_min,         dt_max=args.dt_max,
        )

    generate_dataset(
        output_path=str(data_path),
        num_cases=args.num_cases,
        nsteps=args.nsteps,
        maxiter=args.maxiter_data,
        num_nodes=3,
        seed=args.seed,
        **gen_kwargs,
    )

    # ---- 2. FNO training ---------------------------------------------------
    print('=' * 60)
    print('Step 2: training FNO …')
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
    ))
    best_ckpt = str(ckpt_dir / 'best.pt')
    print(f'Best checkpoint: {best_ckpt}')

    # ---- 3. Benchmark (real FNO inference) ---------------------------------
    print('=' * 60)
    print('Step 3: benchmarking FNO vs classical sweeper (live inference) …')
    rows, regime_summary = run_benchmark(best_ckpt, args)

    # ---- 4. Save JSON report -----------------------------------------------
    report_path = root / 'benchmark_report.json'
    with open(report_path, 'w') as fh:
        json.dump({
            'config': vars(args),
            'rows':   rows,
            'regime_summary': regime_summary,
        }, fh, indent=2)
    print(f'\nFull report saved → {report_path}')


if __name__ == '__main__':
    main()


