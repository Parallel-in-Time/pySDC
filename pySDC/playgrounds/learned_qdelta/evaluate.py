"""Compare classical sweeper vs learned sweeper with fallback."""

from __future__ import annotations

import argparse
import time

import numpy as np

from pySDC.helpers.stats_helper import get_sorted
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_unforced
from pySDC.implementations.problem_classes.TestEquation_0D import testequation0d
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.playgrounds.learned_qdelta.hooks import LearnedQDeltaHook
from pySDC.playgrounds.learned_qdelta.learned_sweeper import LearnedQDeltaSweeper


def build_problem(problem: str, lam: float, nu: float):
    if problem == 'dahlquist':
        return testequation0d, {'lambdas': np.array([lam]), 'u0': 1.0}
    if problem == 'heat1d':
        return heatNd_unforced, {'nvars': 127, 'nu': nu, 'freq': 2, 'bc': 'dirichlet-zero'}
    raise ValueError(f'Unknown problem preset: {problem}')


def run_setup(sweeper_class, sweeper_params, dt, Tend, maxiter, problem: str, lam: float, nu: float):
    level_params = {'dt': dt, 'restol': 1e-10}
    step_params = {'maxiter': maxiter}
    problem_class, problem_params = build_problem(problem, lam=lam, nu=nu)

    description = {
        'problem_class': problem_class,
        'problem_params': problem_params,
        'sweeper_class': sweeper_class,
        'sweeper_params': sweeper_params,
        'level_params': level_params,
        'step_params': step_params,
    }

    controller_params = {'logger_level': 30, 'hook_class': [LearnedQDeltaHook]}
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    P = controller.MS[0].levels[0].prob
    u0 = getattr(P, 'u_exact')(0.0)

    t0 = time.perf_counter()
    _, stats = controller.run(u0=u0, t0=0.0, Tend=Tend)
    runtime = time.perf_counter() - t0

    residuals = [v for _, v in get_sorted(stats, type='residual_post_sweep', sortby='time')]
    niters = [v for _, v in get_sorted(stats, type='niter', sortby='time')]

    accepts = [v for _, v in get_sorted(stats, type='learned_accept', sortby='time')]

    result = {
        'runtime': runtime,
        'avg_residual_ratio': float(np.mean(np.array(residuals[1:]) / np.array(residuals[:-1]))) if len(residuals) > 1 else np.nan,
        'avg_niter': float(np.mean(niters)) if niters else np.nan,
        'acceptance_rate': float(np.mean(accepts)) if accepts else np.nan,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description='Evaluate learned sweeper against classical baseline.')
    parser.add_argument('--checkpoint', type=str, default='pySDC/playgrounds/learned_qdelta/checkpoints/best.pt')
    parser.add_argument('--problem', type=str, choices=['dahlquist', 'heat1d'], default='dahlquist')
    parser.add_argument('--dt', type=float, default=0.1)
    parser.add_argument('--lam', type=float, default=-8.0)
    parser.add_argument('--nu', type=float, default=0.1)
    parser.add_argument('--Tend', type=float, default=1.0)
    parser.add_argument('--maxiter', type=int, default=6)
    parser.add_argument('--accept-factor', type=float, default=0.95)
    args = parser.parse_args()

    classical_params = {'quad_type': 'RADAU-RIGHT', 'num_nodes': 3, 'QI': 'IE'}
    learned_params = {
        'quad_type': 'RADAU-RIGHT',
        'num_nodes': 3,
        'QI': 'IE',
        'model_checkpoint': args.checkpoint,
        'accept_factor': args.accept_factor,
        'model_device': 'cpu',
        'fallback_sweeper_class': generic_implicit,
    }

    baseline = run_setup(
        generic_implicit,
        classical_params,
        args.dt,
        args.Tend,
        args.maxiter,
        problem=args.problem,
        lam=args.lam,
        nu=args.nu,
    )
    learned = run_setup(
        LearnedQDeltaSweeper,
        learned_params,
        args.dt,
        args.Tend,
        args.maxiter,
        problem=args.problem,
        lam=args.lam,
        nu=args.nu,
    )

    print('=== Baseline (classical sweeper) ===')
    for k, v in baseline.items():
        print(f'{k}: {v}')

    print('\n=== Learned sweeper with fallback ===')
    for k, v in learned.items():
        print(f'{k}: {v}')


if __name__ == '__main__':
    main()


