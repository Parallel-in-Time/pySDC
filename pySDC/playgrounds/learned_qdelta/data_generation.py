"""Generate training samples from pySDC sweeps for small prototype problems."""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_unforced
from pySDC.implementations.problem_classes.TestEquation_0D import testequation0d
from pySDC.playgrounds.learned_qdelta.dataset import (
    DATA_CONTRACT_VERSION,
    build_feature_vector,
    build_target_vector,
)
from pySDC.playgrounds.learned_qdelta.learned_sweeper import DataCollectingImplicitSweeper


def _stack_samples(samples: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = samples[0].keys()
    return {k: np.stack([s[k] for s in samples], axis=0) for k in keys}


def run_case(
    problem_class,
    problem_params: dict,
    dt: float,
    maxiter: int,
    num_nodes: int,
    nsteps: int,
    use_z_param: bool = False,
) -> list[dict[str, np.ndarray]]:
    level_params = {'dt': dt, 'restol': 1e-14}
    sweeper_params = {'quad_type': 'RADAU-RIGHT', 'num_nodes': num_nodes, 'QI': 'IE',
                      'use_z_param': use_z_param}
    step_params = {'maxiter': maxiter}
    description = {
        'problem_class': problem_class,
        'problem_params': problem_params,
        'sweeper_class': DataCollectingImplicitSweeper,
        'sweeper_params': sweeper_params,
        'level_params': level_params,
        'step_params': step_params,
    }
    controller_params = {'logger_level': 30}

    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    t0 = 0.0
    Tend = nsteps * dt
    P = controller.MS[0].levels[0].prob
    u_exact = getattr(P, 'u_exact')
    u0 = u_exact(t0)

    controller.run(u0=u0, t0=t0, Tend=Tend)

    sweeper = controller.MS[0].levels[0].sweep
    drain_samples = getattr(sweeper, 'drain_samples')
    return drain_samples()


def generate_dataset(
    output_path: str,
    problem: str = 'dahlquist',
    num_cases: int = 1000,
    nsteps: int = 4,
    maxiter: int = 4,
    num_nodes: int = 3,
    seed: int = 7,
    lambda_min: float = -1000.0,
    lambda_max: float = -0.1,
    dt_min: float = 1.0,
    dt_max: float = 1.0,
    z_min: float = 0.01,
    z_max: float = 100.0,
):
    rng = np.random.default_rng(seed)
    all_samples = []
    use_z_param = False

    for _ in range(num_cases):
        if problem == 'dahlquist':
            dt = float(rng.uniform(dt_min, dt_max))
            problem_class = testequation0d
            problem_params = {'lambdas': np.array([float(rng.uniform(lambda_min, lambda_max))]), 'u0': 1.0}
        elif problem == 'dahlquist_z':
            # z = lambda*dt: fix dt=1, sample z log-uniformly in [-z_max, -z_min]
            use_z_param = True
            dt = 1.0
            log_abs_z = float(rng.uniform(np.log(z_min), np.log(z_max)))
            z = -np.exp(log_abs_z)
            problem_class = testequation0d
            problem_params = {'lambdas': np.array([z]), 'u0': 1.0}
        elif problem == 'heat1d':
            dt = float(rng.uniform(0.002, 0.02))
            problem_class = heatNd_unforced
            problem_params = {
                'nvars': 127,
                'nu': float(rng.uniform(0.02, 0.2)),
                'freq': 2,
                'bc': 'dirichlet-zero',
            }
        elif problem == 'heat1d_multiscale':
            # Sample nvars from a set of grid sizes so the model learns
            # scale-factor corrections that generalise across resolutions.
            dt = float(rng.uniform(0.002, 0.02))
            problem_class = heatNd_unforced
            nvars_choices = [63, 127, 255, 511]
            nvars = int(rng.choice(nvars_choices))
            problem_params = {
                'nvars': nvars,
                'nu': float(rng.uniform(0.02, 0.2)),
                'freq': 2,
                'bc': 'dirichlet-zero',
            }
        else:
            raise ValueError(f'Unknown problem preset: {problem}')

        all_samples.extend(
            run_case(
                problem_class=problem_class,
                problem_params=problem_params,
                dt=dt,
                maxiter=maxiter,
                num_nodes=num_nodes,
                nsteps=nsteps,
                use_z_param=use_z_param,
            )
        )

    if not all_samples:
        raise RuntimeError('No samples were generated.')

    data = _stack_samples(all_samples)
    data['contract_version'] = np.array([DATA_CONTRACT_VERSION], dtype=np.int32)
    data['problem_name'] = np.array([problem])
    data['seed'] = np.array([seed], dtype=np.int64)
    data['use_z_param'] = np.array([int(use_z_param)], dtype=np.int32)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **data)

    print(f'Saved {data["u0"].shape[0]} samples to {out}')


AGNOSTIC_DATA_VERSION = 1


def generate_agnostic_dataset(
    output_path: str,
    problem: str = 'heat1d_multiscale',
    num_cases: int = 500,
    nsteps: int = 4,
    maxiter: int = 4,
    num_nodes: int = 3,
    seed: int = 7,
):
    """Generate a pre-projected dimension-agnostic dataset.

    Unlike ``generate_dataset``, raw state arrays (which differ in shape
    across grid sizes) are never stacked.  Instead, each sample is immediately
    projected into the fixed-size (12D feature, M-target) representation before
    saving, so the resulting `.npz` can be loaded and trained on directly.

    Supported problems: heat1d, heat1d_multiscale.
    """
    rng = np.random.default_rng(seed)
    X_list, Y_list = [], []

    for _ in range(num_cases):
        if problem == 'heat1d':
            dt = float(rng.uniform(0.002, 0.02))
            problem_class = heatNd_unforced
            problem_params = {
                'nvars': 127,
                'nu': float(rng.uniform(0.02, 0.2)),
                'freq': 2,
                'bc': 'dirichlet-zero',
            }
        elif problem == 'heat1d_multiscale':
            dt = float(rng.uniform(0.002, 0.02))
            problem_class = heatNd_unforced
            nvars = int(rng.choice([63, 127, 255, 511]))
            problem_params = {
                'nvars': nvars,
                'nu': float(rng.uniform(0.02, 0.2)),
                'freq': 2,
                'bc': 'dirichlet-zero',
            }
        else:
            raise ValueError(f'generate_agnostic_dataset: unsupported problem "{problem}"')

        raw_samples = run_case(
            problem_class=problem_class,
            problem_params=problem_params,
            dt=dt,
            maxiter=maxiter,
            num_nodes=num_nodes,
            nsteps=nsteps,
        )
        for s in raw_samples:
            x = build_feature_vector(s, dimension_agnostic=True)      # (12,)
            y = build_target_vector(s, dimension_agnostic=True)       # (M,)
            X_list.append(x)
            Y_list.append(y)

    if not X_list:
        raise RuntimeError('No samples were generated.')

    X = np.stack(X_list, axis=0)   # (N, 12)
    Y = np.stack(Y_list, axis=0)   # (N, M)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        X=X.astype(np.float32),
        Y=Y.astype(np.float32),
        agnostic_version=np.array([AGNOSTIC_DATA_VERSION], dtype=np.int32),
        problem_name=np.array([problem]),
        seed=np.array([seed], dtype=np.int64),
    )
    print(f'Saved {X.shape[0]} agnostic samples ({X.shape[1]}D features) to {out}')


def main():
    parser = argparse.ArgumentParser(description='Generate one-sweep correction data from pySDC.')
    parser.add_argument('--output', type=str, default='pySDC/playgrounds/learned_qdelta/data/sdc_sweeps.npz')
    parser.add_argument('--problem', type=str, choices=['dahlquist', 'dahlquist_z', 'heat1d', 'heat1d_multiscale'], default='dahlquist')
    parser.add_argument('--num-cases', type=int, default=40)
    parser.add_argument('--nsteps', type=int, default=4)
    parser.add_argument('--maxiter', type=int, default=4)
    parser.add_argument('--num-nodes', type=int, default=3)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--lambda-min', type=float, default=-25.0)
    parser.add_argument('--lambda-max', type=float, default=-1.0)
    parser.add_argument('--dt-min', type=float, default=0.02)
    parser.add_argument('--dt-max', type=float, default=0.2)
    parser.add_argument('--z-min', type=float, default=0.01, help='|z| lower bound for dahlquist_z')
    parser.add_argument('--z-max', type=float, default=100.0, help='|z| upper bound for dahlquist_z')
    args = parser.parse_args()

    generate_dataset(
        output_path=args.output,
        problem=args.problem,
        num_cases=args.num_cases,
        nsteps=args.nsteps,
        maxiter=args.maxiter,
        num_nodes=args.num_nodes,
        seed=args.seed,
        lambda_min=args.lambda_min,
        lambda_max=args.lambda_max,
        dt_min=args.dt_min,
        dt_max=args.dt_max,
        z_min=args.z_min,
        z_max=args.z_max,
    )


if __name__ == '__main__':
    main()

