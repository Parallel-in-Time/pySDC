"""Convergence study for the manufactured nonlinear reaction-diffusion test."""

import json
import time
from pathlib import Path

import numpy as np

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.playgrounds.FEniCS.order_reduction.nonlinear_reaction_diffusion import (
    manufactured_reaction_diffusion_cosine,
    manufactured_reaction_diffusion_cosine_lift,
    manufactured_reaction_diffusion_sine,
)


def run_sdc(problem_class, dt, num_nodes=3, Tend=1.0, nvars=127, **problem_kwargs):
    description = {
        'problem_class': problem_class,
        'problem_params': {'nvars': nvars, **problem_kwargs},
        'sweeper_class': generic_implicit,
        'sweeper_params': {'quad_type': 'RADAU-RIGHT', 'num_nodes': num_nodes},
        'level_params': {'restol': 1e-12, 'dt': dt},
        'step_params': {'maxiter': 100},
    }
    controller = controller_nonMPI(num_procs=1, controller_params={'logger_level': 30}, description=description)
    P = controller.MS[0].levels[0].prob
    lifted = isinstance(P, manufactured_reaction_diffusion_cosine_lift)
    uinit = P.u_exact_lifted(0.0) if lifted else P.u_exact(0.0)
    start = time.perf_counter()
    uend, stats = controller.run(u0=uinit, t0=0.0, Tend=Tend)
    elapsed = time.perf_counter() - start
    if lifted:
        uend = uend + P.lift(Tend)
    error = float(abs(P.u_exact(Tend) - uend) / abs(P.u_exact(Tend)))
    iterations = [item[1] for item in __import__('pySDC.helpers.stats_helper', fromlist=['get_sorted']).get_sorted(stats, type='niter', sortby='time')]
    return error, float(np.mean(iterations)), elapsed


def local_orders(dts, errors):
    return (np.log(np.asarray(errors[:-1]) / errors[1:]) / np.log(np.asarray(dts[:-1]) / dts[1:])).tolist()


def main():
    dts = [0.5 / 2**k for k in range(5)]
    common = {'nvars': 255, 'nu': 0.1, 'reaction': 1.0, 'c': 0.25, 'Tend': 1.0}
    cases = []
    definitions = [
        (manufactured_reaction_diffusion_sine, 'Nonlinear sine (constant-in-time BCs)'),
        (manufactured_reaction_diffusion_cosine, 'Nonlinear cosine (time-dependent BCs)'),
        (manufactured_reaction_diffusion_cosine_lift, 'Nonlinear cosine + boundary lifting'),
    ]
    for problem_class, label in definitions:
        errors, iterations, timings = [], [], []
        print(f'\n>>> {label}')
        for dt in dts:
            error, mean_iter, elapsed = run_sdc(problem_class, dt, **common)
            errors.append(error)
            iterations.append(mean_iter)
            timings.append(elapsed)
            print(f'    dt={dt:.5f}  rel.err={error:.4e}  mean_iter={mean_iter:.2f}  ({elapsed:.2f}s)')
        result = {
            'label': label,
            'problem_class': problem_class.__name__,
            'num_nodes': 3,
            'expected_order': 5,
            'dts': dts,
            'errors': errors,
            'mean_iters': iterations,
            'wall_times': timings,
            'local_orders': local_orders(dts, errors),
            'fitted_order': float(np.polyfit(np.log(dts), np.log(errors), 1)[0]),
        }
        cases.append(result)
        print(f"    -> fitted order = {result['fitted_order']:.2f}")

    payload = {'meta': {'equation': 'u_t = nu*u_xx + reaction*u*(1-u) + f', **common, 'dts': dts}, 'cases': cases}
    output = Path(__file__).parent / 'nonlinear_order_study_results.json'
    output.write_text(json.dumps(payload, indent=2))
    print(f'\nWrote results to {output}')


if __name__ == '__main__':
    main()
