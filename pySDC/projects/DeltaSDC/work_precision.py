"""
Work-precision with genuinely low-precision solves: for a target error at T, the cheapest (M, dt, restol) in fp64
and with formats chosen live from the previous step (first step fp64). Work = node solves weighted by format cost.

:func:`sweep` runs one problem for one number of nodes over all step sizes and tolerances; the twelve sweeps
(three problems, M = 2 to 5) take about an hour together on a laptop.
"""

import numpy as np

from pySDC.implementations.problem_classes.AdvectionEquation_ND_FD import advectionNd
from pySDC.projects.DeltaSDC.eta_spec import ADVECTION, ALLEN_CAHN, HEAT, case
from pySDC.projects.DeltaSDC.genuine import COST, genuine
from pySDC.projects.DeltaSDC.inexact import FORMATS
from pySDC.projects.DeltaSDC.precision_theory import bound, kappa, record_run, schedule
from pySDC.projects.DeltaSDC.problems import allencahn_delta, heat_delta

LINEAR = {'linear_implicit': True}
PROBLEMS = {
    'heat': dict(cls=heat_delta, params=dict(HEAT, freq=1), T=0.1, dts=[0.1 / 2**j for j in range(8)], sweeper=LINEAR),
    'advection': dict(
        cls=advectionNd, params=ADVECTION, T=1.0, dts=[s / 128 for s in (16, 8, 4, 2, 1, 0.5, 0.25)], sweeper=LINEAR
    ),
    'allencahn': dict(
        cls=allencahn_delta, params=ALLEN_CAHN, T=0.016, dts=[0.016 / 2**j for j in range(0, 7)], sweeper=None
    ),
}
RESTOLS = [1e-4, 1e-6, 1e-8, 1e-10, 1e-12]
PRECISION = {'fp16': 'float16', 'fp32': 'float32', 'fp64': None}


def spec_for(name, dt, restol, M):
    p = PROBLEMS[name]
    return case(name, name, p['cls'], p['params'], dt, restol, M=M, sweeper=p['sweeper'])


def trajectory(name, dt, restol, M, mixed):
    """Integrate to T; the formats of every step after the first come from its predecessor's run."""
    p = PROBLEMS[name]
    spec = spec_for(name, dt, restol, M)
    nonlinear = p['cls'] is allencahn_delta
    u, previous, work, used = None, None, 0.0, []
    # first step: fp32 if the a-priori stiffness check passes with a wide margin, else fp64
    formats = ['fp32'] if mixed and kappa(spec) * FORMATS['fp32'] <= 1e-3 else ['fp64']
    for n in range(int(round(p['T'] / dt))):
        if mixed and previous is not None and len(previous['history']) >= 2:
            if nonlinear:  # the library's Newton-Krylov solve takes one format per step
                eta = bound(spec, run=previous)['eta_star']
                formats = [next((f for f in ('fp16', 'fp32') if FORMATS[f] <= eta), 'fp64')]
            else:
                formats = schedule(spec, run=previous)
        if nonlinear:
            params = dict(spec['problem_params'], solve_precision=PRECISION[formats[0]])
            run = record_run(spec, problem_params=params, u0=u, t0=n * dt, restol=restol, maxiter=50)
        else:
            params = dict(spec['problem_params'], schedule=tuple(formats), nodes=M)
            run = record_run(
                spec, problem_class=genuine(p['cls']), problem_params=params, u0=u, t0=n * dt, restol=restol, maxiter=50
            )
        its = len(run['history'])
        work += M * sum(COST[formats[min(k, len(formats) - 1)]] for k in range(its))
        used.append(formats[:its] if not nonlinear else formats * its)
        u, previous = run['uend'], run
    return np.asarray(u).copy(), work, used


def reference(name):
    p = PROBLEMS[name]
    dt = p['dts'][-1] / 4
    spec = spec_for(name, dt, 1e-13, 5)
    u = None
    for n in range(int(round(p['T'] / dt))):
        u = record_run(spec, u0=u, t0=n * dt, restol=1e-13, maxiter=50)['uend']
    return np.asarray(u).copy()


def sweep(name, M):
    """
    Every step size and tolerance of one problem, in fp64 and with formats chosen from the previous step.

    Returns
    -------
    list
        One dict per run: ``M``, ``dt``, ``restol``, ``mixed``, the relative error ``err`` at T against
        :func:`reference`, the ``work`` and the formats ``used`` per step. A run that fails, e.g. because
        Newton does not converge, has ``err`` infinite and the exception in ``failed``.
    """
    ref = reference(name)
    rows = []
    for dt in PROBLEMS[name]['dts']:
        for restol in RESTOLS:
            for mixed in (False, True):
                row = dict(M=M, dt=dt, restol=restol, mixed=mixed)
                try:
                    u, work, used = trajectory(name, dt, restol, M, mixed)
                except Exception as error:  # a failing configuration is part of the result, not the end of it
                    rows.append(dict(row, err=np.inf, work=np.inf, used=[], failed=repr(error)))
                    continue
                rows.append(dict(row, err=np.abs(u - ref).max() / np.abs(ref).max(), work=work, used=used))
    return rows
