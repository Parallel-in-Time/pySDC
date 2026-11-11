"""
Node-local solves genuinely in fp32, fp16 or bf16 arithmetic, on any CPU.

A dense LU with partial pivoting in which every operation is rounded to the format: computed in fp32,
then correctly rounded, which for +, -, * and / is exactly the format's own arithmetic, since fp32
carries more than twice the bits of fp16 and bf16 plus two (there is no fused multiply-add). Matrix
and right-hand side are rounded on input. The right-hand side arrives at unit size (the
correction-solve contract); the matrix is not scaled, so it must fit the format's range.

This is a backward-stable solver in the format -- the ``residual`` model of
:mod:`.precision_theory` -- and slow: meant for the 1D test problems, not for production.
"""

import numpy as np


def rounder(fmt):
    if fmt == 'fp32':
        return lambda x: np.asarray(x, dtype=np.float32)
    if fmt == 'fp16':
        return lambda x: np.asarray(x, dtype=np.float32).astype(np.float16).astype(np.float32)
    if fmt == 'bf16':

        def bf16(x):
            x = np.asarray(x, dtype=np.float32)
            bits = np.atleast_1d(x).view(np.uint32)
            out = ((bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))) & np.uint32(0xFFFF0000)).view(np.float32)
            return out.reshape(x.shape)

        return bf16
    raise ValueError(fmt)


def lu(B, fmt):
    r = rounder(fmt)
    A = r(B).copy()
    n = len(A)
    piv = np.arange(n)
    for k in range(n - 1):
        p = k + int(np.argmax(np.abs(A[k:, k])))
        if p != k:
            A[[k, p]] = A[[p, k]]
            piv[[k, p]] = piv[[p, k]]
        A[k + 1 :, k] = r(A[k + 1 :, k] / A[k, k])
        A[k + 1 :, k + 1 :] = r(A[k + 1 :, k + 1 :] - r(np.outer(A[k + 1 :, k], A[k, k + 1 :])))
    if not np.all(np.isfinite(A)):
        raise FloatingPointError(f'{fmt} LU overflowed')
    return A, piv


def lu_solve(factors, b, fmt):
    A, piv = factors
    r = rounder(fmt)
    x = r(np.asarray(b)[piv]).copy()
    n = len(x)
    for k in range(n - 1):  # L, unit diagonal, column by column
        x[k + 1 :] = r(x[k + 1 :] - r(A[k + 1 :, k] * x[k]))
    for k in range(n - 1, -1, -1):
        x[k] = r(x[k] / A[k, k])
        x[:k] = r(x[:k] - r(A[:k, k] * x[k]))
    return x.astype(np.float64)


def genuine(problem_class):
    """
    Subclass ``problem_class`` (with a matrix ``A`` and a linear ``solve_system``) so that its solves
    run genuinely in the format given per iteration by ``schedule`` (a tuple, the last entry
    repeated); ``nodes`` is the number of solves per iteration.
    """

    class lowprec(problem_class):
        def __init__(self, schedule=('fp64',), nodes=3, **kwargs):
            super().__init__(**kwargs)
            self._makeAttributeAndRegister('schedule', 'nodes', localVars=locals())
            self._call, self._factors = 0, {}

        def solve_system(self, rhs, factor, u0, t):
            fmt = self.schedule[min(self._call // self.nodes, len(self.schedule) - 1)]
            self._call += 1
            if fmt == 'fp64':
                return super().solve_system(rhs, factor, u0, t)
            key = (fmt, factor)
            if key not in self._factors:
                B = np.eye(self.A.shape[0]) - factor * self.A.toarray()
                self._factors[key] = lu(B, fmt)
            me = self.dtype_u(self.init)
            me[:] = lu_solve(self._factors[key], np.asarray(rhs).reshape(-1), fmt).reshape(me.shape)
            return me

    lowprec.__name__ = lowprec.__qualname__ = f'{problem_class.__name__}_genuine'
    return lowprec


COST = {'fp16': 0.25, 'fp32': 0.5, 'fp64': 1.0}
"""Cost of a solve relative to fp64: its storage size."""


def table(specs):
    """
    Iterations with genuinely low-precision solves, uniform per format and as scheduled, against the
    exact run and the prediction of the ``residual`` model. Allen-Cahn uses its own Newton-Krylov
    solve at ``solve_precision`` (fp16 stored, computed in fp32), uniformly.

    Returns
    -------
    list
        One dict per test case.
    """
    from pySDC.projects.DeltaSDC.inexact import FORMATS
    from pySDC.projects.DeltaSDC.precision_theory import bound, iterations, record_run, schedule
    from pySDC.projects.DeltaSDC.problems import allencahn_delta

    rows = []
    for spec in specs:
        M = spec['sweeper_params']['num_nodes']
        exact = record_run(spec)
        kbar = iterations(exact['history'], spec['restol'])
        b = bound(spec, run=exact)
        row = {'label': spec['label'], 'kbar': kbar, 'kappa': b['kappa'], 'theorem': b['theorem'], 'rate': b['rate']}
        nonlinear = spec['problem_class'] is allencahn_delta
        for fmt in ('fp32', 'fp16') if nonlinear else ('fp32', 'fp16', 'bf16'):
            if nonlinear:
                params = dict(spec['problem_params'], solve_precision={'fp32': 'float32', 'fp16': 'float16'}[fmt])
                run = record_run(spec, problem_params=params, maxiter=40)
            else:
                params = dict(spec['problem_params'], schedule=(fmt,), nodes=M)
                run = record_run(spec, problem_class=genuine(spec['problem_class']), problem_params=params, maxiter=40)
            row[fmt] = (iterations(run['history'], spec['restol']), FORMATS[fmt] <= b['eta_star'])
        if not nonlinear:
            formats = schedule(spec, run=exact)
            params = dict(spec['problem_params'], schedule=tuple(formats), nodes=M)
            run = record_run(spec, problem_class=genuine(spec['problem_class']), problem_params=params, maxiter=40)
            its = iterations(run['history'], spec['restol'])
            row['schedule'] = (formats, its, sum(COST[formats[min(k, len(formats) - 1)]] for k in range(int(its))))
        rows.append(row)
    return rows


def live(spec, steps=10):
    """
    Time stepping with live data: every step's formats come from the previous step's run alone
    (its residuals and corrections, as recorded with inexact solves); the first step runs in fp64.
    Allen-Cahn chooses one format per step, the linear problems one per iteration.

    Returns
    -------
    list
        Per step: the formats, the iterations, the iterations of the same step in fp64, and the cost.
    """
    from pySDC.projects.DeltaSDC.inexact import FORMATS
    from pySDC.projects.DeltaSDC.precision_theory import bound, iterations, record_run, schedule
    from pySDC.projects.DeltaSDC.problems import allencahn_delta

    nonlinear = spec['problem_class'] is allencahn_delta
    M = spec['sweeper_params']['num_nodes']
    restol = spec['restol']
    u, u64, previous, out = None, None, None, []
    for n in range(steps):
        t0 = n * spec['dt']
        if previous is None:
            formats = ['fp64']
        elif len(previous['history']) < 2:  # converged at once: nothing to learn from, keep the formats
            pass
        elif nonlinear:
            eta = bound(spec, run=previous)['eta_star']
            formats = [next(f for f in ('fp16', 'fp32', 'fp64') if FORMATS[f] <= eta)]
        else:
            formats = schedule(spec, run=previous)
        if nonlinear:
            precision = {'fp16': 'float16', 'fp32': 'float32', 'fp64': None}[formats[0]]
            run = record_run(
                spec,
                problem_params=dict(spec['problem_params'], solve_precision=precision),
                u0=u,
                t0=t0,
                restol=restol,
                maxiter=40,
            )
        else:
            params = dict(spec['problem_params'], schedule=tuple(formats), nodes=M)
            run = record_run(
                spec,
                problem_class=genuine(spec['problem_class']),
                problem_params=params,
                u0=u,
                t0=t0,
                restol=restol,
                maxiter=40,
            )
        reference = record_run(spec, u0=u64, t0=t0, restol=restol, maxiter=40)
        its, its64 = len(run['history']), len(reference['history'])
        out.append((formats, its, its64, sum(COST[formats[min(k, len(formats) - 1)]] for k in range(its))))
        u, u64, previous = run['uend'], reference['uend'], run
    return out
