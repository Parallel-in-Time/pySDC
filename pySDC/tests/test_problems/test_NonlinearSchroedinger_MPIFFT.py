import pytest


@pytest.mark.mpi4py
@pytest.mark.parametrize('spectral', [True, False])
def test_imex_vs_fully_implicit(spectral):
    import numpy as np
    from pySDC.implementations.problem_classes.NonlinearSchroedinger_MPIFFT import (
        nonlinearschroedinger_fully_implicit,
        nonlinearschroedinger_imex,
    )

    params = {
        'spectral': spectral,
        'nvars': (32, 32),
    }
    t = 0

    imex = nonlinearschroedinger_imex(**params)
    full = nonlinearschroedinger_fully_implicit(**params)

    u0 = imex.u_exact(t=t)

    # check right hand side evaluation
    temp = imex.eval_f(u0, t=t)
    f_imex = temp.expl + temp.impl
    f_full = full.eval_f(u0, t=t)
    assert np.allclose(
        f_imex, f_full
    ), 'Right hand side evaluations do not match between fully implicit and IMEX implementations of nonlinear Schroedinger!'

    # check solver with one step of implicit Euler
    dt = 1e-3
    args = {
        'rhs': u0.copy(),
        'factor': dt,
        'u0': u0.copy(),
        't': t,
    }
    u_imex = imex.solve_system(**args)
    u_full = full.solve_system(**args)
    u_exact = imex.u_exact(t + dt)

    e_imex = abs(u_imex - u_exact) / abs(u_exact)
    e_full = abs(u_full - u_exact) / abs(u_exact)

    assert e_imex < 7e-2, 'Error unexpectedly large in IMEX nonlinear Schroedinger problem'
    assert e_full < 2e-5, 'Error unexpectedly large in fully implicit Schroedinger problem'

    # check number of right hand side evaluations
    assert (
        imex.work_counters['rhs'].niter == full.work_counters['rhs'].niter
    ), 'Did not log the same number of right hand side evaluations'


if __name__ == '__main__':
    test_imex_vs_fully_implicit(True)
