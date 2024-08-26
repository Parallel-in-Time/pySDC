import pytest


@pytest.mark.base
def test_solver_imex():
    from pySDC.implementations.problem_classes.Quench import QuenchIMEX
    import numpy as np

    params = {}

    P = QuenchIMEX(**params)
    u = P.u_exact(0)
    f = P.eval_f(u, 0)

    dt = 1e0
    un = P.solve_system(u + f.expl, dt, u, 0)
    fn = P.eval_f(un, dt)
    u_backwards = un - dt * fn.impl - dt * f.expl

    assert not np.allclose(
        un, 0
    ), 'Sadly, it seems as though nothing occurred in spite of the expectation to witness great commotion!'
    assert np.allclose(u, u_backwards), 'Inconsistent solver and RHS evaluation in IMEX implementation quench!'


@pytest.mark.base
@pytest.mark.parametrize('t', [0, 370])
def test_solver(t):
    from pySDC.implementations.problem_classes.Quench import Quench
    import numpy as np

    params = {}

    P = Quench(**params)
    u = P.u_exact(t=t)

    dt = 1e0
    un = P.solve_system(u, dt, u, 0)
    fn = P.eval_f(un, dt)
    u_backwards = un - dt * fn

    assert not np.allclose(
        un, 0
    ), 'Sadly, it seems as though nothing occurred in spite of the expectation to witness great commotion!'
    assert np.allclose(u, u_backwards), 'Inconsistent solver and RHS evaluation in quench!'


if __name__ == '__main__':
    test_solver_imex()
