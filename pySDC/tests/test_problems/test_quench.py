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


@pytest.mark.base
@pytest.mark.parametrize('name', ['Quench', 'QuenchIMEX'])
def test_parameters_registered_once(name):
    from pySDC.implementations.problem_classes import Quench

    prob = getattr(Quench, name)()
    assert not prob._parNamesReadOnly & prob._parNames, 'parameters registered as read-only and as writable'


if __name__ == '__main__':
    test_solver_imex()
