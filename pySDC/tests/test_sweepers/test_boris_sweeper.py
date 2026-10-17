import numpy as np
import pytest


def get_level(quad_type, QI, QE):
    from pySDC.core.level import Level
    from pySDC.implementations.problem_classes.PenningTrap_3D import penningtrap
    from pySDC.implementations.sweeper_classes.boris_2nd_order import boris_2nd_order

    problem_params = {
        'omega_E': 4.9,
        'omega_B': 25.0,
        'u0': np.array([[10, 0, 0], [100, 0, 100], [1], [1]], dtype=object),
        'nparts': 1,
        'sig': 0.1,
    }
    sweeper_params = {'num_nodes': 3, 'quad_type': quad_type, 'QI': QI, 'QE': QE}
    return Level(penningtrap, problem_params, boris_2nd_order, sweeper_params, {'dt': 1.0 / 16}, 0)


@pytest.mark.base
@pytest.mark.parametrize('quad_type', ['RADAU-RIGHT', 'LOBATTO', 'GAUSS'])
@pytest.mark.parametrize('QI, QE', [('LU', 'EE'), ('IE', 'PIC'), ('MIN-SR-S', 'EE'), ('TRAP', 'EE')])
def test_preconditioners_the_boris_step_does_not_match_are_refused(quad_type, QI, QE):
    """These pairs used to stagnate at a residual of order one, since the Boris step adds back another trapezoid."""
    from pySDC.core.errors import ParameterError

    with pytest.raises(ParameterError):
        get_level(quad_type, QI, QE)


@pytest.mark.base
@pytest.mark.parametrize('quad_type', ['RADAU-RIGHT', 'LOBATTO', 'GAUSS'])
@pytest.mark.parametrize('QI, QE', [('IE', 'EE'), ('PIC', 'PIC')])
def test_matching_preconditioners_solve_the_collocation_problem(quad_type, QI, QE):
    L = get_level(quad_type, QI, QE)
    L.status.time = 0.0
    L.u[0] = L.prob.u_init()
    L.sweep.predict()
    for k in range(100):
        L.status.sweep = k + 1
        L.sweep.update_nodes()
    L.sweep.compute_residual()
    assert L.status.residual < 1e-12
