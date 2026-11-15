"""
The mass-matrix sweepers and the mass-matrix interface of problems, independent of any finite element library.
"""

import pytest


@pytest.mark.base
def test_apply_mass_matrix_has_no_silent_default():
    """A problem used in a mass formulation without implementing it must fail, not return u."""
    from pySDC.core.problem import Problem

    class NoMass(Problem):
        pass

    with pytest.raises(NotImplementedError, match='apply_mass_matrix'):
        NoMass.apply_mass_matrix(NoMass.__new__(NoMass), None)


def _identity_mass_level(sweeper_name, residual_type):
    """A scalar level on a mass sweeper whose mass matrix is the identity, after one sweep.

    With M = I every apply_mass_matrix returns an exact copy, so the mass residual is bitwise the
    one ``core.sweeper.Sweeper.compute_residual`` computes from the same nodes. ``u0 = 100`` keeps
    the absolute and the relative residual apart. The problems do not set ``fix_bc_for_residual``.
    """
    from pySDC.core.level import Level
    from pySDC.implementations.problem_classes.TestEquation_0D import testequation0d, test_equation_IMEX
    from pySDC.implementations.sweeper_classes.generic_implicit_mass import generic_implicit_mass
    from pySDC.implementations.sweeper_classes.imex_1st_order_mass import imex_1st_order_mass

    class Identity:
        def apply_mass_matrix(self, u):
            return self.dtype_u(u)

    if sweeper_name == 'generic_implicit_mass':
        sweeper, prob, prob_params = generic_implicit_mass, testequation0d, {'lambdas': [-1.3 + 0.4j], 'u0': 100.0}
    else:
        sweeper, prob = imex_1st_order_mass, test_equation_IMEX
        prob_params = {'lambdas_implicit': [-1.0], 'lambdas_explicit': [-0.3 + 0.5j], 'u0': 100.0}

    L = Level(
        type('IdentityMass', (Identity, prob), {}),
        prob_params,
        sweeper,
        {'num_nodes': 3, 'quad_type': 'RADAU-RIGHT', 'QI': 'LU'},
        {'dt': 0.4, 'residual_type': residual_type},
        0,
    )
    L.status.time = 0.0
    L.u[0] = L.prob.u_exact(0.0)
    L.status.sweep = 1
    L.sweep.predict()
    L.sweep.update_nodes()
    return L


@pytest.mark.base
def test_imex_mass_residual_without_fix_bc_for_residual():
    """The base Problem has no ``fix_bc_for_residual``; the IMEX mass residual must not need it."""
    from pySDC.core.sweeper import Sweeper

    L = _identity_mass_level('imex_1st_order_mass', 'full_abs')
    L.sweep.compute_residual()
    mass = L.status.residual
    Sweeper.compute_residual(L.sweep)
    assert mass == L.status.residual


@pytest.mark.base
@pytest.mark.parametrize('sweeper_name', ['generic_implicit_mass', 'imex_1st_order_mass'])
@pytest.mark.parametrize('residual_type', ['full_abs', 'last_abs', 'full_rel', 'last_rel'])
def test_mass_residual_honours_residual_type(sweeper_name, residual_type):
    """The mass sweepers must reduce the node residuals as the base sweeper does, for every type."""
    from pySDC.core.sweeper import Sweeper

    L = _identity_mass_level(sweeper_name, residual_type)
    L.sweep.compute_residual()
    mass = L.status.residual
    Sweeper.compute_residual(L.sweep)
    assert mass == L.status.residual, f'{residual_type}: mass sweeper {mass:.6e}, base {L.status.residual:.6e}'


@pytest.mark.base
@pytest.mark.parametrize('sweeper_name', ['generic_implicit_mass', 'imex_1st_order_mass'])
def test_mass_residual_rejects_unknown_residual_type(sweeper_name):
    from pySDC.core.errors import ParameterError

    L = _identity_mass_level(sweeper_name, 'nonsense')
    with pytest.raises(ParameterError, match='residual_type = nonsense not implemented'):
        L.sweep.compute_residual()
