r"""
Consistency tests across the Allen-Cahn implementations (see issue #434).

pySDC spells Allen-Cahn two ways. The FD/FFT classes solve

.. math:: u_t = \Delta u + \frac{1}{\varepsilon^2} u (1 - u^\nu),\qquad \nu = 2,

with wells at :math:`u = \pm 1`, while the MPIFFT classes solve

.. math:: u_t = \Delta u - \frac{2}{\varepsilon^2} u (1 - u)(1 - 2u) - 6 d_w u (1 - u),

with wells at :math:`u = 0, 1`. Substituting :math:`u = (1 + v) / 2` into the second turns it into
the first, with the same :math:`\varepsilon` and the same time scale, so for :math:`d_w = 0` these
are one problem under an affine change of variables:

.. math:: u_\mathrm{MPIFFT} = \frac{1 + u_\mathrm{FD/FFT}}{2},
          \qquad f_\mathrm{MPIFFT} = \frac{f_\mathrm{FD/FFT}}{2}.

These tests pin that identity down. They exist so that the convention can be unified without
silently changing the physics: whichever spelling survives, the mapped fields must still agree.

The Laplacian is the one piece that legitimately differs -- second-order FD against spectral -- so
the cross-discretization checks compare only the parts that are discretization-free.
"""

import numpy as np
import pytest

NVARS = (64, 64)
EPS = 0.04
RADIUS = 0.25
L = 1.0


def total_rhs(f):
    """Sum an rhs over whatever splitting the variant uses, so all variants become comparable."""
    for parts in (('impl', 'expl'), ('comp1', 'comp2')):
        if hasattr(f, parts[0]):
            return sum(np.asarray(getattr(f, part)) for part in parts)
    return np.asarray(f)


def family_A(cls_name):
    """One of the +-1 classes, built at the shared parameters."""
    if cls_name == 'allencahn2d_imex':
        from pySDC.implementations.problem_classes.AllenCahn_2D_FFT import allencahn2d_imex

        return allencahn2d_imex(nvars=NVARS, eps=EPS, radius=RADIUS, nu=2, L=L)

    import pySDC.implementations.problem_classes.AllenCahn_2D_FD as FD

    return getattr(FD, cls_name)(nvars=NVARS, eps=EPS, radius=RADIUS, nu=2)


def family_B():
    """The 0..1 class, at the same parameters and with the driving force switched off."""
    from pySDC.implementations.problem_classes.AllenCahn_MPIFFT import allencahn_imex

    return allencahn_imex(nvars=NVARS, eps=EPS, radius=RADIUS, L=L, spectral=False, dw=0.0)


FD_VARIANTS = [
    'allencahn_fullyimplicit',
    'allencahn_semiimplicit',
    'allencahn_semiimplicit_v2',
    'allencahn_multiimplicit',
]


@pytest.mark.base
@pytest.mark.parametrize('cls_name', FD_VARIANTS + ['allencahn2d_imex'])
def test_family_A_shares_one_initial_condition(cls_name):
    """The +-1 classes discretize differently but must start from the very same field."""
    reference = family_A('allencahn_fullyimplicit').u_exact(0.0)
    u = family_A(cls_name).u_exact(0.0)

    assert abs(np.asarray(u) - np.asarray(reference)).max() == 0.0, f'{cls_name} starts from a different field'


@pytest.mark.base
@pytest.mark.parametrize('cls_name', FD_VARIANTS[1:])
def test_FD_variants_agree_on_the_rhs(cls_name):
    """The FD variants only differ in how they split the rhs, never in what it sums to."""
    reference = total_rhs(_eval_at_t0('allencahn_fullyimplicit'))
    f = total_rhs(_eval_at_t0(cls_name))

    assert abs(f - reference).max() < 1e-12, f'{cls_name} sums to a different rhs'


def _eval_at_t0(cls_name):
    P = family_A(cls_name)
    return P.eval_f(P.u_exact(0.0), 0.0)


@pytest.mark.mpi4py
def test_FFT_and_MPIFFT_are_the_same_problem():
    """Same spectral discretization, so the affine map must hold to roundoff, term by term."""
    A, B = family_A('allencahn2d_imex'), family_B()
    uA, uB = A.u_exact(0.0), B.u_exact(0.0)

    assert abs(np.asarray(uB) - (1 + np.asarray(uA)) / 2).max() == 0.0, 'initial conditions are not affinely related'

    fA, fB = A.eval_f(uA, 0.0), B.eval_f(uB, 0.0)
    for part in ('impl', 'expl'):
        a, b = np.asarray(getattr(fA, part)), np.asarray(getattr(fB, part))
        assert abs(b - a / 2).max() < 1e-10 * max(abs(a).max(), 1.0), f'f.{part} breaks the map'


@pytest.mark.mpi4py
def test_FD_nonlinearity_matches_MPIFFT():
    """The reaction term is pointwise, so it maps exactly even across discretizations."""
    A, B = family_A('allencahn_semiimplicit'), family_B()
    fA = A.eval_f(A.u_exact(0.0), 0.0)
    fB = B.eval_f(B.u_exact(0.0), 0.0)

    a, b = np.asarray(fA.expl), np.asarray(fB.expl)
    assert abs(b - a / 2).max() < 1e-10 * abs(a).max(), 'the FD reaction term breaks the map'


@pytest.mark.mpi4py
def test_the_map_survives_time_stepping():
    """End to end: integrate both spellings and the solutions still differ only by the map."""
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.problem_classes.AllenCahn_2D_FFT import allencahn2d_imex
    from pySDC.implementations.problem_classes.AllenCahn_MPIFFT import allencahn_imex
    from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order

    dt, nsteps = 1e-4, 4

    def run(problem_class, problem_params):
        description = {
            'problem_class': problem_class,
            'problem_params': problem_params,
            'sweeper_class': imex_1st_order,
            'sweeper_params': {'quad_type': 'RADAU-RIGHT', 'num_nodes': 3, 'QI': 'LU'},
            'level_params': {'restol': -1, 'dt': dt},
            'step_params': {'maxiter': 8},
        }
        controller = controller_nonMPI(num_procs=1, controller_params={'logger_level': 30}, description=description)
        prob = controller.MS[0].levels[0].prob
        uend, _ = controller.run(u0=prob.u_exact(0.0), t0=0.0, Tend=nsteps * dt)
        return np.asarray(uend)

    uA = run(allencahn2d_imex, {'nvars': NVARS, 'eps': EPS, 'radius': RADIUS, 'nu': 2, 'L': L})
    uB = run(allencahn_imex, {'nvars': NVARS, 'eps': EPS, 'radius': RADIUS, 'L': L, 'spectral': False, 'dw': 0.0})

    error = abs(uB - (1 + uA) / 2).max()
    assert error < 1e-10, f'the two spellings drift apart by {error:.3e} after {nsteps} steps'


if __name__ == '__main__':
    test_FFT_and_MPIFFT_are_the_same_problem()
    test_FD_nonlinearity_matches_MPIFFT()
    test_the_map_survives_time_stepping()
    print('ok')
