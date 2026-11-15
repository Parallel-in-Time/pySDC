"""
Tests for the forced 1D heat equation of `Heat1DForcedFiredrake`, with constant and time-dependent boundary values.
"""

import pytest

POINTS = [0.1, 0.33, 0.5, 0.77]
# u at POINTS after two steps of dt = 0.1 (64 cells, CG4, nu = 0.1, c = 1, three RADAU-RIGHT nodes, LU, exactly three
# iterations), with constant (False) and time-dependent (True) boundary values, recorded with `Heat1DForcedFiredrake`
# (Firedrake 2026.10) as a check against unintended changes.
REFERENCE = {
    False: [1.3028572528983278, 1.843584561667503, 1.9800666578411854, 1.6481297096338856],
    True: [1.932099788119632, 1.4988945321634681, 0.9999999999999997, 0.26484063166742783],
}


def run(time_dependent_bc, dt, Tend, maxiter, restol, n=64):
    """Run IMEX SDC in mass form and return the problem and the end point."""
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.problem_classes.HeatFiredrake import Heat1DForcedFiredrake
    from pySDC.implementations.sweeper_classes.imex_1st_order_mass import imex_1st_order_mass

    description = {
        'problem_class': Heat1DForcedFiredrake,
        'problem_params': {'n': n, 'order': 4, 'nu': 0.1, 'c': 1.0, 'time_dependent_bc': time_dependent_bc},
        'sweeper_class': imex_1st_order_mass,
        'sweeper_params': {'quad_type': 'RADAU-RIGHT', 'num_nodes': 3, 'QI': 'LU'},
        'level_params': {'dt': dt, 'restol': restol},
        'step_params': {'maxiter': maxiter},
    }
    controller = controller_nonMPI(num_procs=1, controller_params={'logger_level': 30}, description=description)
    P = controller.MS[0].levels[0].prob
    uend, _ = controller.run(u0=P.u_exact(0.0), t0=0.0, Tend=Tend)
    return P, uend


@pytest.mark.firedrake
@pytest.mark.parametrize('time_dependent_bc', [False, True])
def test_reference_values(time_dependent_bc):
    """Two steps give the recorded values, to round-off."""
    import firedrake as fd
    import numpy as np

    P, uend = run(time_dependent_bc, dt=0.1, Tend=0.2, maxiter=3, restol=-1.0)
    values = fd.PointEvaluator(P.V.mesh(), [[x] for x in POINTS]).evaluate(uend.functionspace)
    assert np.allclose(values, REFERENCE[time_dependent_bc], rtol=0, atol=1e-10), values


@pytest.mark.firedrake
@pytest.mark.parametrize('time_dependent_bc', [False, True])
def test_accuracy(time_dependent_bc):
    """Converged SDC on a fine mesh is accurate, with constant and with time-dependent boundary values."""
    P, uend = run(time_dependent_bc, dt=0.05, Tend=0.5, maxiter=50, restol=1e-11)
    assert abs(uend - P.u_exact(0.5)) < 1e-7
