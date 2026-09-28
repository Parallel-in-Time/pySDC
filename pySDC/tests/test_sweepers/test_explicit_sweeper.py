import pytest


@pytest.mark.base
@pytest.mark.parametrize('quad_type', ['RADAU-RIGHT', 'GAUSS'])
@pytest.mark.parametrize('maxiter', [1, 2, 3])
def test_explicit_sweeper_order(maxiter, quad_type):
    """
    Explicit Euler SDC gains one order per sweep. GAUSS has no node at the right end, so `compute_end_point` does the
    collocation update, which is one more application of the Picard operator and gains one more order.
    """
    import numpy as np
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.problem_classes.TestEquation_0D import testequation0d
    from pySDC.implementations.sweeper_classes.explicit import explicit

    Tend = 1.0
    dts = [Tend / 8, Tend / 16]
    errors = []
    for dt in dts:
        description = {
            'problem_class': testequation0d,
            'problem_params': {'lambdas': np.array([-1.0, -0.5]), 'u0': 1.0},
            'sweeper_class': explicit,
            'sweeper_params': {'num_nodes': 3, 'quad_type': quad_type},
            'level_params': {'dt': dt, 'restol': -1},
            'step_params': {'maxiter': maxiter},
        }
        controller = controller_nonMPI(num_procs=1, controller_params={'logger_level': 30}, description=description)
        prob = controller.MS[0].levels[0].prob
        uend, _ = controller.run(u0=prob.u_exact(0.0), t0=0.0, Tend=Tend)
        errors.append(abs(uend - prob.u_exact(Tend)))

    expected_order = maxiter + (quad_type == 'GAUSS')
    order = np.log(errors[0] / errors[1]) / np.log(dts[0] / dts[1])
    assert np.isclose(order, expected_order, atol=0.2), f'Expected order {expected_order}, got {order:.2f}'
