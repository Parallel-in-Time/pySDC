import pytest


def driven_particle_error(dt, Tend=0.5):
    """
    Error of RKN for one particle driven by a uniform field E(t) = 20 cos(5t) alone (the trap's own fields switched
    off), where x(t) = x0 + v0 t + 0.8 (1 - cos(5t)) is known.
    """
    import numpy as np
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.problem_classes.PenningTrap_3D import penningtrap
    from pySDC.implementations.sweeper_classes.Runge_Kutta_Nystrom import RKN

    class driven(penningtrap):
        def eval_f(self, part, t):
            f = super().eval_f(part, t)
            f.elec[0, :] += 20 * np.cos(5 * t)
            return f

    u0 = np.array([[0, 0, 0], [1, 0, 0], [1], [1]], dtype=object)
    description = {
        'problem_class': driven,
        'problem_params': {'omega_E': 0.0, 'omega_B': 0.0, 'u0': u0, 'nparts': 1, 'sig': 0.1},
        'sweeper_class': RKN,
        'sweeper_params': {},
        'level_params': {'dt': dt},
        'step_params': {'maxiter': 1},
    }
    controller = controller_nonMPI(num_procs=1, controller_params={'logger_level': 30}, description=description)
    prob = controller.MS[0].levels[0].prob
    uend, _ = controller.run(u0=prob.u_init(), t0=0.0, Tend=Tend)
    return abs(uend.pos[0, 0] - (Tend + 0.8 * (1 - np.cos(5 * Tend))))


@pytest.mark.base
def test_order_with_a_time_dependent_force():
    """RKN is fourth order also when the force depends on time, which needs every stage evaluated at its own time."""
    import numpy as np

    errors = [driven_particle_error(dt) for dt in [1 / 50, 1 / 100, 1 / 200]]
    orders = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    assert all(np.round(orders) == 4), f'expected order 4, got {orders}'
