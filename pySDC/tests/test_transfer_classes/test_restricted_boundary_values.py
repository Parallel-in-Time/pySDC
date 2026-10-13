import numpy as np
import pytest

from pySDC.implementations.problem_classes.HeatEquation_Chebychev import Heat1DUltraspherical


class Heat1DMovingBCs(Heat1DUltraspherical):
    """Heat equation with the Dirichlet data `a` and `b` of `Heat1DUltraspherical` times cos(t)"""

    def solve_system(self, rhs, dt, u0, t, *args, **kwargs):
        self.spectral.rhs_BCs_hat[0, -1] = self.a * np.cos(t)
        self.spectral.rhs_BCs_hat[0, -2] = self.b * np.cos(t)
        return super().solve_system(rhs, dt, u0, t, *args, **kwargs)


def run(num_nodes, num_procs):
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
    from pySDC.implementations.transfer_classes.TransferMesh_NoCoarse import mesh_to_mesh

    description = {
        'problem_class': Heat1DMovingBCs,
        'problem_params': {'nvars': 32, 'a': 1, 'b': 2, 'f': 4, 'nu': 1e-2},
        'sweeper_class': generic_implicit,
        'sweeper_params': {'quad_type': 'RADAU-RIGHT', 'num_nodes': num_nodes, 'QI': 'LU'},
        'level_params': {'dt': 0.25, 'restol': 1e-13},
        'step_params': {'maxiter': 40},
        'space_transfer_class': mesh_to_mesh,
    }
    controller = controller_nonMPI(num_procs=num_procs, controller_params={'logger_level': 30}, description=description)
    P = controller.MS[0].levels[0].prob
    uend, _ = controller.run(u0=P.u_exact(0), t0=0, Tend=1)
    return uend


@pytest.mark.base
@pytest.mark.parametrize('num_procs', [1, 4])
def test_multilevel_matches_SDC_with_time_dependent_BCs(num_procs):
    """
    With the data at the coarse nodes in the boundary rows, the restricted converged fine solution, whose boundary
    values interpolate the data at the fine nodes, is no fixed point of the coarse sweep: MLSDC and PFASST used to
    stall about 1e-8 away from SDC here.
    """
    u_SDC = run([4], 1)
    u_ML = run([4, 2], num_procs)
    assert abs(u_ML - u_SDC) < 1e-12, abs(u_ML - u_SDC)
