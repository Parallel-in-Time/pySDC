"""
FormFiredrake on several MPI ranks, in space and in time. Each parallel run is compared with the same setup run
serially on `COMM_SELF` on every rank. Distributed meshes number their dofs differently, so the solutions are compared
by their values at a few points.
"""

import pytest

SWEEPER_PARAMS = {'quad_type': 'RADAU-RIGHT', 'num_nodes': 3, 'QI': 'LU'}


def point_values(u, points):
    """Values of all components of `u` at `points`, the same on every rank."""
    import firedrake as fd
    import numpy as np

    parts = u.functionspace.subfunctions
    return np.concatenate([fd.PointEvaluator(p.function_space().mesh(), points).evaluate(p) for p in parts])


def setup(case, comm):
    """
    The description of a test case on meshes distributed over `comm`, and the points to compare the solutions at.

    - `imex`: Part H of tutorial step 7, IMEX SDC with time-dependent Dirichlet values on a 2D mesh
    - `mlsdc`: the same fully implicit, on two levels, so that the coarse solves take restricted boundary values
    - `grayscott`: Gray-Scott, two unknowns on a mixed space, on three levels
    - `floor`: the 1D nonlinear manufactured problem with the residual floor from the spatial error
    """
    import firedrake as fd
    from pySDC.tutorial.step_7.H_pySDC_with_Firedrake_forms import get_forms
    from pySDC.implementations.problem_classes.FormFiredrake import FormFiredrake, FormFiredrakeIMEX
    from pySDC.implementations.problem_classes.GrayScott_1D_Firedrake import GrayScott1DFiredrake
    from pySDC.implementations.sweeper_classes.generic_implicit_mass import generic_implicit_mass
    from pySDC.implementations.sweeper_classes.imex_1st_order_mass import imex_1st_order_mass
    from pySDC.implementations.transfer_classes.BaseTransfer_mass import base_transfer_mass
    from pySDC.implementations.transfer_classes.TransferFiredrakeMesh import MeshToMeshFiredrakeHierarchy

    transfer = {'base_transfer_class': base_transfer_mass, 'space_transfer_class': MeshToMeshFiredrakeHierarchy}
    description = {
        'sweeper_class': generic_implicit_mass,
        'sweeper_params': SWEEPER_PARAMS,
        'level_params': {'dt': 0.1, 'restol': 1e-8},
        'step_params': {'maxiter': 30},
    }

    if case in ['imex', 'mlsdc']:
        levels = 1 if case == 'imex' else 2
        meshes = list(fd.MeshHierarchy(fd.UnitSquareMesh(4, 4, comm=comm), 1))[::-1][:levels]
        forms = [get_forms(case == 'imex', mesh) for mesh in meshes]
        description['problem_class'] = FormFiredrakeIMEX if case == 'imex' else FormFiredrake
        description['problem_params'] = {key: [level[key] for level in forms] for key in forms[0]}
        if case == 'imex':
            description['sweeper_class'] = imex_1st_order_mass
        else:
            description.update(transfer)
        return description, [[0.3, 0.4], [0.5, 0.5], [0.8, 0.1]]

    if case == 'grayscott':
        meshes = list(fd.MeshHierarchy(fd.IntervalMesh(32, 100, comm=comm), 2))[::-1]
        description['problem_class'] = GrayScott1DFiredrake
        description['problem_params'] = {'mesh': meshes, 'solver_parameters': {'snes_rtol': 1e-13, 'snes_atol': 1e-14}}
        description['level_params']['dt'] = 1.0
        description.update(transfer)
        return description, [[40.0], [45.0], [50.0]]

    if case == 'floor':
        from test_form_firedrake import nonlinear_forms
        from pySDC.implementations.convergence_controller_classes.spatial_error_floor_firedrake import (
            SpatialErrorFloorFiredrake,
        )

        coarse_mesh, fine_mesh = fd.MeshHierarchy(fd.UnitIntervalMesh(4, comm=comm), 1)
        description['problem_class'] = FormFiredrake
        description['problem_params'] = nonlinear_forms(fd.FunctionSpace(fine_mesh, 'CG', 2), 0.1, False, smooth=False)
        description['convergence_controllers'] = {
            SpatialErrorFloorFiredrake: {'coarse_space': fd.FunctionSpace(coarse_mesh, 'CG', 2), 'Tend': 0.5}
        }
        return description, [[0.2], [0.5], [0.7]]

    raise NotImplementedError(case)


def run_nonMPI(case, comm, num_procs=1, Tend=None):
    """Run a case with `controller_nonMPI` on meshes distributed over `comm`; return point values and iterations."""
    import numpy as np
    from pySDC.helpers.stats_helper import get_sorted
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

    description, points = setup(case, comm)
    Tend = 5 * description['level_params']['dt'] if Tend is None else Tend
    controller = controller_nonMPI(num_procs=num_procs, controller_params={'logger_level': 30}, description=description)
    P = controller.MS[0].levels[0].prob
    uend, stats = controller.run(u0=P.u_exact(0.0), t0=0.0, Tend=Tend)
    return point_values(uend, points), np.array([me[1] for me in get_sorted(stats, type='niter', sortby='time')])


@pytest.mark.firedrake
@pytest.mark.parallel(2)
@pytest.mark.parametrize('case', ['imex', 'mlsdc', 'grayscott', 'floor'])
def test_space_parallel_matches_serial(case):
    """The same run distributed in space and serial gives the same solution, in the same iterations."""
    import numpy as np
    from mpi4py import MPI

    values, iterations = run_nonMPI(case, MPI.COMM_WORLD)
    values_serial, iterations_serial = run_nonMPI(case, MPI.COMM_SELF)

    assert np.array_equal(iterations, iterations_serial), (iterations, iterations_serial)
    assert np.allclose(values, values_serial, rtol=0, atol=1e-10), np.abs(values - values_serial).max()


@pytest.mark.firedrake
@pytest.mark.parallel([2, 4])
def test_pfasst_across_time_ranks():
    """
    PFASST with two time ranks (each with one or two space ranks) gives the same solution as PFASST on two steps run
    serially. On the coarse level, the initial value is a load vector, which is sent between the time ranks as such.
    """
    import numpy as np
    from mpi4py import MPI
    from pySDC.helpers.firedrake_ensemble_communicator import FiredrakeEnsembleCommunicator
    from pySDC.implementations.controller_classes.controller_MPI import controller_MPI

    ensemble = FiredrakeEnsembleCommunicator(MPI.COMM_WORLD, MPI.COMM_WORLD.size // 2)
    description, points = setup('mlsdc', ensemble.space_comm)
    controller = controller_MPI(controller_params={'logger_level': 30}, description=description, comm=ensemble)
    P = controller.S.levels[0].prob
    Tend = 2 * description['level_params']['dt']
    uend, _ = controller.run(u0=P.u_exact(0.0), t0=0.0, Tend=Tend)

    values_serial, _ = run_nonMPI('mlsdc', MPI.COMM_SELF, num_procs=2, Tend=Tend)
    values = point_values(uend, points)
    assert np.allclose(values, values_serial, rtol=0, atol=1e-10), np.abs(values - values_serial).max()
