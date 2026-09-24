"""
Time-parallel runs on GPUs.

`controller_MPI` sends the solution from one time rank to the next with `cupy_mesh.isend`, which
hands MPI a device pointer. That works only when MPI was built CUDA-aware *and* told to use it:
conda-forge's OpenMPI is built with it and ships it off, so the suite sets
`OMPI_MCA_opal_cuda_support=true`.
"""

import pytest


def run_time_parallel(levels, useGPU, dt=1e-2, nsteps=2):
    """A time-parallel run over however many ranks the launcher gave us."""
    from mpi4py import MPI

    from pySDC.implementations.controller_classes.controller_MPI import controller_MPI
    from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_unforced
    from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
    from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh

    # `setup_GPU` switches the class rather than the instance, so the GPU side runs on a throwaway
    # subclass and leaves the shared class alone for whatever runs next
    problem_class = type('heatNd_on_GPU', (heatNd_unforced,), {}) if useGPU else heatNd_unforced

    description = {
        'problem_class': problem_class,
        'problem_params': {
            'nvars': [32, 16][:levels] if levels > 1 else 32,
            'freq': 2,
            'bc': 'periodic',
            'useGPU': useGPU,
        },
        'sweeper_class': generic_implicit,
        'sweeper_params': {'quad_type': 'RADAU-RIGHT', 'num_nodes': 3, 'QI': 'LU'},
        'level_params': {'dt': dt, 'restol': 1e-8},
        'step_params': {'maxiter': 8},
    }
    if levels > 1:
        description['space_transfer_class'] = mesh_to_mesh
        description['space_transfer_params'] = {'rorder': 2, 'iorder': 4, 'periodic': True}

    controller = controller_MPI(controller_params={'logger_level': 30}, description=description, comm=MPI.COMM_WORLD)
    prob = controller.S.levels[0].prob
    uend, _ = controller.run(u0=prob.u_exact(0.0), t0=0.0, Tend=nsteps * dt)
    return abs(uend - prob.u_exact(nsteps * dt))


@pytest.mark.cupy
@pytest.mark.parallel(2)
@pytest.mark.parametrize('levels', [1, 2])
def test_time_parallel_on_GPU(levels):
    """One level is parallel-in-time SDC, two is PFASST. Neither had ever run on a GPU."""
    error = run_time_parallel(levels=levels, useGPU=True)
    assert error < 1e-8, f'time-parallel GPU run with {levels} level(s) was inaccurate: {error:.3e}'
