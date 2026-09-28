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


def run_space_time(space_comm, time_comm, nvars=(32, 32), dt=1e-2, nsteps=2, levels=2):
    """A run distributed over both communicators, returning the end value and the problem.

    Two levels and a space transfer make this PFASST rather than parallel-in-time SDC; one level
    is the same run without the coarse correction, which is what the serial reference uses.
    """
    from pySDC.implementations.controller_classes.controller_MPI import controller_MPI
    from pySDC.implementations.problem_classes.generic_MPIFFT_Laplacian import IMEX_Laplacian_MPIFFT
    from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
    from pySDC.implementations.transfer_classes.TransferMesh_MPIFFT import fft_to_fft

    # a list of resolutions is how pySDC gives each level its own problem
    resolutions = [tuple(n // 2**i for n in nvars) for i in range(levels)]

    description = {
        'problem_class': IMEX_Laplacian_MPIFFT,
        'problem_params': {
            'nvars': resolutions if levels > 1 else nvars,
            'comm': space_comm,
            'useGPU': True,
            'spectral': False,
        },
        'sweeper_class': imex_1st_order,
        'sweeper_params': {'quad_type': 'RADAU-RIGHT', 'num_nodes': 3, 'QI': 'LU'},
        'level_params': {'dt': dt, 'restol': 1e-10},
        'step_params': {'maxiter': 8},
    }
    if levels > 1:
        description['space_transfer_class'] = fft_to_fft

    controller = controller_MPI(controller_params={'logger_level': 30}, description=description, comm=time_comm)

    # a run that is multi-level in name only would still pass the comparison below, so check that
    # the coarse level was actually built
    assert len(controller.S.levels) == levels, f'asked for {levels} levels, got {len(controller.S.levels)}'

    prob = controller.S.levels[0].prob

    u0 = prob.u_init
    u0[...] = prob.xp.sin(prob.X[0]) * prob.xp.sin(prob.X[1])

    uend, _ = controller.run(u0=u0, t0=0.0, Tend=nsteps * dt)
    return uend, prob


@pytest.mark.cupy
@pytest.mark.parallel(4)
def test_PFASST_with_distributed_space_on_GPU():
    """PFASST over two parallel steps, each of them distributed over two GPUs in space.

    The GPU tests are parallel in time, or over collocation nodes, or -- since the distributed
    transform test -- in space, but never in two of those at once, and none of them ran PFASST:
    the coarse level and the `fft_to_fft` transfer between two distributed grids were untouched on
    a GPU. Checked against a single-rank run on the slice of the global array this rank owns.
    """
    from mpi4py import MPI

    world = MPI.COMM_WORLD
    assert world.size == 4, f'this test decomposes four ranks as 2x2, not {world.size}'

    # ranks sharing a time step are together in space, ranks holding the same slice are together
    # in time, so rank r sits at time r // 2 and space r % 2
    space_comm = world.Split(color=world.rank // 2)
    time_comm = world.Split(color=world.rank % 2)

    try:
        assert space_comm.size == 2 and time_comm.size == 2, 'the 2x2 split did not come out square'

        parallel, prob = run_space_time(space_comm, time_comm, levels=2)
        serial, _ = run_space_time(MPI.COMM_SELF, MPI.COMM_SELF, levels=2)

        mine = serial[prob.fft.local_slice(False)]
        difference = float(abs(parallel - mine))
        assert difference < 1e-10, f'PFASST run differs from the serial one by {difference:.3e}'

        # and the space decomposition has to have actually split something, or the comparison above
        # is between two identical serial runs
        assert prob.fft.shape(False) != prob.fft.global_shape(), 'the problem was not distributed in space'
    finally:
        space_comm.Free()
        time_comm.Free()
