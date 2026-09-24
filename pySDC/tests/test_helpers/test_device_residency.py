"""
Check that a GPU run keeps its data on the GPU.

This is not a benchmark and does not look at timings, which on shared hardware say little. It
counts every copy from device to host that pySDC makes during a run and where it was made, and
requires that all of them are scalars. A field crossing the bus is a defect whatever the clock
says, and is otherwise invisible: the run still produces the right answer, only slowly.

What it cannot see is what MPI and NCCL do internally. A time-parallel run does send fields
between ranks, and whether that goes device to device or stages through host memory is a property
of how MPI was built, not of pySDC.
"""

import collections
import contextlib
import traceback

import pytest


@contextlib.contextmanager
def transfers_to_host():
    """Count copies from device to host, attributed to the pySDC line that caused them.

    Patching `cupy.ndarray.get` is enough to catch all of them: `cupy.asnumpy` and
    `float(some_device_array)` both reach it.
    """
    import cupy as cp

    counted = collections.Counter()
    largest = collections.Counter()
    original = cp.ndarray.get

    def blame():
        for frame in reversed(traceback.extract_stack()[:-2]):
            if 'pySDC' in frame.filename and 'site-packages' not in frame.filename:
                return f'{frame.filename.split("pySDC/", 1)[-1]}:{frame.lineno} in {frame.name}'
        return '<outside pySDC>'

    def counting_get(self, *args, **kwargs):
        site = blame()
        counted[site] += 1
        largest[site] = max(largest[site], self.nbytes)
        return original(self, *args, **kwargs)

    cp.ndarray.get = counting_get
    try:
        yield counted, largest
    finally:
        cp.ndarray.get = original


def build(levels, comm=None):
    """A GPU heat equation, on one space level or two, serial or parallel in time."""
    from pySDC.implementations.controller_classes.controller_MPI import controller_MPI
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_unforced
    from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
    from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh

    nvars = [128, 64] if levels > 1 else 128
    description = {
        'problem_class': heatNd_unforced,
        'problem_params': {'nvars': nvars, 'freq': 2, 'bc': 'periodic', 'useGPU': True},
        'sweeper_class': generic_implicit,
        'sweeper_params': {'quad_type': 'RADAU-RIGHT', 'num_nodes': 3, 'QI': 'LU'},
        'level_params': {'dt': 1e-2, 'restol': 1e-10},
        'step_params': {'maxiter': 20},
    }
    if levels > 1:
        description['space_transfer_class'] = mesh_to_mesh
        description['space_transfer_params'] = {'rorder': 2, 'iorder': 4, 'periodic': True}

    if comm is None:
        controller = controller_nonMPI(num_procs=1, controller_params={'logger_level': 30}, description=description)
        prob = controller.MS[0].levels[0].prob
    else:
        controller = controller_MPI(controller_params={'logger_level': 30}, description=description, comm=comm)
        prob = controller.S.levels[0].prob

    # a field, in bytes: nothing this size has any business being copied to the host mid-run
    field = (nvars[0] if levels > 1 else nvars) * 8
    return controller, prob, field


def assert_only_scalars_leave_the_device(counted, largest, field, what):
    """Every transfer during the run has to be a scalar, not a field."""
    assert counted, f'{what}: nothing was copied to the host at all, so this test measured nothing'

    too_big = {site: n for site, n in largest.items() if n > field // 8}
    assert not too_big, (
        f'{what}: these sites copied more than a scalar to the host, ' f'against a field of {field} bytes: {too_big}'
    )


@pytest.mark.cupy
@pytest.mark.parametrize('levels', [1, 2])
def test_only_scalars_leave_the_device(levels):
    """SDC on one level, MLSDC on two. The controller is built outside the count, so setup -- which
    legitimately moves matrices around -- is not what is being measured."""
    controller, prob, field = build(levels)
    u0 = prob.u_exact(0.0)

    with transfers_to_host() as (counted, largest):
        controller.run(u0=u0, t0=0.0, Tend=8e-2)

    assert_only_scalars_leave_the_device(counted, largest, field, f'{levels} level(s)')


@pytest.mark.cupy
@pytest.mark.parallel(2)
def test_only_scalars_leave_the_device_in_parallel():
    """The same for a time-parallel run, where the controller also communicates between ranks."""
    from mpi4py import MPI

    controller, prob, field = build(levels=2, comm=MPI.COMM_WORLD)
    u0 = prob.u_exact(0.0)

    with transfers_to_host() as (counted, largest):
        controller.run(u0=u0, t0=0.0, Tend=8e-2)

    assert_only_scalars_leave_the_device(counted, largest, field, 'time-parallel')
