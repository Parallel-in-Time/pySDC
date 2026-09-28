"""
Give each test process a GPU of its own.

Nothing in pySDC selects a device, and on a batch system it does not have to: the scheduler hands
each task its own, so every process taking device 0 is correct. Inside one container with several
GPUs attached it is not -- NCCL refuses two ranks on one device, and four xdist workers sharing one
device throws away the point of splitting the suite.

Both cases are the same question asked by different launchers, so they are answered in one place:
``OMPI_COMM_WORLD_LOCAL_RANK`` when mpiexec forked the process, ``PYTEST_XDIST_WORKER`` when xdist
did. Neither is set for an ordinary serial run, where device 0 is right and this does nothing.

Enable it with ``-p gpu_bind`` and ``etc`` on the ``PYTHONPATH``; see ``etc/modal_gpu_tests.py``.
"""

import os


def _requested_device():
    """Which device this process should use, or None to leave the choice alone."""
    worker = os.environ.get('PYTEST_XDIST_WORKER')
    if worker is not None:
        return int(worker.removeprefix('gw'))

    rank = os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK')
    if rank is not None:
        return int(rank)

    return None


def pytest_configure(config):
    index = _requested_device()
    if index is None:
        return

    import cupy as cp

    # modulo, so that more workers than devices is slow rather than a crash
    cp.cuda.Device(index % cp.cuda.runtime.getDeviceCount()).use()
