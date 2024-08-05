import pytest


def get_dtype(name):
    if name == 'Tensor':
        from pySDC.playgrounds.ML_initial_guess.tensor import Tensor as dtype_cls
    elif name in ['mesh', 'imex_mesh']:
        import pySDC.implementations.datatype_classes.mesh as mesh

        dtype_cls = eval(f'mesh.{name}')
    else:
        raise NotImplementedError(f'Don\'t know a dtype of name {name!r}!')

    return dtype_cls


def single_test(name, useMPI=False):
    """
    This test checks that the communicator and datatype are maintained when generating new instances.
    Also, it makes sure that you can supply different communicators.
    """
    import numpy as np

    dtype_cls = get_dtype(name)

    shape = (5,)
    comm = None
    dtype = np.dtype('f')

    if useMPI:
        from mpi4py import MPI

        comm_wd = MPI.COMM_WORLD
        comm = comm_wd.Split(comm_wd.rank < comm_wd.size - 1)

        expected_rank = comm_wd.rank % (comm_wd.size - 1)

    init = (shape, comm, dtype)

    a = dtype_cls(init, val=1.0)
    b = dtype_cls(init, val=99.0)
    c = dtype_cls(a)
    d = a + b

    for me in [a, b, c, d]:
        assert type(me) == dtype_cls
        assert me.comm == comm

        if hasattr(me, 'shape') and not hasattr(me, 'components'):
            assert me.shape == shape

        if useMPI:
            assert comm.rank == expected_rank
            assert comm.size < comm_wd.size


def launch_test(name, useMPI, num_procs=1):
    if useMPI:
        import os
        import subprocess

        # Set python path once
        my_env = os.environ.copy()
        my_env['PYTHONPATH'] = '../../..:.'
        my_env['COVERAGE_PROCESS_START'] = 'pyproject.toml'

        cmd = f"mpirun -np {num_procs} python {__file__} --name={name} --useMPI=True"

        p = subprocess.Popen(cmd.split(), env=my_env, cwd=".")

        p.wait()
        assert p.returncode == 0, 'ERROR: did not get return code 0, got %s with %2i processes' % (
            p.returncode,
            num_procs,
        )
    else:
        single_test(name, False)


@pytest.mark.pytorch
@pytest.mark.parametrize('useMPI', [True, False])
def test_PyTorch_dtype(useMPI):
    launch_test('Tensor', useMPI=useMPI, num_procs=4)


@pytest.mark.mpi4py
@pytest.mark.parametrize('name', ['mesh', 'imex_mesh'])
def test_mesh_dtypes_MPI(name):
    launch_test(name, useMPI=True, num_procs=4)


@pytest.mark.base
@pytest.mark.parametrize('name', ['mesh', 'imex_mesh'])
def test_mesh_dtypes(name):
    launch_test(name, useMPI=False)


if __name__ == '__main__':
    str_to_bool = lambda me: False if me == 'False' else True
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, help='Name of the datatype')
    parser.add_argument('--useMPI', type=str_to_bool, help='Toggle for MPI', choices=[True, False])
    args = parser.parse_args()

    single_test(**vars(args))
