import pytest


def getLevel(nvars, num_nodes, index, useMPI):
    from pySDC.core.level import Level
    from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_unforced

    if useMPI:
        from pySDC.implementations.sweeper_classes.generic_implicit_MPI import generic_implicit_MPI as sweeper_class
    else:
        from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit as sweeper_class

    level_params = {}
    level_params['problem_class'] = heatNd_unforced
    level_params['problem_params'] = {'nvars': nvars}
    level_params['sweeper_class'] = sweeper_class
    level_params['sweeper_params'] = {'num_nodes': num_nodes, 'quad_type': 'GAUSS', 'do_coll_update': True}
    level_params['level_params'] = {'dt': 1.0}
    level_params['level_index'] = index

    L = Level(**level_params)

    L.status.time = 0.0
    L.status.unlocked = True
    L.u[0] = L.prob.u_exact(t=0)
    L.sweep.predict()
    return L


def get_base_transfer(nvars, num_nodes, useMPI):
    from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh

    if useMPI:
        from pySDC.implementations.transfer_classes.BaseTransferMPI import base_transfer_MPI as transfer_class
    else:
        from pySDC.core.base_transfer import BaseTransfer as transfer_class

    params = {}
    params['fine_level'] = getLevel(nvars[0], num_nodes[0], 0, useMPI)
    params['coarse_level'] = getLevel(nvars[1], num_nodes[1], 1, useMPI)
    params['base_transfer_params'] = {}
    params['space_transfer_class'] = mesh_to_mesh
    params['space_transfer_params'] = {}
    return transfer_class(**params)


@pytest.mark.mpi4py
@pytest.mark.parametrize('nvars', [32, 16])
@pytest.mark.parametrize('num_procs', [2, 3])
def test_MPI_nonMPI_consistency(num_procs, nvars):
    import os
    import subprocess

    my_env = os.environ.copy()
    my_env['PYTHONPATH'] = '../../..:.'
    my_env['COVERAGE_PROCESS_START'] = 'pyproject.toml'

    cmd = f"mpirun -np {num_procs} python {__file__} --nvars={nvars}".split()

    p = subprocess.Popen(cmd, env=my_env, cwd=".")

    p.wait()
    assert p.returncode == 0, 'ERROR: did not get return code 0, got %s with %2i processes' % (
        p.returncode,
        num_procs,
    )


def _test_MPI_nonMPI_consistency(nvars):
    import numpy as np
    from mpi4py import MPI

    num_nodes = (MPI.COMM_WORLD.size, MPI.COMM_WORLD.size)
    _nvars = [nvars, nvars // 2]
    base_transfer_params = {'nvars': _nvars, 'num_nodes': num_nodes}

    T = {useMPI: get_base_transfer(**base_transfer_params, useMPI=useMPI) for useMPI in [True, False]}
    CF, CG = T[True].comm_fine, T[True].comm_coarse

    def assert_all_equal(operation_name):
        err = []
        if not np.allclose(*(T[useMPI].fine.u[CF.rank + 1] for useMPI in [True, False])):
            err += [f'Difference in u on fine level after {operation_name} on rank {CF.rank}']
        if not np.allclose(*(T[useMPI].fine.f[CF.rank + 1] for useMPI in [True, False])):
            err += [f'Difference in f on fine level after {operation_name} on rank {CF.rank}']
        if not np.allclose(*(T[useMPI].coarse.u[CG.rank + 1] for useMPI in [True, False])):
            err += [f'Difference in u on coarse level after {operation_name} on rank {CG.rank}']
        if not np.allclose(*(T[useMPI].coarse.f[CG.rank + 1] for useMPI in [True, False])):
            err += [f'Difference in f on coarse level after {operation_name} on rank {CG.rank}']

        if any(me is not None for me in T[False].fine.tau):
            if not np.allclose(*(T[useMPI].fine.tau[CF.rank] for useMPI in [True, False])):
                err += [f'Difference in tau correction on fine level after {operation_name} on rank {CF.rank}']
        if any(me is not None for me in T[False].coarse.tau):
            if not np.allclose(*(T[useMPI].coarse.tau[CG.rank] for useMPI in [True, False])):
                err += [f'Difference in tau correction on coarse level after {operation_name} on rank {CG.rank}']

        globel_err = CF.allgather(err)
        if any(len(me) > 0 for me in globel_err):
            raise Exception(globel_err)

    assert_all_equal('initialization')

    for function in [
        'restrict',
        'prolong',
        'prolong_f',
    ]:
        for me in T.values():
            me.__getattribute__(function)()
        assert_all_equal(function)
    print(f'Passed with {nvars=} and {CF.size=}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--nvars', type=int, nargs=1, help='Number of degrees of freedom in space')
    args = parser.parse_args()

    _test_MPI_nonMPI_consistency(args.nvars[0])
