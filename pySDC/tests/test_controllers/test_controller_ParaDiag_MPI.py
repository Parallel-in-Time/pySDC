import os
import subprocess
import sys

import numpy as np
import pytest


def get_composite_collocation_problem(L, M, N, alpha=1e-4, dt=1e-1, problem='Dahlquist', useMPI=False, comm=None):
    """
    Build a ParaDiag controller for one of a few small test problems.

    Args:
        L (int): number of parallel time steps
        M (int): number of collocation nodes
        N (int): number of degrees of freedom in space
        alpha: ParaDiag alpha; number, sequence or callable of the iteration index
        dt (float): step size
        problem (str): 'Dahlquist', 'Dahlquist_IMEX' or 'vdp'
        useMPI (bool): whether to use the MPI controller
        comm: MPI communicator, only for useMPI=True

    Returns:
        tuple: the controller and the problem of its first local step
    """
    from pySDC.implementations.sweeper_classes.ParaDiagSweepers import QDiagonalization, QDiagonalizationIMEX

    average_jacobian = False
    if problem == 'Dahlquist':
        from pySDC.implementations.problem_classes.TestEquation_0D import testequation0d as problem_class

        sweeper_class = QDiagonalization
        problem_params = {'lambdas': -1.0 * np.ones(shape=(N)), 'u0': 1}
    elif problem == 'Dahlquist_IMEX':
        from pySDC.implementations.problem_classes.TestEquation_0D import test_equation_IMEX as problem_class

        sweeper_class = QDiagonalizationIMEX
        problem_params = {
            'lambdas_implicit': -1.0 * np.ones(shape=(N)),
            'lambdas_explicit': -1.0e-1 * np.ones(shape=(N)),
            'u0': 1.0,
        }
    elif problem == 'vdp':
        from pySDC.implementations.problem_classes.Van_der_Pol_implicit import vanderpol as problem_class

        sweeper_class = QDiagonalization
        problem_params = {'newton_maxiter': 1, 'mu': 1e0, 'crash_at_maxiter': False}
        average_jacobian = True
    else:
        raise NotImplementedError(f'No such problem: {problem!r}')

    description = {
        'problem_class': problem_class,
        'problem_params': problem_params,
        'sweeper_class': sweeper_class,
        'sweeper_params': {'quad_type': 'RADAU-RIGHT', 'num_nodes': M, 'initial_guess': 'spread'},
        'level_params': {'dt': dt, 'restol': 1e-8},
        'step_params': {'maxiter': 99},
    }
    controller_params = {'logger_level': 30, 'alpha': alpha, 'average_jacobian': average_jacobian}

    if useMPI:
        from pySDC.implementations.controller_classes.controller_ParaDiag_MPI import controller_ParaDiag_MPI

        controller = controller_ParaDiag_MPI(controller_params=controller_params, description=description, comm=comm)
        steps = [controller.S]
    else:
        from pySDC.implementations.controller_classes.controller_ParaDiag_nonMPI import controller_ParaDiag_nonMPI

        controller_params['mssdc_jac'] = False
        controller = controller_ParaDiag_nonMPI(
            controller_params=controller_params, description=description, num_procs=L
        )
        steps = controller.MS

    # ParaDiag diagonalises in time, so the solution is complex
    for prob in [S.levels[0].prob for S in steps]:
        prob.init = tuple([*prob.init[:2]] + [np.dtype('complex128')])

    return controller, steps[0].levels[0].prob


def run_ParaDiag(L, M, N, alpha, problem, useMPI, comm=None, Tend=None, dt=1e-1):
    """Run one ParaDiag setup and return the end value plus the iteration counts."""
    from pySDC.helpers.stats_helper import get_sorted

    controller, P = get_composite_collocation_problem(
        L, M, N, alpha=alpha, dt=dt, problem=problem, useMPI=useMPI, comm=comm
    )
    Tend = L * dt if Tend is None else Tend
    uend, stats = controller.run(u0=P.u_exact(0), t0=0, Tend=Tend)
    niter = [int(v) for _, v in get_sorted(stats, type='niter', sortby='time')]
    return np.asarray(uend).astype(np.complex128).ravel(), niter


# --------------------------------------------------------------------------------------- helpers


def launch_MPI(num_procs, args):
    """Run this file under mpirun with the given argv tail."""
    my_env = os.environ.copy()
    my_env['PYTHONPATH'] = os.getcwd()
    my_env['COVERAGE_PROCESS_START'] = 'pyproject.toml'
    cmd = f'mpirun -np {num_procs} {sys.executable} {__file__} ' + ' '.join(str(a) for a in args)
    p = subprocess.Popen(cmd.split(), env=my_env, cwd='.')
    p.wait()
    assert p.returncode == 0, f'ERROR: mpirun returned {p.returncode} with {num_procs} processes'


# --------------------------------------------------------------------------------------- tests


@pytest.mark.base
@pytest.mark.parametrize('alpha', ['scalar', 'list', 'callable'])
def test_get_alpha(alpha):
    """The alpha parameter may be a number, a sequence or a callable of the iteration index."""
    spec = {'scalar': 1e-4, 'list': [1e-2, 1e-4, 1e-8], 'callable': lambda k: 1e-2 / 10 ** (2 * k)}[alpha]
    controller, _ = get_composite_collocation_problem(2, 2, 1, alpha=spec)

    if alpha == 'scalar':
        assert [controller.get_alpha(k) for k in range(4)] == [1e-4] * 4
    elif alpha == 'list':
        # the last entry is reused once the sequence runs out
        assert [controller.get_alpha(k) for k in range(4)] == [1e-2, 1e-4, 1e-8, 1e-8]
    else:
        assert [controller.get_alpha(k) for k in range(3)] == [1e-2, 1e-4, 1e-6]


@pytest.mark.base
def test_FFT_matrices_rebuilt_only_when_alpha_changes():
    """The FFT matrices are cached, and the cache is keyed on the alpha actually in use."""
    controller, _ = get_composite_collocation_problem(4, 2, 1, alpha=[1e-2, 1e-2, 1e-8])

    first = controller.get_FFT_matrices(0)
    assert controller.get_FFT_matrices(1)[0] is first[0], 'unchanged alpha should reuse the matrix'
    assert controller.get_FFT_matrices(2)[0] is not first[0], 'changed alpha must rebuild the matrix'
    assert not np.allclose(controller.get_FFT_matrices(2)[0], first[0])


@pytest.mark.base
@pytest.mark.parametrize('L', [1, 4])
def test_variable_alpha_converges_nonMPI(L):
    """A varying alpha still converges, and to the same answer as a fixed one."""
    fixed, niter_fixed = run_ParaDiag(L, 3, 2, 1e-8, 'Dahlquist', useMPI=False)
    varying, niter_var = run_ParaDiag(L, 3, 2, [1e-2, 1e-4, 1e-8], 'Dahlquist', useMPI=False)

    assert max(niter_var) < 99, 'varying alpha did not converge'
    # both are converged to the same collocation problem, so they agree to the residual tolerance
    assert np.allclose(fixed, varying, atol=1e-7), f'got {abs(fixed - varying).max():.2e}'


@pytest.mark.base
@pytest.mark.parametrize('problem', ['Dahlquist', 'Dahlquist_IMEX', 'vdp'])
def test_convergence_nonMPI(problem):
    """ParaDiag reaches the residual tolerance well inside maxiter, and matches the exact solution."""
    L, M, dt = 4, 3, 1e-1
    uend, niter = run_ParaDiag(L, M, 2, 1e-8, problem, useMPI=False, dt=dt)

    assert max(niter) < 20, f'took suspiciously many iterations: {niter}'
    if problem == 'Dahlquist':
        expected = np.exp(-L * dt)
        assert abs(uend[0] - expected) < 1e-4, f'got {uend[0]}, expected ~{expected}'


@pytest.mark.base
def test_multilevel_is_rejected():
    """ParaDiag has no multi-level version; asking for one must fail loudly."""
    from pySDC.implementations.controller_classes.controller_ParaDiag_nonMPI import controller_ParaDiag_nonMPI
    from pySDC.implementations.problem_classes.TestEquation_0D import testequation0d
    from pySDC.implementations.sweeper_classes.ParaDiagSweepers import QDiagonalization

    description = {
        'problem_class': testequation0d,
        'problem_params': {'lambdas': [[-1.0], [-1.0]], 'u0': 1},
        'sweeper_class': QDiagonalization,
        'sweeper_params': {'quad_type': 'RADAU-RIGHT', 'num_nodes': [2, 2], 'initial_guess': 'spread'},
        'level_params': {'dt': 1e-1, 'restol': 1e-8},
        'step_params': {'maxiter': 9},
    }
    with pytest.raises(Exception):
        controller_ParaDiag_nonMPI(
            controller_params={'logger_level': 30, 'alpha': 1e-4}, description=description, num_procs=2
        )


@pytest.mark.mpi4py
@pytest.mark.parametrize('L', [1, 2, 4])
@pytest.mark.parametrize('problem', ['Dahlquist', 'Dahlquist_IMEX', 'vdp'])
def test_ParaDiag_MPI_matches_nonMPI(L, problem, tmp_path):
    """
    The MPI and the virtual ParaDiag controller must agree.

    They do genuinely different arithmetic -- a dense matrix-vector product across the block versus a
    ring reduction -- so bit-identity is not expected. Measured agreement over 72 configurations is
    ~1e-16 relative, i.e. about one ulp, so 1e-14 leaves a wide margin while still catching a defect.
    """
    M, N, alpha = 3, 2, 1e-4
    out = tmp_path / 'mpi.npy'
    launch_MPI(L, ['run', L, M, N, alpha, problem, str(out)])

    mpi_uend = np.load(out)
    mpi_niter = list(np.load(str(out).replace('.npy', '_niter.npy')))
    ref_uend, ref_niter = run_ParaDiag(L, M, N, alpha, problem, useMPI=False)

    assert mpi_niter == ref_niter[-len(mpi_niter) :], f'iteration counts differ: {ref_niter} vs {mpi_niter}'
    assert np.allclose(ref_uend, mpi_uend, rtol=1e-14, atol=0), (
        f'MPI and nonMPI ParaDiag differ for {problem} on {L} processes by ' f'{abs(ref_uend - mpi_uend).max():.3e}'
    )


@pytest.mark.mpi4py
@pytest.mark.parametrize('L', [2, 4])
def test_variable_alpha_MPI(L, tmp_path):
    """An iteration dependent alpha works under MPI too, and agrees with the virtual controller."""
    out = tmp_path / 'mpi_var.npy'
    launch_MPI(L, ['run_variable_alpha', L, 3, 2, 0.0, 'Dahlquist', str(out)])

    mpi_uend = np.load(out)
    ref_uend, _ = run_ParaDiag(L, 3, 2, [1e-2, 1e-4, 1e-8], 'Dahlquist', useMPI=False)
    assert np.allclose(
        ref_uend, mpi_uend, rtol=1e-14, atol=0
    ), f'variable alpha differs by {abs(ref_uend - mpi_uend).max():.3e}'


@pytest.mark.mpi4py
def test_multiple_blocks_MPI(tmp_path):
    """Running more steps than ranks means several blocks in sequence."""
    out = tmp_path / 'mpi_blocks.npy'
    launch_MPI(2, ['run_two_blocks', 2, 3, 2, 1e-8, 'Dahlquist', str(out)])

    uend = np.load(out)
    # two blocks of two steps at dt=0.1 -> t = 0.4
    assert abs(uend[0] - np.exp(-0.4)) < 1e-4, f'got {uend[0]}, expected ~{np.exp(-0.4)}'


@pytest.mark.mpi4py
def test_solves_past_Tend_MPI(tmp_path):
    """
    ParaDiag always finishes the block it started.

    With Tend in the middle of a block it solves past it and says so, rather than truncating -- all
    steps of a block have to run, so there is no partial block to stop at.
    """
    out = tmp_path / 'mpi_past.npy'
    launch_MPI(2, ['run_past_Tend', 2, 3, 2, 1e-8, 'Dahlquist', str(out)])

    uend = np.load(out)
    # Tend=0.15 sits inside the first block of 2 x dt=0.1, so we land on t=0.2, not 0.15
    assert abs(uend[0] - np.exp(-0.2)) < 1e-4, f'got {uend[0]}, expected ~{np.exp(-0.2)}'


# --------------------------------------------------------------------------- MPI entry point

if __name__ == '__main__':
    from mpi4py import MPI

    mode = sys.argv[1]
    L, M, N = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    alpha, problem, out = float(sys.argv[5]), sys.argv[6], sys.argv[7]
    comm = MPI.COMM_WORLD

    if mode == 'run':
        uend, niter = run_ParaDiag(L, M, N, alpha, problem, useMPI=True, comm=comm)
    elif mode == 'run_variable_alpha':
        uend, niter = run_ParaDiag(L, M, N, [1e-2, 1e-4, 1e-8], problem, useMPI=True, comm=comm)
    elif mode == 'run_two_blocks':
        uend, niter = run_ParaDiag(L, M, N, alpha, problem, useMPI=True, comm=comm, Tend=4 * 1e-1)
    elif mode == 'run_past_Tend':
        uend, niter = run_ParaDiag(L, M, N, alpha, problem, useMPI=True, comm=comm, Tend=0.15)
    else:
        raise NotImplementedError(mode)

    if comm.rank == comm.size - 1:
        np.save(out, uend)
        np.save(out.replace('.npy', '_niter.npy'), np.array(niter))
