import pytest


def _collect_mpi_output(stem, sizes):
    """
    Stitch the per-rank-count files the MPI runs wrote into the one the tutorial documents.

    Each run writes its own file so the passes never have to agree on which of them truncates --
    they are separate processes, and the serial one runs last.
    """
    with open(f'data/{stem}_out.txt', 'w') as out:
        for n in sizes:
            with open(f'data/{stem}_np{n}.txt') as part:
                out.write(part.read())


@pytest.mark.base
def test_step_9_A():
    import pySDC.tutorial.step_9.A_paradiag_for_linear_problems


@pytest.mark.base
def test_step_9_B():
    import pySDC.tutorial.step_9.B_paradiag_for_nonlinear_problems


@pytest.mark.base
@pytest.mark.parametrize('problem', ['advection', 'vdp'])
def test_step_9_C(problem):

    from pySDC.tutorial.step_9.C_paradiag_in_pySDC import compare_ParaDiag_and_PFASST

    compare_ParaDiag_and_PFASST(n_steps=16, problem=problem)


@pytest.mark.mpi4py
@pytest.mark.parallel([1, 2, 4])
def test_step_9_D_MPI():
    """One rank per time-step, so the block size is the size of the communicator."""
    from pathlib import Path
    from mpi4py import MPI
    from pySDC.tutorial.step_9.playground_ParaDiag_MPI import main

    comm = MPI.COMM_WORLD
    fname = f'step_9_D_np{comm.size}.txt'
    if comm.rank == 0:
        Path('data').mkdir(parents=True, exist_ok=True)
        open('data/' + fname, 'w').close()
    comm.Barrier()

    main(fname)


@pytest.mark.mpi4py
def test_step_9_D():
    from pySDC.tutorial.step_9.D_paradiag_MPI import main as main_D

    _collect_mpi_output('step_9_D', [1, 2, 4])

    cwd = 'pySDC/tutorial/step_9'
    main_D(cwd)

    with open('data/step_9_D_out.txt', 'r') as file:
        lines = [line for line in file.read().splitlines() if line.strip()]

    # three block sizes, run with both the MPI and the virtually parallel controller
    assert len(lines) == 6, 'ERROR: expected one line per block size and controller, got %s' % len(lines)

    results = {}
    for line in lines:
        mode, rest = line.split(':', 1)
        block_size = int(rest.split('block size ')[1].split(',')[0])
        niter = rest.split('iterations ')[1].split(']')[0] + ']'
        error = float(rest.split('error ')[1])
        results[(mode.strip(), block_size)] = (niter, error)

    # windowing must not change the answer: every block size does the same number of steps in total
    iterations = {v[0] for v in results.values()}
    assert len(iterations) == 1, 'ERROR: iteration counts differ between block sizes: %s' % iterations

    errors = [v[1] for v in results.values()]
    assert max(errors) - min(errors) < 1e-9, 'ERROR: errors differ between block sizes: %s' % errors

    # and both controllers have to agree
    for block_size in [1, 2, 4]:
        assert results[('MPI', block_size)] == results[('virtual', block_size)], (
            'ERROR: MPI and virtual ParaDiag differ for block size %s' % block_size
        )


@pytest.mark.mpi4py
@pytest.mark.parallel(4)
def test_step_9_E_MPI():
    """`num_steps_total` steps in one block, so one rank each."""
    from pathlib import Path
    from mpi4py import MPI
    from pySDC.tutorial.step_9.playground_adaptive_alpha import main

    comm = MPI.COMM_WORLD
    fname = f'step_9_E_np{comm.size}.txt'
    if comm.rank == 0:
        Path('data').mkdir(parents=True, exist_ok=True)
        open('data/' + fname, 'w').close()
    comm.Barrier()

    main(fname)


@pytest.mark.mpi4py
def test_step_9_E():
    from pySDC.tutorial.step_9.E_adaptive_alpha import alpha_settings, main as main_E

    _collect_mpi_output('step_9_E', [4])

    cwd = 'pySDC/tutorial/step_9'
    main_E(cwd)

    with open('data/step_9_E_out.txt', 'r') as file:
        lines = [line for line in file.read().splitlines() if line.strip()]

    # every alpha setting, run with both the MPI and the virtually parallel controller
    assert len(lines) == 2 * len(alpha_settings), 'ERROR: expected %s lines, got %s' % (
        2 * len(alpha_settings),
        len(lines),
    )

    results = {}
    for line in lines:
        mode, rest = line.split(':', 1)
        alpha = rest.split('alpha ')[1].split('->')[0].strip()
        niter = int(rest.split('->')[1].split('iterations')[0])
        error = float(rest.split('error ')[1].split(',')[0])
        final_alpha = float(rest.split('final alpha ')[1])
        results[(mode.strip(), alpha)] = (niter, error, final_alpha)

    # alpha belongs to the method, not to the parallelization, so both controllers have to agree
    for alpha in alpha_settings:
        assert results[('MPI', str(alpha))] == results[('virtual', str(alpha))], (
            'ERROR: MPI and virtual ParaDiag differ for alpha %s' % alpha
        )

    # the adaptive strategy has to find its way to the best fixed alpha we tried without being told
    fixed = {a: results[('virtual', str(a))][0] for a in alpha_settings if a != 'adaptive'}
    adaptive_iter, _, adaptive_alpha = results[('virtual', 'adaptive')]
    assert adaptive_iter <= min(
        fixed.values()
    ), 'ERROR: adaptive alpha needed %s iterations, the best fixed one only %s (%s)' % (
        adaptive_iter,
        min(fixed.values()),
        fixed,
    )
    assert max(fixed.values()) > min(fixed.values()), 'ERROR: alpha made no difference at all: %s' % fixed

    # and it should get there with a much better conditioned alpha than the smallest fixed value
    assert adaptive_alpha > min(a for a in alpha_settings if a != 'adaptive'), (
        'ERROR: expected a better conditioned alpha than the smallest fixed one, got %s' % adaptive_alpha
    )

    # alpha changes the iteration, not the problem
    errors = [v[1] for v in results.values()]
    assert max(errors) - min(errors) < 1e-9, 'ERROR: errors differ between alpha settings: %s' % errors
