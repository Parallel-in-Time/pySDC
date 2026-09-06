import pytest


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
def test_step_9_D():
    from pySDC.tutorial.step_9.D_paradiag_MPI import main as main_D

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
