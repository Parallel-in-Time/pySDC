import pytest


def _run(part):
    """The parts are notebook-style scripts, so running one is the test: afresh every time, and closing its figures"""
    import runpy

    import matplotlib.pyplot as plt

    try:
        runpy.run_module(f'pySDC.tutorial.step_6.{part}', run_name='__main__')
    finally:
        plt.close('all')


# Parts A and B write the results Part C's comparison below reads, so they run in the same (mpi4py) job
@pytest.mark.mpi4py
def test_A():
    _run('A_run_non_MPI_controller')


@pytest.mark.mpi4py
def test_B():
    _run('B_odd_temporal_distribution')


# Part C writes two output files: one with the rank counts Part A's multi-level run used, one with
# the rest. The marker is their union, so the runs and the comparison cannot drift apart.
C_OUTPUTS = {'step_6_C1_out.txt': [1, 2, 4, 8], 'step_6_C2_out.txt': [3, 5, 7, 9]}
C_RANKS = sorted(n for sizes in C_OUTPUTS.values() for n in sizes)


@pytest.mark.mpi4py
@pytest.mark.parallel(C_RANKS)
def test_C_run():
    """
    Run Part C's playground on as many ranks as the marker asks for.

    One file per rank count, so the passes do not have to agree on who truncates what; `test_C`
    below stitches them into the two files the tutorial shows.
    """
    import runpy

    # it writes data/step_6_C_np<ranks>.txt itself
    runpy.run_module('pySDC.tutorial.step_6.C_MPI_parallelization', run_name='__main__')


@pytest.mark.mpi4py
def test_C():
    """
    Compare the MPI controller against the non-MPI one, per Parts A and B.

    Runs after `test_C_run` and after Parts A and B: the rank counts above are launched in their own
    passes first, and within this one pytest keeps to file order.
    """
    for out, sizes in C_OUTPUTS.items():
        with open('data/' + out, 'w') as f:
            for n in sizes:
                with open(f'data/step_6_C_np{n}.txt') as part:
                    f.write(part.read())

    with open('data/step_6_C1_out.txt', 'r') as file1:
        with open('data/step_6_A_ml_out.txt', 'r') as file2:
            diff = set(file1).difference(file2)
    diff.discard('\n')
    for line in diff:
        assert 'iterations' not in line, (
            'ERROR: iteration counts differ between MPI and nonMPI for even ' 'distribution of time-steps'
        )

    with open('data/step_6_C2_out.txt', 'r') as file1:
        with open('data/step_6_B_out.txt', 'r') as file2:
            diff = set(file1).difference(file2)
    diff.discard('\n')
    for line in diff:
        assert 'iterations' not in line, (
            'ERROR: iteration counts differ between MPI and nonMPI for odd distribution ' 'of time-steps'
        )

    # The errors of the MPI and the non-MPI runs have to agree. This compared lines with "Diff" in them, which
    # neither part prints any more, so it compared two empty lists and checked nothing.
    diff_MPI = []
    with open("data/step_6_C1_out.txt") as f:
        for line in f:
            if "Error vs. exact solution" in line:
                diff_MPI.append(float(line.split()[-1]))

    diff_nonMPI = []
    with open("data/step_6_A_ml_out.txt") as f:
        for line in f:
            if "Error vs. exact solution" in line:
                diff_nonMPI.append(float(line.split()[-1]))

    assert len(diff_MPI) == len(diff_nonMPI), (
        'ERROR: got different number of results form MPI and nonMPI for even ' 'distribution of time-steps'
    )

    for i, j in zip(diff_MPI, diff_nonMPI, strict=True):
        assert abs(i - j) < 6e-11, (
            'ERROR: difference between MPI and nonMPI results is too large for even '
            'distributions of time-steps, got %s' % abs(i - j)
        )

    diff_MPI = []
    with open("data/step_6_C2_out.txt") as f:
        for line in f:
            if "Error vs. exact solution" in line:
                diff_MPI.append(float(line.split()[-1]))

    diff_nonMPI = []
    with open("data/step_6_B_out.txt") as f:
        for line in f:
            if "Error vs. exact solution" in line:
                diff_nonMPI.append(float(line.split()[-1]))

    assert len(diff_MPI) == len(diff_nonMPI), (
        'ERROR: got different number of results form MPI and nonMPI for odd ' 'distribution of time-steps'
    )

    for i, j in zip(diff_MPI, diff_nonMPI, strict=True):
        assert abs(i - j) < 6e-11, (
            'ERROR: difference between MPI and nonMPI results is too large for odd '
            'distributions of time-steps, got %s' % abs(i - j)
        )
