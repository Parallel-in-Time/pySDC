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


def _results(fname):
    """The errors and the iteration lines of one results file, in order"""
    with open('data/' + fname) as f:
        lines = [line.strip() for line in f if line.strip()]
    errors = [float(line.split()[-1]) for line in lines if 'Error vs. exact solution' in line]
    iterations = [line for line in lines if 'iterations' in line]
    return errors, iterations


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

    # Line by line and in order: the same iterations for every step, and the same errors to the precision they are
    # printed with. This used to compare sets of lines, which let a count through that also occurred elsewhere, and
    # lines with "Diff" in them, which no part prints any more, so that the errors were never compared at all.
    for mpi, non_mpi, distribution in [
        ('step_6_C1_out.txt', 'step_6_A_ml_out.txt', 'even'),
        ('step_6_C2_out.txt', 'step_6_B_out.txt', 'odd'),
    ]:
        errors_mpi, iterations_mpi = _results(mpi)
        errors_non_mpi, iterations_non_mpi = _results(non_mpi)

        assert (
            iterations_mpi == iterations_non_mpi
        ), f'ERROR: iteration counts differ between MPI and nonMPI for the {distribution} distribution of time-steps'
        assert (
            len(errors_mpi) == len(errors_non_mpi) == len(C_OUTPUTS[mpi])
        ), f'ERROR: expected one error per run for the {distribution} distribution of time-steps'
        for error_mpi, error_non_mpi in zip(errors_mpi, errors_non_mpi, strict=True):
            assert abs(error_mpi - error_non_mpi) <= 1e-8 * abs(error_non_mpi), (
                f'ERROR: MPI and nonMPI errors differ for the {distribution} distribution of time-steps: '
                f'{error_mpi} vs. {error_non_mpi}'
            )
