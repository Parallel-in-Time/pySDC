import pytest


@pytest.mark.mpi4py
def test_A():
    from pySDC.tutorial.step_6.A_run_non_MPI_controller import main as main_A

    main_A(num_proc_list=[1], fname='step_6_A_sl_out.txt', multi_level=False)
    main_A(num_proc_list=[1, 2, 4, 8], fname='step_6_A_ml_out.txt', multi_level=True)


@pytest.mark.mpi4py
def test_B():
    from pySDC.tutorial.step_6.B_odd_temporal_distribution import main as main_B

    main_B()


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
    from pathlib import Path
    from mpi4py import MPI
    from pySDC.tutorial.step_6.C_MPI_parallelization import main

    comm = MPI.COMM_WORLD
    fname = f'step_6_C_np{comm.size}.txt'
    if comm.rank == 0:
        Path('data').mkdir(parents=True, exist_ok=True)
        open('data/' + fname, 'w').close()
    comm.Barrier()

    main(fname)


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
