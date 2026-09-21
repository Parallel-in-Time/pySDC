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


@pytest.mark.mpi4py
@pytest.mark.parallel([1, 2, 3, 4, 5, 7, 8, 9])
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


@pytest.mark.mpi4py
def test_C():
    """
    Compare the MPI controller against the non-MPI one, per Parts A and B.

    Runs after `test_C_run` and after Parts A and B: the rank counts above are launched in their own
    passes first, and within this one pytest keeps to file order.
    """
    for out, sizes in (('step_6_C1_out.txt', [1, 2, 4, 8]), ('step_6_C2_out.txt', [3, 5, 7, 9])):
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

    diff_MPI = []
    with open("data/step_6_C1_out.txt") as f:
        for line in f:
            if "Diff" in line:
                diff_MPI.append(float(line.split()[1]))

    diff_nonMPI = []
    with open("data/step_6_A_ml_out.txt") as f:
        for line in f:
            if "Diff" in line:
                diff_nonMPI.append(float(line.split()[1]))

    assert len(diff_MPI) == len(diff_nonMPI), (
        'ERROR: got different number of results form MPI and nonMPI for even ' 'distribution of time-steps'
    )

    for i, j in zip(diff_MPI, diff_nonMPI):
        assert abs(i - j) < 6e-11, (
            'ERROR: difference between MPI and nonMPI results is too large for even '
            'distributions of time-steps, got %s' % abs(i - j)
        )

    diff_MPI = []
    with open("data/step_6_C2_out.txt") as f:
        for line in f:
            if "Diff" in line:
                diff_MPI.append(float(line.split()[1]))

    diff_nonMPI = []
    with open("data/step_6_B_out.txt") as f:
        for line in f:
            if "Diff" in line:
                diff_nonMPI.append(float(line.split()[1]))

    assert len(diff_MPI) == len(diff_nonMPI), (
        'ERROR: got different number of results form MPI and nonMPI for odd ' 'distribution of time-steps'
    )

    for i, j in zip(diff_MPI, diff_nonMPI):
        assert abs(i - j) < 6e-11, (
            'ERROR: difference between MPI and nonMPI results is too large for odd '
            'distributions of time-steps, got %s' % abs(i - j)
        )
