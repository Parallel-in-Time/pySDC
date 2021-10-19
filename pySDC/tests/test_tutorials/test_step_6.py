import pkg_resources
import pytest

from pySDC.tutorial.step_6.A_run_non_MPI_controller import main as main_A
from pySDC.tutorial.step_6.B_odd_temporal_distribution import main as main_B
from pySDC.tutorial.step_6.C_MPI_parallelization import main as main_C


def test_A():
    main_A(num_proc_list=[1], fname='step_6_A_sl_out.txt', multi_level=False)
    main_A(num_proc_list=[1, 2, 4, 8], fname='step_6_A_ml_out.txt', multi_level=True)

def test_B():
    main_B()

@pytest.mark.parallel
def test_C():
    installed_packages = [d for d in pkg_resources.working_set]
    flat_installed_packages = [package.project_name for package in installed_packages]

    if "mpi4py" in flat_installed_packages:

        cwd = 'pySDC/tutorial/step_6'
        main_C(cwd)

        with open('step_6_C1_out.txt', 'r') as file1:
            with open('step_6_A_ml_out.txt', 'r') as file2:
                diff = set(file1).difference(file2)
        diff.discard('\n')
        for line in diff:
            assert 'iterations' not in line, 'ERROR: iteration counts differ between MPI and nonMPI for even ' \
                                             'distribution of time-steps'

        with open('step_6_C2_out.txt', 'r') as file1:
            with open('step_6_B_out.txt', 'r') as file2:
                diff = set(file1).difference(file2)
        diff.discard('\n')
        for line in diff:
            assert 'iterations' not in line, 'ERROR: iteration counts differ between MPI and nonMPI for odd distribution ' \
                                             'of time-steps'

        diff_MPI = []
        with open("step_6_C1_out.txt") as f:
            for line in f:
                if "Diff" in line:
                    diff_MPI.append(float(line.split()[1]))

        diff_nonMPI = []
        with open("step_6_A_ml_out.txt") as f:
            for line in f:
                if "Diff" in line:
                    diff_nonMPI.append(float(line.split()[1]))

        assert len(diff_MPI) == len(diff_nonMPI), 'ERROR: got different number of results form MPI and nonMPI for even ' \
                                                  'distribution of time-steps'

        for i, j in zip(diff_MPI, diff_nonMPI):
            assert abs(i-j) < 6E-11, 'ERROR: difference between MPI and nonMPI results is too large for even ' \
                                     'distributions of time-steps, got %s' %abs(i - j)

        diff_MPI = []
        with open("step_6_C2_out.txt") as f:
            for line in f:
                if "Diff" in line:
                    diff_MPI.append(float(line.split()[1]))

        diff_nonMPI = []
        with open("step_6_B_out.txt") as f:
            for line in f:
                if "Diff" in line:
                    diff_nonMPI.append(float(line.split()[1]))

        assert len(diff_MPI) == len(diff_nonMPI), 'ERROR: got different number of results form MPI and nonMPI for odd ' \
                                                  'distribution of time-steps'

        for i, j in zip(diff_MPI, diff_nonMPI):
            assert abs(i - j) < 6E-11, 'ERROR: difference between MPI and nonMPI results is too large for odd ' \
                                       'distributions of time-steps, got %s' %abs(i - j)
