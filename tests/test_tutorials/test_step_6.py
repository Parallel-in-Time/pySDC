import filecmp
from shutil import copyfile

from tutorial.step_6.A_classic_vs_multigrid_controller import main as main_A
from tutorial.step_6.B_odd_temporal_distribution import main as main_B
from tutorial.step_6.C_MPI_parallelization import main as main_C


def test_A():
    main_A(num_proc_list=[1,2,4,8], fname='step_6_A_out.txt')

def test_B():
    main_B()

def test_C():
    cwd = 'tutorial/step_6'
    main_C(cwd)

    # compare output with the one from part A. Should exactly be the same!
    assert filecmp.cmp('step_6_C1_out.txt', 'step_6_A_out.txt'), 'ERROR: got different results from MPI and nonMPI for even distribution of time-steps'
    assert filecmp.cmp('step_6_C2_out.txt', 'step_6_B_out.txt'), 'ERROR: got different results from MPI and nonMPI for odd distribution of time-steps'

