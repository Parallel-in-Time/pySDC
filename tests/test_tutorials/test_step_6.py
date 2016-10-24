import filecmp
from shutil import copyfile

from tutorial.step_6.A_classic_vs_multigrid_controller import main as main_A
from tutorial.step_6.B_MPI_parallelization import main as main_B


def test_A():
    main_A()

def test_B():
    cwd = 'tutorial/step_6'
    main_B(cwd)
    copyfile(cwd+'/step_6_B_out.txt','step_6_B_out.txt')
    # compare output with the one from part A. Should exactly be the same!
    assert filecmp.cmp('step_6_B_out.txt', 'step_6_A_out.txt'), 'ERROR: got different results from MPI and nonMPI'

