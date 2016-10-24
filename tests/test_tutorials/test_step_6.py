import filecmp
from shutil import copyfile

from tutorial.step_6.A_visualize_residuals import main as main_A
from tutorial.step_6.B_classic_vs_multigrid_controller import main as main_B
from tutorial.step_6.C_multistep_SDC import main as main_C
from tutorial.step_6.D_MPI_parallelization import main as main_D

def test_A():
    main_A()

def test_B():
    main_B()

def test_C():
    main_C()

def test_D():
    cwd = 'tutorial/step_6'
    main_D(cwd)
    copyfile(cwd+'/step_6_D_out.txt','step_6_D_out.txt')
    # compare output with the one from step_5, part B. Should exactly be the same!
    assert filecmp.cmp('step_6_D_out.txt', 'step_5_B_out.txt'), 'ERROR: got different results from MPI and nonMPI'

