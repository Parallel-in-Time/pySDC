from tutorial.step_6.A_visualize_residuals import main as main_A
from tutorial.step_6.B_classic_vs_multigrid_controller import main as main_B
from tutorial.step_6.C_multistep_SDC import main as main_C

def test_A():
    main_A()

def test_B():
    main_B()
#
def test_C():
    main_C()