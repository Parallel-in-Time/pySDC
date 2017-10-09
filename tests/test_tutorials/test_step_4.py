from pySDC.tutorial.step_4.A_spatial_transfer_operators import main as main_A
from pySDC.tutorial.step_4.B_multilevel_hierarchy import main as main_B
from pySDC.tutorial.step_4.C_SDC_vs_MLSDC import main as main_C
from pySDC.tutorial.step_4.D_MLSDC_with_particles import main as main_D


def test_A():
    main_A()

def test_B():
    main_B()

def test_C():
    main_C()

def test_D():
    main_D()