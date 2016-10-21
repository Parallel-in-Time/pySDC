from tutorial.step_1.A_spatial_problem_setup import main as main_A
from tutorial.step_1.B_spatial_accuracy_check import main as main_B
from tutorial.step_1.C_collocation_problem_setup import main as main_C
from tutorial.step_1.D_collocation_accuracy_check import main as main_D

def test_A():
    main_A()

def test_B():
    main_B()

def test_C():
    main_C()

def test_D():
    main_D()