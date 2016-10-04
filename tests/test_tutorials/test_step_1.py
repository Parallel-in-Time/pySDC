from examples.tutorial.step_1.A_spatial_problem_setup import main as main_A
from examples.tutorial.step_1.B1_spatial_accuracy_check import main as main_B1
from examples.tutorial.step_1.B2_plot_spatial_accuracy import main as main_B2
from examples.tutorial.step_1.C_collocation_problem_setup import main as main_C
from examples.tutorial.step_1.D_collocation_accuracy_check import main as main_D

def test_A():
    main_A()

def test_B1():
    main_B1()

def test_B2():
    main_B2()

def test_C():
    main_C()

def test_D():
    main_D()