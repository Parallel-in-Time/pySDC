import pytest


@pytest.mark.base
def test_A():
    from pySDC.tutorial.step_1.A_spatial_problem_setup import main as main_A

    main_A()


@pytest.mark.base
def test_B():
    from pySDC.tutorial.step_1.B_spatial_accuracy_check import main as main_B

    main_B()


@pytest.mark.base
def test_C():
    from pySDC.tutorial.step_1.C_collocation_problem_setup import main as main_C

    main_C()


@pytest.mark.base
def test_D():
    from pySDC.tutorial.step_1.D_collocation_accuracy_check import main as main_D

    main_D()
