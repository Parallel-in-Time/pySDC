import pytest


@pytest.mark.base
def test_A():
    from pySDC.tutorial.step_8.A_visualize_residuals import main as main_A

    main_A()


@pytest.mark.base
def test_B():
    from pySDC.tutorial.step_8.B_multistep_SDC import main as main_B

    main_B()


@pytest.mark.base
def test_C():
    from pySDC.tutorial.step_8.C_iteration_estimator import main as main_C

    main_C()
