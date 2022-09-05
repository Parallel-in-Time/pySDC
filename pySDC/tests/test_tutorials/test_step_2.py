import pytest


@pytest.mark.base
def test_A():
    from pySDC.tutorial.step_2.A_step_data_structure import main as main_A

    main_A()


@pytest.mark.base
def test_B():
    from pySDC.tutorial.step_2.B_my_first_sweeper import main as main_B

    main_B()


@pytest.mark.base
def test_C():
    from pySDC.tutorial.step_2.C_using_pySDCs_frontend import main as main_C

    main_C()
