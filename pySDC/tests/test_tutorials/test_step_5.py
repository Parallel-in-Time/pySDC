import pytest


@pytest.mark.base
def test_A():
    from pySDC.tutorial.step_5.A_multistep_multilevel_hierarchy import main as main_A

    main_A()


@pytest.mark.base
def test_B():
    from pySDC.tutorial.step_5.B_my_first_PFASST_run import main as main_B

    main_B()


@pytest.mark.base
def test_C():
    from pySDC.tutorial.step_5.C_advection_and_PFASST import main as main_C

    main_C()
