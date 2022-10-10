import pytest


@pytest.mark.base
def test_A():
    from pySDC.tutorial.step_3.A_getting_statistics import main as main_A

    main_A()


@pytest.mark.base
def test_B():
    from pySDC.tutorial.step_3.B_adding_statistics import main as main_B

    main_B()


@pytest.mark.base
def test_C():
    from pySDC.tutorial.step_3.C_study_collocations import main as main_C

    main_C()
