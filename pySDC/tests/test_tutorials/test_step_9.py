import pytest


@pytest.mark.base
def test_step_9_A():
    import pySDC.tutorial.step_9.A_paradiag_for_linear_problems


@pytest.mark.base
def test_step_9_B():
    import pySDC.tutorial.step_9.B_paradiag_for_nonlinear_problems


@pytest.mark.base
@pytest.mark.parametrize('problem', ['advection', 'vdp'])
def test_step_9_C(problem):

    from pySDC.tutorial.step_9.C_paradiag_in_pySDC import compare_ParaDiag_and_PFASST

    compare_ParaDiag_and_PFASST(n_steps=16, problem=problem)
