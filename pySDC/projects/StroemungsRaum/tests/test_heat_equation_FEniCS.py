import pytest


@pytest.mark.fenics
@pytest.mark.mpi4py
def test_problem_class():
    """
    This test checks the functionality of the problem class for the heat equation implemented in FEniCS.
    It runs a short simulation and checks if the relative error at the final time is below a certain threshold,
    indicating that the problem class is correctly implemented and can be used for time integration.
    """

    from pySDC.projects.StroemungsRaum.run_heat_equation_FEniCS import setup, run_simulation

    t0 = 0.0
    Tend = 1.0

    # run the setup to get description and controller parameters
    description, controller_params = setup(t0=t0)

    # run the simulation and get the problem, stats and relative error
    _, _, rel_err = run_simulation(description, controller_params, Tend)

    assert rel_err <= 2e-4, 'ERROR: Relative error is too high, got rel_err = {}'.format(rel_err)
