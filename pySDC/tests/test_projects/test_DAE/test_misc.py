import pytest


#
# Tests that problem class enforces parameter requirements
@pytest.mark.base
def test_problem_class_main():
    from pySDC.projects.DAE.problems.simple_DAE import simple_dae_1

    # initialize problem parameters
    problem_params = dict()

    # instantiate problem
    try:
        simple_dae_1(**problem_params)
    # ensure error thrown is correct
    except Exception as error:
        assert type(error) == TypeError, "Parameter error was not thrown correctly"
    else:
        raise Exception("Parameter error was not thrown correctly")
