import pytest

#
# Tests that problem class enforces parameter requirements
@pytest.mark.base
def test_problem_class_main():
    from pySDC.projects.DAE.problems.simple_DAE import simple_dae_1
    from pySDC.implementations.datatype_classes.mesh import mesh
    from pySDC.core.Errors import ParameterError

    # initialize problem parameters
    problem_params = dict()

    # instantiate problem
    try:
        prob = simple_dae_1(problem_params=problem_params, dtype_u=mesh, dtype_f=mesh)
    # ensure error thrown is correct
    except Exception as error:
        assert type(error) == ParameterError, "Parameter error was not thrown correctly"
    else:
        raise Exception("Parameter error was not thrown correctly")
