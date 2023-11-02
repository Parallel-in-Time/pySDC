import pytest


@pytest.mark.base
def test_setup_helper():
    from pySDC.helpers.setup_helper import generate_description
    from pySDC.implementations.problem_classes.AdvectionEquation_ND_FD import advectionNd
    from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit

    # build classic description
    # initialize level parameters
    level_params = {}
    level_params['dt'] = 0.05

    # initialize sweeper parameters
    sweeper_params = {}
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 3
    sweeper_params['QI'] = 'IE'

    problem_params = {'freq': 2, 'nvars': 2**9, 'c': 1.0, 'stencil_type': 'center', 'order': 4, 'bc': 'periodic'}

    # initialize step parameters
    step_params = {}
    step_params['maxiter'] = 5

    description = {}
    description['problem_class'] = advectionNd
    description['problem_params'] = problem_params
    description['sweeper_class'] = generic_implicit
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params
    description['convergence_controllers'] = {}

    easy_description = generate_description(
        problem_class=advectionNd, **problem_params, **level_params, **sweeper_params, **step_params
    )

    assert (
        easy_description == description
    ), 'The generate description function did not reproduce the desired description'

    easy_description = generate_description(
        problem_class=advectionNd,
        sweeper_class=generic_implicit,
        **problem_params,
        **level_params,
        **sweeper_params,
        **step_params
    )

    assert (
        easy_description == description
    ), 'The generate description function did not reproduce the desired description when supplying a sweeper class'


if __name__ == '__main__':
    test_setup_helper()
