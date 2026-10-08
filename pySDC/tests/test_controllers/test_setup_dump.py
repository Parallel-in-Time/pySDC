"""
The setup overview a controller prints marks with `-->` what the user set, with `->` what follows from it, and
leaves the defaults unmarked. These tests pin the cases where it did not.
"""

import pytest


def get_description():
    from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_forced
    from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order

    return {
        'problem_class': heatNd_forced,
        'problem_params': {'nvars': 127, 'bc': 'dirichlet-zero'},
        'sweeper_class': imex_1st_order,
        'sweeper_params': {'num_nodes': 3, 'quad_type': 'RADAU-RIGHT'},
        'level_params': {'dt': 0.1},
        'step_params': {'maxiter': 5},
    }


def get_setup_dump(capsys, controller_params, description):
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

    capsys.readouterr()
    controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)
    return capsys.readouterr().out


@pytest.mark.base
def test_controller_params_are_left_alone(capsys):
    # the default hooks used to be written into the caller's dictionary, and a second controller got them again
    controller_params = {'logger_level': 20}
    first = get_setup_dump(capsys, controller_params, get_description())
    assert controller_params == {'logger_level': 20}

    second = get_setup_dump(capsys, controller_params, get_description())
    hooks = [line for line in second.splitlines() if 'hook_class' in line]
    assert hooks == [line for line in first.splitlines() if 'hook_class' in line]
    assert hooks[0].count('DefaultHooks') == 1


@pytest.mark.base
def test_markers(capsys):
    lines = get_setup_dump(capsys, {'logger_level': 20}, get_description()).splitlines()

    def marker(name):
        (line,) = [line for line in lines if line.strip(' ->').startswith(name)]
        return line[:3]

    # set by the user
    for name in ['maxiter =', 'dt =', 'nvars =', 'num_nodes =', 'Problem:', 'Sweeper:']:
        assert marker(name) == '-->', name
    # defaults, including the hooks the controller always adds
    for name in ['hook_class =', 'restol =', 'nu =', 'QI =']:
        assert marker(name) == '   ', name
    # these follow from the problem and the sweeper
    for name in ['Data type u:', 'Data type f:', 'Collocation:']:
        assert marker(name) == ' ->', name


@pytest.mark.base
def test_step_params_are_optional(capsys):
    description = get_description()
    del description['step_params']
    assert 'Step:' in get_setup_dump(capsys, {'logger_level': 20}, description)
