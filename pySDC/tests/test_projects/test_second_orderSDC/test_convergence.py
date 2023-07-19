import pytest


METHODS = ['RKN', 'Velocity_Verlet']


def get_sweeper(sweeper_name):
    """
    Retrieve a sweeper from a name

    Args:
        sweeper_name (str):

    Returns:
        pySDC.Sweeper.RungeKutta: The sweeper
    """
    import pySDC.implementations.sweeper_classes.Runge_Kutta_Nystrom as RK

    return eval(f'RK.{sweeper_name}')


def penningtrap_params(sweeper_name):
    import numpy as np
    from pySDC.implementations.problem_classes.PenningTrap_3D import penningtrap

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-16
    level_params['dt'] = 0.015625

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'GAUSS'
    sweeper_params['num_nodes'] = 5
    sweeper_params['do_coll_update'] = True
    sweeper_params['initial_guess'] = 'zero'  # 'zero', 'spread'

    # initialize problem parameters for the penning trap
    problem_params = dict()
    problem_params['omega_E'] = 4.9
    problem_params['omega_B'] = 25.0
    problem_params['u0'] = np.array([[10, 0, 0], [100, 0, 100], [1], [1]], dtype=object)
    problem_params['nparts'] = 1
    problem_params['sig'] = 0.1

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = penningtrap
    description['problem_params'] = problem_params
    description['sweeper_params'] = sweeper_params
    description['sweeper_class'] = sweeper_name
    description['level_params'] = level_params

    return description, controller_params


@pytest.mark.base
@pytest.mark.parametrize('axis', [0, 2])
def test_global_convergence(axis):
    import numpy as np

    expected_order, num_order = BorisSDC_global_convergence()

    assert np.isclose(
        num_order['position'][axis, :], expected_order['position'][axis, :], atol=2.6e-1
    ).all(), 'Expected order in {} {}, got {}!'.format(
        'position', expected_order['position'][axis, :], num_order['position'][axis, :]
    )


def BorisSDC_global_convergence():
    from pySDC.projects.Second_orderSDC.penningtrap_Simulation import Convergence
    from pySDC.implementations.sweeper_classes.boris_2nd_order import boris_2nd_order

    description, controller_params = penningtrap_params(boris_2nd_order)
    description['level_params']['dt'] = 0.015625 * 2
    conv = Convergence(controller_params, description, time_iter=3, K_iter=(1, 2, 3))
    conv.error_type = 'Global'
    conv.compute_global_error_data()

    conv.find_approximate_order(filename='data/Global-conv-data.txt')

    expected_order, num_order = sort_order(filename='data/Global_order_vs_approxorder.txt')

    return expected_order, num_order


def string_to_array(string):
    import numpy as np

    numbers = string.strip('[]').split()
    array = [float(num) for num in numbers]
    return np.array(array)


def sort_order(cwd='', filename='data/Local_order_vs_approxorder.txt'):
    import numpy as np

    expected_order = {'position': np.array([]).reshape([0, 3]), 'velocity': np.array([]).reshape([0, 3])}
    num_order = {'position': np.array([]).reshape([0, 3]), 'velocity': np.array([]).reshape([0, 3])}

    file = open(cwd + filename, 'r')
    while True:
        line = file.readline()
        if not line:
            break

        items = str.split(
            line,
            " * ",
        )
        expected_order['position'] = np.vstack((expected_order['position'], string_to_array(items[0])))
        expected_order['velocity'] = np.vstack((expected_order['velocity'], string_to_array(items[2])))
        num_order['position'] = np.vstack((num_order['position'], np.round(string_to_array(items[1]))))
        num_order['velocity'] = np.vstack((num_order['velocity'], np.round(string_to_array(items[3][:-1]))))

    return num_order, expected_order


def BorisSDC_horizontal_axis():
    from pySDC.projects.Second_orderSDC.penningtrap_Simulation import Convergence
    from pySDC.implementations.sweeper_classes.boris_2nd_order import boris_2nd_order

    description, controller_params = penningtrap_params(boris_2nd_order)
    description['level_params']['dt'] = 0.015625 / 4

    conv = Convergence(controller_params, description, time_iter=3)
    conv.compute_local_error_data()
    conv.find_approximate_order()

    expected_order, num_order = sort_order()

    return expected_order, num_order


@pytest.mark.base
@pytest.mark.parametrize('value', ['position', 'velocity'])
def test_horizontal_axis(value):
    import numpy as np

    expected_order, num_order = BorisSDC_horizontal_axis()

    assert np.isclose(
        num_order[value][0, :], expected_order[value][0, :], atol=2.6e-1
    ).all(), 'Expected order in {} {}, got {}!'.format(value, expected_order[value][0, :], num_order[value][0, :])


def BorisSDC_vertical_axis():
    from pySDC.projects.Second_orderSDC.penningtrap_Simulation import Convergence
    from pySDC.implementations.sweeper_classes.boris_2nd_order import boris_2nd_order

    description, controller_params = penningtrap_params(boris_2nd_order)
    description['level_params']['dt'] = 0.015625 * 4

    conv = Convergence(controller_params, description, time_iter=3)
    conv.compute_local_error_data()
    conv.find_approximate_order()

    expected_order, num_order = sort_order()
    return expected_order, num_order


@pytest.mark.base
@pytest.mark.parametrize('value', ['position', 'velocity'])
def test_vertical_axis(value):
    import numpy as np

    expected_order, num_order = BorisSDC_vertical_axis()
    assert np.isclose(
        num_order[value][2, :], expected_order[value][2, :], atol=2.6e-1
    ).all(), 'Expected order in {} {}, got {}!'.format(value, expected_order[value][2, :], num_order[value][2, :])


def numerical_order(time_data, error):
    import numpy as np

    approx_order = np.polyfit(np.log(time_data), np.log(error), 1)[0].real

    return approx_order


@pytest.mark.base
@pytest.mark.parametrize('sweeper_name', METHODS)
def test_RKN_VV(sweeper_name, cwd=''):
    import numpy as np
    from pySDC.projects.Second_orderSDC.penningtrap_Simulation import Work_precision

    description, controller_params = penningtrap_params(get_sweeper(sweeper_name))

    orders = {'RKN': 4, 'Velocity_Verlet': 2}
    description['level_params']['dt'] = 0.015625
    time_iter = np.array([1, 1 / 2, 1 / 4])
    time = description['level_params']['dt'] * time_iter

    P = Work_precision(controller_params, description, time_iter=3)

    eval(f'P.func_eval_{sweeper_name}')()
    if sweeper_name == 'RKN':
        [N, func_eval_RKN, error_RKN, *_] = P.organize_data(
            filename=cwd + 'data/func_eval_vs_error_RKN{}{}.txt'.format(P.time_iter, P.num_nodes), time_iter=P.time_iter
        )
        num_order = round(numerical_order(time, error_RKN['pos'][0, :][0]))
    elif sweeper_name == 'Velocity_Verlet':
        [N, func_eval_VV, error_VV, *_] = P.organize_data(
            filename=cwd + 'data/func_eval_vs_error_VV{}{}.txt'.format(P.time_iter, P.num_nodes), time_iter=P.time_iter
        )
        num_order = round(numerical_order(time, error_VV['pos'][0, :][0]))

    expected_order = orders.get(sweeper_name)

    assert np.isclose(num_order, expected_order, atol=2.6e-1), f'Expected order {expected_order}, got {num_order}!'


if __name__ == '__main__':
    BorisSDC_global_convergence()
