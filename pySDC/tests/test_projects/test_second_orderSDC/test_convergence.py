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


# @pytest.mark.base
# @pytest.mark.parametrize('axis', [0, 2])
# def test_global_convergence(axis):
#     import numpy as np

#     expected_order, num_order = BorisSDC_global_convergence()

#     assert np.isclose(
#         num_order['position'][axis, :], expected_order['position'][axis, :], atol=2.6e-1
#     ).all(), 'Expected order in {} {}, got {}!'.format(
#         'position', expected_order['position'][axis, :], num_order['position'][axis, :]
#     )


# def BorisSDC_global_convergence():
#     from pySDC.projects.Second_orderSDC.penningtrap_run_error import penningtrap_param
#     from pySDC.projects.Second_orderSDC.penningtrap_Simulation import Convergence


#     controller_params, description = penningtrap_param()
#     description['level_params']['dt'] = 0.015625 * 2
#     conv = Convergence(controller_params, description, time_iter=3, K_iter=(1, 2, 3))
#     conv.error_type = 'Global'
#     conv.compute_global_error_data()

#     conv.find_approximate_order(filename='data/Global-conv-data.txt')

#     expected_order, num_order = sort_order(filename='data/Global_order_vs_approxorder.txt')

#     return expected_order, num_order


def string_to_array(string):
    import numpy as np

    numbers = string.strip('[]').split()
    array = [float(num) for num in numbers]
    return np.array(array)


def sort_order(cwd='', filename='data/local_order_vs_approxorder.txt'):
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
    from pySDC.projects.Second_orderSDC.penningtrap_Simulation import compute_error
    from pySDC.projects.Second_orderSDC.penningtrap_run_error import penningtrap_param

    controller_params, description = penningtrap_param()
    description['level_params']['dt'] = 0.015625 / 8

    conv = compute_error(controller_params, description, time_iter=3)
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
    from pySDC.projects.Second_orderSDC.penningtrap_Simulation import compute_error
    from pySDC.projects.Second_orderSDC.penningtrap_run_error import penningtrap_param

    controller_params, description = penningtrap_param()
    description['level_params']['dt'] = 0.015625 * 8

    conv = compute_error(controller_params, description, time_iter=3)
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
    from pySDC.projects.Second_orderSDC.penningtrap_Simulation import compute_error
    from pySDC.projects.Second_orderSDC.penningtrap_run_error import penningtrap_param

    controller_params, description = penningtrap_param()
    description['sweeper_class'] = get_sweeper(sweeper_name)
    orders = {'RKN': 4, 'Velocity_Verlet': 2}
    description['level_params']['dt'] = 0.015625
    time_iter = np.array([1, 1 / 2, 1 / 4])
    time = description['level_params']['dt'] * time_iter

    P = compute_error(controller_params, description, time_iter=3)

    P.compute_error_RKN_VV()
    if sweeper_name == 'Velocity_Verlet':
        P.compute_error_RKN_VV(VV=True)
    if sweeper_name == 'RKN':
        [N, func_eval_RKN, error_RKN, *_] = P.organize_data(
            filename=cwd + 'data/rhs_eval_vs_global_errorRKN.txt', time_iter=P.time_iter
        )
        num_order = round(numerical_order(time, error_RKN['pos'][0, :][0]))
    elif sweeper_name == 'Velocity_Verlet':
        [N, func_eval_VV, error_VV, *_] = P.organize_data(
            filename=cwd + 'data/rhs_eval_vs_global_errorVV.txt', time_iter=P.time_iter
        )
        num_order = round(numerical_order(time, error_VV['pos'][0, :][0]))

    expected_order = orders.get(sweeper_name)

    assert np.isclose(num_order, expected_order, atol=2.6e-1), f'Expected order {expected_order}, got {num_order}!'


if __name__ == '__main__':
    pass
    # BorisSDC_global_convergence()
