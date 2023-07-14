exec(open("check_data_folder.py").read())

import numpy as np

from pySDC.implementations.problem_classes.PenningTrap_3D import penningtrap
from pySDC.implementations.sweeper_classes.boris_2nd_order import boris_2nd_order
from pySDC.projects.Second_orderSDC.penningtrap_Simulation import Work_precision


def penningtrap_param():
    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-16
    level_params['dt'] = 0.015625 * 2

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'GAUSS'
    sweeper_params['num_nodes'] = 5
    sweeper_params['do_coll_update'] = True
    sweeper_params['initial_guess'] = 'random'  # 'zero', 'spread'

    # initialize problem parameters for the penning trap
    problem_params = dict()
    problem_params['omega_E'] = 4.9
    problem_params['omega_B'] = 25.0
    problem_params['u0'] = np.array([[10, 0, 0], [100, 0, 100], [1], [1]], dtype=object)
    problem_params['nparts'] = 1
    problem_params['sig'] = 0.1

    # problem_params['Tend']=128 * 0.015625

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = penningtrap
    description['problem_params'] = problem_params
    description['sweeper_params'] = sweeper_params
    description['sweeper_class'] = boris_2nd_order
    description['level_params'] = level_params

    return controller_params, description


if __name__ == '__main__':
    """
    Convergence plot for the second order SDC:
        K_iter: The number of iterations
        time_iter: the number of time slices in the time/2**time_iter
        axes: Axis to show the plot
        RKN True or False: To implement together RKN method
        VV True or False: To implement Velocity-Verlet scheme
    """
    Tend = 128 * 0.015625
    controller_params, description = penningtrap_param()
    work_pre = Work_precision(
        controller_params, description, time_iter=3, Tend=Tend, K_iter=(2, 4, 6), RKN=True, VV=False, axes=(0,)
    )
    work_pre.run_work_precision
