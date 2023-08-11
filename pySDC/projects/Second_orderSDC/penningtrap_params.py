import numpy as np

from pySDC.implementations.problem_classes.PenningTrap_3D import penningtrap
from pySDC.implementations.sweeper_classes.boris_2nd_order import boris_2nd_order



def penningtrap_params():
    """
    Define the penning trap probem params
    Returns:
        controller params
        description
    """
    # It checks whether data folder exicits or not
    exec(open("check_data_folder.py").read())
    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-16
    # It needs to be changed according to the axis that you are choosing
    level_params['dt'] = 0.015625

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'GAUSS'
    sweeper_params['num_nodes'] = 3
    sweeper_params['do_coll_update'] = True
    # initial guess can be changed and it affects the convergence order of the SDC method
    sweeper_params['initial_guess'] = 'random'  # 'zero', 'spread'

    # initialize problem parameters for the penning trap
    problem_params = dict()
    problem_params['omega_E'] = 4.9 # amplititude of electric field
    problem_params['omega_B'] = 25.0 # amplititude of magnetic field
    problem_params['u0'] = np.array([[10, 0, 0], [100, 0, 100], [1], [1]], dtype=object) # initial condition together q and m values
    problem_params['nparts'] = 1 # number of particles
    problem_params['sig'] = 0.1 # sigma

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
