import numpy as np
from pySDC.implementations.problem_classes.HarmonicOscillator import harmonic_oscillator
from pySDC.implementations.sweeper_classes.boris_2nd_order import boris_2nd_order


def get_default_harmonic_oscillator_description():
    """
    Routine to compute modules of the stability function

    Returns:
        description (dict): A dictionary containing parameters for the damped harmonic oscillator problem
    """

    # Initialize level parameters
    level_params = {'restol': 1e-16, 'dt': 1.0}

    # Initialize problem parameters for the Damped harmonic oscillator problem
    problem_params = {'k': 0, 'mu': 0, 'u0': np.array([1, 1])}

    # Initialize sweeper parameters
    sweeper_params = {'quad_type': 'GAUSS', 'num_nodes': 3, 'do_coll_update': True, 'picard_mats_sweep': True}

    # Initialize step parameters
    step_params = {'maxiter': 50}

    # Fill description dictionary for easy step instantiation
    description = {
        'problem_class': harmonic_oscillator,
        'problem_params': problem_params,
        'sweeper_class': boris_2nd_order,
        'sweeper_params': sweeper_params,
        'level_params': level_params,
        'step_params': step_params,
    }

    return description
