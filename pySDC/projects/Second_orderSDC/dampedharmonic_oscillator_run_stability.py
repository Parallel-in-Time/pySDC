import numpy as np

from pySDC.implementations.problem_classes.HarmonicOscillator import harmonic_oscillator
from pySDC.implementations.sweeper_classes.boris_2nd_order import boris_2nd_order
from pySDC.projects.Second_orderSDC.penningtrap_Simulation import Stability_implementation


def dampedharmonic_oscillator_params():
    """
    Runtine to compute modulues of the stability function

    Args:
        None


    Returns:
        numpy.narray: values for the spring pendulum
        numpy.narray: values for the Friction
        numpy.narray: number of num_nodes
        numpy.narray: number of iterations
        numpy.narray: moduli for the SDC
        numpy.narray: moduli for the K_{sdc} marrix
        numpy.narray: moduli for the Pircard iteration
        numpy.narray: moduli for the K_{sdc} Picard iteration
    """

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-16
    level_params["dt"] = 1.0

    # initialize problem parameters for the Damped harmonic oscillator problem
    problem_params = dict()
    problem_params["k"] = 0
    problem_params["mu"] = 0
    problem_params["u0"] = np.array([1, 1])

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'GAUSS'
    sweeper_params["num_nodes"] = 3
    sweeper_params["do_coll_update"] = True
    sweeper_params["picard_mats_sweep"] = True

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 100

    # fill description dictionary for easy step instantiation
    description = dict()
    description["problem_class"] = harmonic_oscillator
    description["problem_params"] = problem_params
    description["sweeper_class"] = boris_2nd_order
    description["sweeper_params"] = sweeper_params
    description["level_params"] = level_params
    description["step_params"] = step_params

    return description


if __name__ == '__main__':
    """
    Damped harmonic oscillatro as test problem for the stability plot:
        x'=v
        v'=-kappa*x-mu*v
        kappa: spring constant
        mu: friction

        https://beltoforion.de/en/harmonic_oscillator/
    """
    # exec(open("check_data_folder.py").read())
    description = dampedharmonic_oscillator_params()
    Stability = Stability_implementation(description, kappa_max=18, mu_max=18, Num_iter=(200, 200))
    Stability.run_SDC_stability()
    Stability.run_Picard_stability()
    Stability.run_RKN_stability()
    Stability.run_Ksdc()
    # Stability.run_Kpicard
