import pytest


def dampedharmonic_oscillator_params():
    import numpy as np
    from pySDC.implementations.problem_classes.HarmonicOscillator import harmonic_oscillator
    from pySDC.implementations.sweeper_classes.boris_2nd_order import boris_2nd_order

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
    step_params['maxiter'] = 50

    # fill description dictionary for easy step instantiation
    description = dict()
    description["problem_class"] = harmonic_oscillator
    description["problem_params"] = problem_params
    description["sweeper_class"] = boris_2nd_order
    description["sweeper_params"] = sweeper_params
    description["level_params"] = level_params
    description["step_params"] = step_params

    return description


def test_stability():
    """
    Stability domain test
    only the values of mu=[6, 20] and kappa=[3, 20]
    It is stable at mu=6, kappa=3 otherwise it is instable
    """
    import numpy as np
    from pySDC.projects.Second_orderSDC.penningtrap_Simulation import Stability_implementation

    description = dampedharmonic_oscillator_params()
    Stability = Stability_implementation(description, kappa_max=14, mu_max=14, Num_iter=(2, 2))
    Stability.lambda_kappa = np.array([6, 20])
    Stability.lambda_mu = np.array([3, 20])
    SDC, KSDC, *_ = Stability.stability_data()
    assert (
        SDC[0, 0] <= 1
    ), f'The SDC method is instable at mu={Stability.lambda_mu[0]} and kappa={Stability.lambda_kappa[0]}'
    assert (
        SDC[-1, -1] > 1
    ), f'The SDC method is stable at mu={Stability.lambda_mu[-1]} and kappa={Stability.lambda_kappa[-1]}'


if __name__ == '__main__':
    pass
