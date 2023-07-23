import pytest


def test_stability():
    """
    Stability domain test
    only the values of mu=[6, 20] and kappa=[3, 20]
    It is stable at mu=6, kappa=3 otherwise it is instable
    """
    import numpy as np
    from pySDC.projects.Second_orderSDC.penningtrap_Simulation import Stability_implementation
    from pySDC.projects.Second_orderSDC.dampedharmonic_oscillator_run_stability import dampedharmonic_oscillator_params

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
