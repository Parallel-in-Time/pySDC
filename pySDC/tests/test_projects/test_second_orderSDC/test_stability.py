import pytest


@pytest.mark.base
def test_stability():
    """
    Stability domain test
    only the values of mu=[6, 20] and kappa=[3, 20]
    It is stable at mu=6, kappa=3 otherwise it is instable
    """
    import numpy as np
    from pySDC.projects.Second_orderSDC.stability_simulation import StabilityImplementation
    from pySDC.projects.Second_orderSDC.harmonic_oscillator_params import harmonic_oscillator_params

    description = harmonic_oscillator_params()
    stability = StabilityImplementation(description, kappa_max=14, mu_max=14, Num_iter=(2, 2))
    stability.lambda_kappa = np.array([6, 20])
    stability.lambda_mu = np.array([3, 20])
    SDC, KSDC, *_ = stability.stability_data()
    assert (
        SDC[0, 0] <= 1
    ), f'The SDC method is instable at mu={stability.lambda_mu[0]} and kappa={stability.lambda_kappa[0]}'
    assert (
        SDC[-1, -1] > 1
    ), f'The SDC method is stable at mu={stability.lambda_mu[-1]} and kappa={stability.lambda_kappa[-1]}'


@pytest.mark.base
def test_RKN_stability():
    """
    Stability domain test
    only the values of mu=[1, 20] and kappa=[1, 20]
    It is stable at mu=6, kappa=3 otherwise it is instable
    """
    import numpy as np
    from pySDC.projects.Second_orderSDC.stability_simulation import StabilityImplementation
    from pySDC.projects.Second_orderSDC.harmonic_oscillator_params import harmonic_oscillator_params

    description = harmonic_oscillator_params()
    stability = StabilityImplementation(description, kappa_max=14, mu_max=14, Num_iter=(2, 2))
    stability.lambda_kappa = np.array([1, 20])
    stability.lambda_mu = np.array([1, 20])
    stab_RKN = stability.stability_data_RKN()
    assert (
        stab_RKN[0, 0] <= 1
    ), f'The RKN method is instable at mu={stability.lambda_mu[0]} and kappa={stability.lambda_kappa[0]}'
    assert (
        stab_RKN[-1, -1] > 1
    ), f'The RKN method is stable at mu={stability.lambda_mu[-1]} and kappa={stability.lambda_kappa[-1]}'


if __name__ == '__main__':
    pass
