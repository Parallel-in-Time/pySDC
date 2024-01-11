import pytest


@pytest.mark.base
def test_stability_SDC():
    """
    Stability domain test
    only the values of mu=[6, 20] and kappa=[3, 20]
    It is stable at mu=6, kappa=3 otherwise it is instable
    """
    import numpy as np
    from pySDC.projects.Second_orderSDC.stability_simulation import check_points_and_interval
    from pySDC.projects.Second_orderSDC.harmonic_oscillator_params import get_default_harmonic_oscillator_description

    description = get_default_harmonic_oscillator_description()
    # Additonal params to compute stability points
    helper_params = {
        'quad_type_list': ('GAUSS',),
        'Num_iter': (2, 2),
        'num_nodes_list': np.arange(3, 4, 1),
        'max_iter_list': np.arange(5, 6, 1),
    }

    points = ((3, 6), (20, 20))
    # Iterate through points and perform stability check
    point0 = check_points_and_interval(description, helper_params, points[0], check_stability_point=True)
    point1 = check_points_and_interval(description, helper_params, points[1], check_stability_point=True)

    assert point0[-1][-1] <= 1, f'The SDC method is instable at mu={points[0][1]} and kappa={points[0][0]}'
    assert point1[-1][-1] > 1, f'The SDC method is stable at mu={points[1][1]} and kappa={points[1][0]}'


@pytest.mark.base
def test_RKN_stability():
    """
    Stability domain test for RKN
    only the values of mu=[1, 20] and kappa=[1, 20]
    It is stable at mu=6, kappa=3 otherwise it is instable
    """
    import numpy as np
    from pySDC.projects.Second_orderSDC.stability_simulation import StabilityImplementation
    from pySDC.projects.Second_orderSDC.harmonic_oscillator_params import get_default_harmonic_oscillator_description

    description = get_default_harmonic_oscillator_description()
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
