from pySDC.projects.Second_orderSDC.harmonic_oscillator_params import harmonic_oscillator_params
from pySDC.projects.Second_orderSDC.stability_simulation import StabilityImplementation


if __name__ == '__main__':
    """
    Damped harmonic oscillator as a test problem for the stability plot:
        x' = v
        v' = -kappa * x - mu * v
        kappa: spring constant
        mu: friction
        Source: https://beltoforion.de/en/harmonic_oscillator/
    """
    # Execute the stability analysis for the damped harmonic oscillator
    description = harmonic_oscillator_params()
    model_stab = StabilityImplementation(description, kappa_max=30, mu_max=30, Num_iter=(200, 200))

    model_stab.run_SDC_stability()
    model_stab.run_Picard_stability()
    model_stab.run_RKN_stability()
    model_stab.run_Ksdc()
    # model_stab.run_Kpicard
