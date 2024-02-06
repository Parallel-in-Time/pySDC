from pySDC.projects.Second_orderSDC.harmonic_oscillator_params import get_default_harmonic_oscillator_description
from pySDC.projects.Second_orderSDC.stability_simulation import StabilityImplementation


if __name__ == '__main__':
    """
    To implement Stability region for the Harmonic Oscillator problem
    Run for
        SDC stability region: model_stab.run_SDC_stability()
        Picard stability region: model_stab.run_Picard_stability()
        Runge-Kutta-Nzström stability region: model_run_RKN_stability()
    To implement spectral radius of iteration matrix
    Run:
        Iteration matrix of SDC method: model_stab.run_Ksdc()
        Iteration matrix of Picard method: model_stab.run_Kpicard()

    """
    # This code checks if the "data" folder exists or not.
    exec(open("check_data_folder.py").read())
    # Execute the stability analysis for the damped harmonic oscillator
    description = get_default_harmonic_oscillator_description()
    # =============================================================================
    #    maxiter can be changed here manually. By default, maxiter is 50
    description['step_params']['maxiter'] = 50
    # =============================================================================

    model_stab = StabilityImplementation(description, kappa_max=18, mu_max=18, Num_iter=(200, 200))

    model_stab.run_SDC_stability()
    model_stab.run_Picard_stability()
    model_stab.run_RKN_stability()
    model_stab.run_Ksdc()
    # model_stab.run_Kpicard()
