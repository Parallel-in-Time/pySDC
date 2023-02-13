import pytest


@pytest.mark.base
@pytest.mark.parametrize("imex", [True, False])
@pytest.mark.parametrize("direct_solver", [True, False])
def test_scipy_reference_solution(imex, direct_solver, plotting=False):
    """
    Test if the IMEX and fully implicit SDC schemes can match the solutions obtained by a scipy reference solution.
    Since the scipy solutions require many steps to accurately solve the problem explicitly, we divide the temporal
    domain in three stages and solve only short parts of it with scipy.
    First, the temperature is below the runaway threshold until about t=300, then it transitions to runaway heating
    until about t=400 and from then on the magnet heats up really quickly.

    We compare a single step of SDC to however many steps scipy uses for a time point in each of these sections and
    make sure a certain threshold is not exceeded.

    Args:
        imex (bool): Solve the problem IMEX or fully implicit
        plotting (bool): Plot the solution or not

    Returns:
        None
    """
    import numpy as np
    from pySDC.helpers.stats_helper import get_sorted
    from pySDC.implementations.hooks.log_errors import LogGlobalErrorPostRun, LogLocalErrorPostStep
    from pySDC.projects.Resilience.leaky_superconductor import run_leaky_superconductor, plot_solution, LogData
    from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity

    dt = 1e-2
    dt_max = {True: 2e1, False: np.inf}
    custom_description = {}
    custom_description['problem_params'] = {
        'newton_tol': 1e-4,
        'newton_iter': 15,
        'lintol': 1e-4,
        'liniter': 5,
        'direct_solver': direct_solver,
    }
    custom_controller_params = {'logger_level': 30}
    custom_description['convergence_controllers'] = {Adaptivity: {'e_tol': 1e-6, 'dt_max': dt_max[imex]}}
    custom_description['level_params'] = {'dt': dt}

    Tend = [150, 350, 420]
    Tend = np.sort([me + dt for me in Tend] + Tend)
    u0 = None
    t0 = None
    for i in range(len(Tend)):
        stats, controller, _ = run_leaky_superconductor(
            custom_description=custom_description,
            custom_controller_params=custom_controller_params,
            imex=imex,
            Tend=Tend[i],
            u0=u0,
            t0=t0,
            hook_class=LogLocalErrorPostStep if i % 2 == 1 else LogData,
        )

        u = get_sorted(stats, type='u', recomputed=False)
        u0 = u[-1][1]
        t0 = u[-1][0]
        if i % 2:
            assert (
                len(u) == 1
            ), f"Expected to solve the problem with a single step, but needed {len(u)} steps in phase {i}!"
            e = max([me[1] for me in get_sorted(stats, type='e_local_post_step')])
            thresh = 1e-5
            assert e < thresh, f"Error in phase {i} exceeds allowed threshold: e={e:.2e}>{thresh:.2e}!"
        elif plotting:  # pragma no cover
            plot_solution(stats, controller)


@pytest.mark.base
def test_imex_vs_fully_implicit_leaky_superconductor():
    """
    Test if the IMEX and fully implicit schemes get the same solution. This is a test that the global accuracy is ok.
    """
    from pySDC.projects.Resilience.leaky_superconductor import compare_imex_full

    compare_imex_full()


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    test_scipy_reference_solution(False, True)
    plt.show()
