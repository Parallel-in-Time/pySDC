import matplotlib.pyplot as plt
import numpy as np
import pytest

from pySDC.implementations.sweeper_classes.Runge_Kutta import (
    RK1,
    RK4,
    MidpointMethod,
    CrankNicholson,
    Cash_Karp,
    Heun_Euler,
)
from pySDC.implementations.convergence_controller_classes.adaptivity import AdaptivityRK
from pySDC.implementations.convergence_controller_classes.estimate_embedded_error import EstimateEmbeddedErrorNonMPI
from pySDC.helpers.stats_helper import get_sorted


colors = {
    RK1: 'blue',
    MidpointMethod: 'red',
    RK4: 'orange',
    CrankNicholson: 'purple',
    Cash_Karp: 'teal',
}


def plot_order(sweeper, prob, dt_list, description=None, ax=None, Tend_fixed=None, implicit=True):
    """
    Make a plot of the order of the scheme and test if it has the correct order

    Args:
        sweeper (pySDC.Sweeper.RungeKutta): The RK rule to try
        prob (function): Some function that runs a pySDC problem and accepts suitable parameters, see resilience project
        dt_list (list): List of step sizes to try
        description (dict): A description to use for running the problem
        ax: Somewhere to plot
        Tend_fixed (float): Time to integrate to with each step size
        implicit (bool): Whether to use implicit or explicit versions of RK rules

    Returns:
        None
    """
    from pySDC.projects.Resilience.accuracy_check import plot_orders, plot_all_errors

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    description = dict() if description is None else description
    description['sweeper_class'] = sweeper
    description['sweeper_params'] = {'implicit': implicit}
    description['step_params'] = {'maxiter': 1}

    custom_controller_params = {'logger_level': 40}

    # determine the order
    plot_orders(
        ax,
        [1],
        True,
        Tend_fixed=Tend_fixed,
        custom_description=description,
        dt_list=dt_list,
        prob=prob,
        custom_controller_params=custom_controller_params,
    )

    # check if we got the expected order for the local error
    orders = {
        RK1: 2,
        MidpointMethod: 3,
        RK4: 5,
        CrankNicholson: 3,
        Cash_Karp: 6,
    }
    numerical_order = float(ax.get_lines()[-1].get_label()[7:])
    expected_order = orders.get(sweeper, numerical_order)
    assert np.isclose(
        numerical_order, expected_order, atol=2.6e-1
    ), f"Expected order {expected_order}, got {numerical_order}!"

    # decorate
    ax.get_lines()[-1].set_color(colors.get(sweeper, 'black'))

    label = f'{sweeper.__name__} - {ax.get_lines()[-1].get_label()[5:]}'
    ax.get_lines()[-1].set_label(label)
    ax.legend(frameon=False)


def plot_stability_single(sweeper, ax=None, description=None, implicit=True, re=None, im=None, crosshair=True):
    """
    Plot the domain of stability for a single RK rule.

    Args:
        sweeper (pySDC.Sweeper.RungeKutta)
        ax: Somewhere to plot
        description (dict): A description to use for running the problem
        implicit (bool): Whether to use implicit or explicit versions of RK rules
        re (numpy.ndarray): A range of values for the real axis
        im (numpy.ndarray): A range of values for the imaginary axis
        crosshair (bool): Whether to emphasize the axes

    Returns:
        None
    """
    from pySDC.projects.Resilience.dahlquist import run_dahlquist, plot_stability

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    description = dict() if description is None else description
    description['sweeper_class'] = sweeper
    description['sweeper_params'] = {'implicit': implicit}
    description['step_params'] = {'maxiter': 1}

    custom_controller_params = {'logger_level': 40}

    re = np.linspace(-30, 30, 400) if re is None else re
    im = np.linspace(-50, 50, 400) if im is None else im
    lambdas = np.array([[complex(re[i], im[j]) for i in range(len(re))] for j in range(len(im))]).reshape(
        (len(re) * len(im))
    )
    custom_problem_params = {'lambdas': lambdas}

    stats, _, _ = run_dahlquist(
        custom_description=description,
        custom_problem_params=custom_problem_params,
        custom_controller_params=custom_controller_params,
    )
    plot_stability(stats, ax=ax, iter=[1], colors=[colors.get(sweeper, 'black')], crosshair=crosshair, fill=True)

    ax.get_lines()[-1].set_label(sweeper.__name__)
    ax.legend(frameon=False)


@pytest.mark.skip(reason="This test can only be evaluated optically by humans so far")
def plot_all_stability():
    """
    Make a figure showing domains of stability for a range of RK rules, both implicit and explicit.

    Returns:
        None
    """
    fig, axs = plt.subplots(1, 2, figsize=(11, 5))

    impl = [True, False]
    sweepers = [[RK1, MidpointMethod, CrankNicholson], [RK1, MidpointMethod, RK4, Cash_Karp]]
    titles = ['implicit', 'explicit']
    re = np.linspace(-4, 4, 400)
    im = np.linspace(-4, 4, 400)
    crosshair = [True, False, False, False]

    for j in range(len(impl)):
        for i in range(len(sweepers[j])):
            plot_stability_single(sweepers[j][i], implicit=impl[j], ax=axs[j], re=re, im=im, crosshair=crosshair[i])
        axs[j].set_title(titles[j])

    fig.tight_layout()


def plot_all_orders(prob, dt_list, Tend, sweepers, implicit=True):
    """
    Make a plot with various sweepers and check their order.

    Args:
        prob (function): Some function that runs a pySDC problem and accepts suitable parameters, see resilience project
        dt_list (list): List of step sizes to try
        Tend (float): Time to solve to with each step size
        sweepers (list): List of RK rules to try
        implicit (bool): Whether to use implicit or explicit versions of RK rules

    Returns:
        None
    """
    fig, ax = plt.subplots(1, 1)
    for i in range(len(sweepers)):
        plot_order(sweepers[i], prob, dt_list, Tend_fixed=Tend, ax=ax, implicit=implicit)


@pytest.mark.base
def test_vdp():
    """
    Here, we check the order in time for various implicit RK rules with the van der Pol problem.
    This is interesting, because van der Pol is non-linear.

    Returns:
        None
    """
    from pySDC.projects.Resilience.vdp import run_vdp, plot_step_sizes

    Tend = 7e-2
    plot_all_orders(run_vdp, Tend * 2.0 ** (-np.arange(8)), Tend, [RK1, MidpointMethod, CrankNicholson, RK4, Cash_Karp])


@pytest.mark.base
def test_advection():
    """
    Here, we check the order in time for various implicit and explicit RK rules with an advection problem.

    Returns:
        None
    """
    from pySDC.projects.Resilience.advection import run_advection

    plot_all_orders(
        run_advection, 1.0e-3 * 2.0 ** (-np.arange(8)), None, [RK1, MidpointMethod, CrankNicholson], implicit=True
    )
    plot_all_orders(run_advection, 1.0e-3 * 2.0 ** (-np.arange(8)), None, [RK1, MidpointMethod], implicit=False)


@pytest.mark.base
@pytest.mark.parametrize("sweeper", [Cash_Karp, Heun_Euler])
def test_embedded_estimate_order(sweeper):
    """
    Test the order of embedded Runge-Kutta schemes. They are not run with adaptivity here,
    so we can simply vary the step size and check the embedded error estimate.

    Args:
        sweeper (pySDC.Sweeper.RungeKutta)

    Returns:
        None
    """
    fig, ax = plt.subplots(1, 1)

    # change only the things in the description that we need for adaptivity
    convergence_controllers = dict()
    convergence_controllers[EstimateEmbeddedErrorNonMPI] = {}

    description = dict()
    description['convergence_controllers'] = convergence_controllers
    description['sweeper_class'] = sweeper
    description['step_params'] = {'maxiter': 1}

    custom_controller_params = {'logger_level': 40}

    Tend = 7e-2
    dt_list = Tend * 2.0 ** (-np.arange(8))
    prob = run_vdp
    plot_all_errors(
        ax,
        [5] if sweeper == Cash_Karp else [2],
        True,
        Tend_fixed=Tend,
        custom_description=description,
        dt_list=dt_list,
        prob=prob,
        custom_controller_params=custom_controller_params,
    )


@pytest.mark.base
def test_embedded_method():
    """
    Here, we test if Cash Karp's method gives a hard-coded result and number of restarts when running with adaptivity.

    Returns:
        None
    """
    sweeper = Cash_Karp
    fig, ax = plt.subplots(1, 1)

    # change only the things in the description that we need for adaptivity
    adaptivity_params = dict()
    adaptivity_params['e_tol'] = 1e-7
    adaptivity_params['update_order'] = 5

    convergence_controllers = dict()
    convergence_controllers[AdaptivityRK] = adaptivity_params

    description = dict()
    description['convergence_controllers'] = convergence_controllers
    description['sweeper_class'] = sweeper
    description['step_params'] = {'maxiter': 1}

    custom_controller_params = {'logger_level': 40}

    stats, _, _ = run_vdp(description, 1, custom_controller_params=custom_controller_params)
    plot_step_sizes(stats, ax)

    fig.tight_layout()

    dt_last = get_sorted(stats, type='dt')[-2][1]
    restarts = sum([me[1] for me in get_sorted(stats, type='restart')])
    assert np.isclose(dt_last, 0.14175080252629996), "Cash-Karp has computed a different last step size than before!"
    assert restarts == 17, "Cash-Karp has restarted a different number of times than before"


if __name__ == '__main__':
    test_embedded_method()
    test_embedded_estimate_order(Cash_Karp)
    test_vdp()
    test_advection()
    plot_all_stability()
    plt.show()
