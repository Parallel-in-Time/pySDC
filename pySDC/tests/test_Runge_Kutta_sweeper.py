import pytest

COLORS = {
    'RK1': 'blue',
    'MidpointMethod': 'red',
    'RK4': 'orange',
    'CrankNicholson': 'purple',
    'Cash_Karp': 'teal',
}

EXPLICIT_METHODS = ['RK1', 'MidpointMethod', 'CrankNicholson']
IMPLICIT_METHODS = ['RK1', 'MidpointMethod', 'RK4', 'Cash_Karp']
EMBEDDED_METHODS = ['Cash_Karp', 'Heun_Euler', 'DIRK34']


def get_sweeper(sweeper_name):
    """
    Retrieve a sweeper from a name

    Args:
        sweeper_name (str):

    Returns:
        pySDC.Sweeper.RungeKutta: The sweeper
    """
    import pySDC.implementations.sweeper_classes.Runge_Kutta as RK

    return eval(f'RK.{sweeper_name}')


def plot_order(sweeper_name, prob, dt_list, description=None, ax=None, Tend_fixed=None, implicit=True):
    """
    Make a plot of the order of the scheme and test if it has the correct order

    Args:
        sweeper_name (str): Name of the RK rule you want
        prob (function): Some function that runs a pySDC problem and accepts suitable parameters, see resilience project
        dt_list (list): List of step sizes to try
        description (dict): A description to use for running the problem
        ax: Somewhere to plot
        Tend_fixed (float): Time to integrate to with each step size
        implicit (bool): Whether to use implicit or explicit versions of RK rules

    Returns:
        None
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from pySDC.projects.Resilience.accuracy_check import plot_orders

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    description = {} if description is None else description
    description['sweeper_class'] = get_sweeper(sweeper_name)
    description['sweeper_params'] = {'implicit': implicit}
    description['step_params'] = {'maxiter': 1}
    description['level_params'] = {'restol': +1}

    custom_controller_params = {'logger_level': 30}

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
        embedded_error_flavor='standard',
    )

    # check if we got the expected order for the local error
    orders = {
        'RK1': 2,
        'MidpointMethod': 3,
        'RK4': 5,
        'CrankNicholson': 3,
        'Cash_Karp': 6,
    }
    numerical_order = float(ax.get_lines()[-1].get_label()[7:])
    expected_order = orders.get(sweeper_name, numerical_order)
    assert np.isclose(
        numerical_order, expected_order, atol=2.6e-1
    ), f"Expected order {expected_order}, got {numerical_order}!"

    # decorate
    ax.get_lines()[-1].set_color(COLORS.get(sweeper_name, 'black'))

    label = f'{sweeper_name} - {ax.get_lines()[-1].get_label()[5:]}'
    ax.get_lines()[-1].set_label(label)
    ax.legend(frameon=False)


def plot_stability_single(sweeper_name, ax=None, description=None, implicit=True, re=None, im=None, crosshair=True):
    """
    Plot the domain of stability for a single RK rule.

    Args:
        sweeper_name (pySDC.Sweeper.RungeKutta)
        ax: Somewhere to plot
        description (dict): A description to use for running the problem
        implicit (bool): Whether to use implicit or explicit versions of RK rules
        re (numpy.ndarray): A range of values for the real axis
        im (numpy.ndarray): A range of values for the imaginary axis
        crosshair (bool): Whether to emphasize the axes

    Returns:
        None
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from pySDC.projects.Resilience.dahlquist import run_dahlquist, plot_stability

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    description = {} if description is None else description
    description['sweeper_class'] = get_sweeper(sweeper_name)
    description['sweeper_params'] = {'implicit': implicit}
    description['step_params'] = {'maxiter': 1}

    custom_controller_params = {'logger_level': 30}

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
    Astable = plot_stability(
        stats, ax=ax, iter=[1], COLORS=[COLORS.get(sweeper_name, 'black')], crosshair=crosshair, fill=True
    )

    # check if we think the method should be A-stable
    Astable_methods = ['RK1', 'CrankNicholson', 'MidpointMethod', 'DIRK34']  # only the implicit versions are A-stable
    assert (
        implicit and sweeper_name in Astable_methods
    ) == Astable, f"Unexpected region of stability for {sweeper_name} sweeper!"

    ax.get_lines()[-1].set_label(sweeper_name)
    ax.legend(frameon=False)


def plot_all_stability():
    """
    Make a figure showing domains of stability for a range of RK rules, both implicit and explicit.

    Returns:
        None
    """
    import numpy as np
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, figsize=(11, 5))

    impl = [True, False]
    sweepers = [['RK1', 'MidpointMethod', 'CrankNicholson'], ['RK1', 'MidpointMethod', 'RK4', 'Cash_Karp']]
    titles = ['implicit', 'explicit']
    re = np.linspace(-4, 4, 400)
    im = np.linspace(-4, 4, 400)
    crosshair = [True, False, False, False]

    for j in range(len(impl)):
        for i in range(len(sweepers[j])):
            plot_stability_single(sweepers[j][i], implicit=impl[j], ax=axs[j], re=re, im=im, crosshair=crosshair[i])
        axs[j].set_title(titles[j])

    plot_stability_single('DIRK34', re=re, im=im)

    fig.tight_layout()


@pytest.mark.base
@pytest.mark.parametrize("sweeper_name", EXPLICIT_METHODS)
def test_stability_implicit_methods(sweeper_name):
    plot_stability_single(sweeper_name, implicit=True)


@pytest.mark.base
@pytest.mark.parametrize("sweeper_name", IMPLICIT_METHODS)
def test_stability_explicit_methods(sweeper_name):
    plot_stability_single(sweeper_name, implicit=False)


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
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1)
    for i in range(len(sweepers)):
        plot_order(sweepers[i], prob, dt_list, Tend_fixed=Tend, ax=ax, implicit=implicit)


@pytest.mark.base
@pytest.mark.parametrize("sweeper_name", ['RK1', 'MidpointMethod', 'CrankNicholson', 'RK4', 'Cash_Karp'])
def test_vdp(sweeper_name):
    """
    Here, we check the order in time for various implicit RK rules with the van der Pol problem.
    This is interesting, because van der Pol is non-linear.

    Returns:
        None
    """
    import numpy as np
    from pySDC.projects.Resilience.vdp import run_vdp

    Tend = 7e-2
    plot_order(sweeper_name, prob=run_vdp, dt_list=Tend * 2.0 ** (-np.arange(8)), implicit=True, Tend_fixed=Tend)


@pytest.mark.base
@pytest.mark.parametrize("sweeper_name", ['RK1', 'MidpointMethod'])
def test_advection(sweeper_name):
    """
    Test the order for some explicit RK rules

    Returns:
        None
    """
    from pySDC.projects.Resilience.advection import run_advection
    import numpy as np

    plot_order(
        sweeper_name=sweeper_name,
        prob=run_advection,
        dt_list=1.0e-3 * 2.0 ** (-np.arange(8)),
        implicit=False,
        Tend_fixed=None,
    )


@pytest.mark.base
@pytest.mark.parametrize('sweeper_name', EMBEDDED_METHODS)
def test_embedded_estimate_order(sweeper_name):
    """
    Test the order of embedded Runge-Kutta schemes. They are not run with adaptivity here,
    so we can simply vary the step size and check the embedded error estimate.

    Args:
        sweeper_name (pySDC.Sweeper.RungeKutta)

    Returns:
        None
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from pySDC.projects.Resilience.vdp import run_vdp
    from pySDC.projects.Resilience.accuracy_check import plot_all_errors
    from pySDC.implementations.convergence_controller_classes.estimate_embedded_error import EstimateEmbeddedError

    fig, ax = plt.subplots(1, 1)

    # change only the things in the description that we need for adaptivity
    convergence_controllers = {}
    convergence_controllers[EstimateEmbeddedError.get_implementation('standard')] = {}

    description = {}
    description['convergence_controllers'] = convergence_controllers
    description['sweeper_class'] = get_sweeper(sweeper_name)
    description['step_params'] = {'maxiter': 1}

    custom_controller_params = {'logger_level': 30}

    Tend = 7e-2
    dt_list = Tend * 2.0 ** (-np.arange(8))
    prob = run_vdp
    plot_all_errors(
        ax,
        [get_sweeper(sweeper_name).get_update_order()],
        True,
        Tend_fixed=Tend,
        custom_description=description,
        dt_list=dt_list,
        prob=prob,
        custom_controller_params=custom_controller_params,
        embedded_error_flavor='standard',
        keys=['e', 'e_embedded'],
    )


@pytest.mark.base
def test_embedded_method():
    """
    Here, we test if Cash Karp's method gives a hard-coded result and number of restarts when running with adaptivity.

    Returns:
        None
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from pySDC.projects.Resilience.vdp import run_vdp, plot_step_sizes
    from pySDC.implementations.convergence_controller_classes.adaptivity import AdaptivityRK
    from pySDC.helpers.stats_helper import get_sorted

    sweeper_name = 'Cash_Karp'
    fig, ax = plt.subplots(1, 1)

    # change only the things in the description that we need for adaptivity
    adaptivity_params = {}
    adaptivity_params['e_tol'] = 1e-7
    adaptivity_params['update_order'] = 5

    convergence_controllers = {}
    convergence_controllers[AdaptivityRK] = adaptivity_params

    description = {}
    description['convergence_controllers'] = convergence_controllers
    description['sweeper_class'] = get_sweeper(sweeper_name)
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
    # make plots
    import matplotlib.pyplot as plt
    import numpy as np
    from pySDC.projects.Resilience.vdp import run_vdp
    from pySDC.projects.Resilience.advection import run_advection

    plot_all_orders(
        run_vdp, 7e-2 * 2.0 ** (-np.arange(8)), 7e-2, ['RK1', 'MidpointMethod', 'CrankNicholson', 'RK4', 'Cash_Karp']
    )

    plot_all_orders(
        run_advection, 1.0e-3 * 2.0 ** (-np.arange(8)), None, ['RK1', 'MidpointMethod', 'CrankNicholson'], implicit=True
    )
    plot_all_orders(run_advection, 1.0e-3 * 2.0 ** (-np.arange(8)), None, ['RK1', 'MidpointMethod'], implicit=False)

    test_embedded_method()
    for sweep in EMBEDDED_METHODS:
        test_embedded_estimate_order(sweep)
    plot_all_stability()

    plt.show()
