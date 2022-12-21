import matplotlib.pyplot as plt
import numpy as np

from pySDC.projects.Resilience.accuracy_check import plot_orders, plot_all_errors
from pySDC.projects.Resilience.dahlquist import run_dahlquist, plot_stability

from pySDC.projects.Resilience.advection import run_advection
from pySDC.projects.Resilience.vdp import run_vdp, plot_step_sizes

from pySDC.implementations.sweeper_classes.Runge_Kutta import RK1, RK4, MidpointMethod, CrankNicholson, Cash_Karp
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


def plot_all_stability():
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
    fig, ax = plt.subplots(1, 1)
    for i in range(len(sweepers)):
        plot_order(sweepers[i], prob, dt_list, Tend_fixed=Tend, ax=ax, implicit=implicit)


def test_vdp():
    Tend = 7e-2
    plot_all_orders(run_vdp, Tend * 2.0 ** (-np.arange(8)), Tend, [RK1, MidpointMethod, CrankNicholson, RK4, Cash_Karp])


def test_advection():
    plot_all_orders(
        run_advection, 1.0e-3 * 2.0 ** (-np.arange(8)), None, [RK1, MidpointMethod, CrankNicholson], implicit=True
    )
    plot_all_orders(run_advection, 1.0e-3 * 2.0 ** (-np.arange(8)), None, [RK1, MidpointMethod], implicit=False)


def test_embedded_estimate_order():
    sweeper = Cash_Karp
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
        [5],
        True,
        Tend_fixed=Tend,
        custom_description=description,
        dt_list=dt_list,
        prob=prob,
        custom_controller_params=custom_controller_params,
    )


def test_embedded_method():
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
    test_embedded_estimate_order()
    test_vdp()
    test_advection()
    plot_all_stability()
    plt.show()
