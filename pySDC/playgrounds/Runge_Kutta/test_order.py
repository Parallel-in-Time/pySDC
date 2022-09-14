import matplotlib.pyplot as plt
import numpy as np

from pySDC.projects.Resilience.accuracy_check import plot_orders
from pySDC.playgrounds.Preconditioners.dahlquist import run_dahlquist, plot_stability

from pySDC.projects.Resilience.advection import run_advection
from pySDC.projects.Resilience.vdp import run_vdp

from pySDC.implementations.sweeper_classes.Runge_Kutta import RK1, RK4, MidpointMethod, CrankNicholson


colors = {
    RK1: 'blue',
    MidpointMethod: 'red',
    RK4: 'orange',
    CrankNicholson: 'purple',
}


def plot_order(sweeper, prob, dt_list, description=None, ax=None, Tend_fixed=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    description = dict() if description is None else description
    description['sweeper_class'] = sweeper
    description['sweeper_params'] = {'implicit': True}

    # determine the order
    plot_orders(ax, [1], True, Tend_fixed=Tend_fixed, custom_description=description, dt_list=dt_list, prob=prob)

    # check if we got the expected order for the local error
    orders = {
        RK1: 2,
        MidpointMethod: 3,
        RK4: 5,
        CrankNicholson: 3,
    }
    numerical_order = float(ax.get_lines()[-1].get_label()[7:])
    expected_order = orders.get(sweeper, numerical_order)
    assert np.isclose(numerical_order, expected_order, atol=2.5e-1),\
        f"Expected order {expected_order}, got {numerical_order}!"

    # decorate
    ax.get_lines()[-1].set_color(colors.get(sweeper, 'black'))

    label = f'{sweeper.__name__} - {ax.get_lines()[-1].get_label()[5:]}'
    ax.get_lines()[-1].set_label(label)
    ax.legend(frameon=False)


def plot_stability_single(sweeper, ax=None, description=None, implicit=True):
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    description = dict() if description is None else description
    description['sweeper_class'] = sweeper
    description['sweeper_params'] = {'implicit': implicit}

    stats, _, _ = run_dahlquist(custom_description=description)
    plot_stability(stats, ax=ax, iter=[1], colors=[colors.get(sweeper, 'black')])

    ax.get_lines()[-3].set_label(sweeper.__name__)
    ax.legend(frameon=False)


def plot_all_stability():
    fig, axs = plt.subplots(1, 2, figsize=(11, 5))

    impl = [True, False]
    sweepers = [[RK1, MidpointMethod, CrankNicholson], [RK1, MidpointMethod, RK4]]
    titles = ['implicit', 'explicit']

    for j in range(len(impl)):
        for i in range(len(sweepers[j])):
            plot_stability_single(sweepers[j][i], implicit=impl[j], ax=axs[j])
        axs[j].set_title(titles[j])

    axs[0].set_xlim([-4, 4])
    axs[0].set_ylim([-4, 4])
    fig.tight_layout()


def plot_all_orders(prob, dt_list, Tend, sweepers):
    fig, ax = plt.subplots(1, 1)
    for i in range(len(sweepers)):
        plot_order(sweepers[i], prob, dt_list, Tend_fixed=Tend, ax=ax)


def test_vdp():
    Tend = 5e-2
    plot_all_orders(run_vdp, Tend * 2.**(-np.arange(8)), Tend, [RK1, MidpointMethod, CrankNicholson, RK4])


def test_advection():
    plot_all_orders(run_advection, 1.e-3 * 2.**(-np.arange(8)), None, [RK1, MidpointMethod, CrankNicholson])


if __name__ == '__main__':
    test_vdp()
    test_advection()
    plot_all_stability()
    plt.show()
