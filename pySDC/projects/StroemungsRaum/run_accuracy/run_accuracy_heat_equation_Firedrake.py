"""
The temporal accuracy study of :mod:`pySDC.projects.StroemungsRaum.run_accuracy.run_accuracy_heat_equation_FEniCS`
with the Firedrake port, :mod:`pySDC.projects.StroemungsRaum.run_heat_equation_Firedrake`.
"""

import os
from pathlib import Path

import numpy as np

from pySDC.projects.StroemungsRaum.run_heat_equation_Firedrake import run_simulation, setup

DEFAULT_PATH = f'{os.path.dirname(__file__)}/../data/heat_equation/'


def run_accuracy(c_nvars=64, num_nodes=2):
    """
    Run the heat equation to t = 1 with dt = 0.2, 0.1, 0.05, 0.025 and fit the observed order of accuracy.

    Args:
        c_nvars (int): Number of squares per direction
        num_nodes (int): Number of collocation nodes

    Returns:
        dict of the relative error per step size, and the order from a line fit in log-log space
    """
    description, controller_params, t0 = setup()
    controller_params['logger_level'] = 30
    description['problem_params']['c_nvars'] = c_nvars
    description['sweeper_params']['num_nodes'] = num_nodes

    results = {}
    for dt in [0.2 / 2**p for p in range(4)]:
        description['level_params']['dt'] = dt
        results[dt] = run_simulation(description, controller_params, t0, 1.0)[2]

    dts = list(results)
    order, _ = np.polyfit(np.log(dts), np.log([results[dt] for dt in dts]), 1)
    return results, order


def plot_accuracy(results, p=3, path=DEFAULT_PATH):
    """
    Plot the errors against the step size with a guide line of order `p`, saved as a PNG in `path`.

    Args:
        results (dict): Relative error per step size, from :func:`run_accuracy`
        p (int): The expected order
        path (str): Where to save the figure
    """
    import matplotlib.pyplot as plt

    suffix = 'th' if 10 <= p % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(p % 10, 'th')
    dts = sorted(results)
    errors = [results[dt] for dt in dts]
    guide = [errors[-1] / (dts[-1] / dt) ** p for dt in dts]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.loglog(dts, guide, color='k', ls='--', label=f'{p}{suffix} order')
    ax.loglog(dts, errors, ls=' ', marker='o', markersize=10, label='experiment')
    ax.set(
        xlabel='dt', ylabel='rel. error', xlim=(min(dts) / 1.5, max(dts) * 1.5), ylim=(min(guide) / 2, max(guide) * 2)
    )
    ax.legend(loc=2)
    ax.grid(True, which='minor', ls='--', color='0.8')
    ax.grid(True, which='major', ls='-', color='0.001')

    Path(path).mkdir(parents=True, exist_ok=True)
    fig.savefig(f'{path}/heat_equation_{p}{suffix}_order_time_Firedrake.png', bbox_inches='tight')
    plt.close(fig)


def main():
    num_nodes = 2
    results, _ = run_accuracy(64, num_nodes)
    plot_accuracy(results, 2 * num_nodes - 1)


if __name__ == '__main__':
    main()
