from pathlib import Path
import numpy as np
from pySDC.projects.StroemungsRaum.run_heat_equation_FEniCS import setup, run_simulation
import matplotlib.pyplot as plt


def run_accuracy(c_nvars=64, num_nodes=2):
    """
    Routine to run the accuracy test for the heat equation with FEniCS. It runs the simulation
    for different dt values, collects the errors, computes, and plots the observed order of accuracy.

    Args:
    -----
    c_nvars: int,
        spatial resolution (number of degrees of freedom in space)
    num_nodes: int,
        number of collocation nodes in time

    Returns
    -------
    results: dict
        Dictionary containing the errors for each dt, as well as the list of dts for easier access.
    order: float
        The observed order of accuracy, obtained by fitting a line in the log-log space to the errors and dts.
    """
    # set parameters
    Tend = 1.0
    t0 = 0.0

    description, controller_params = setup(t0=t0)
    description['problem_params']['c_nvars'] = c_nvars
    description['sweeper_params']['num_nodes'] = num_nodes

    # assemble list of dt
    dt_list = [0.2 / 2**p for p in range(0, 4)]

    results = {}
    # loop over all dt values
    for dt in dt_list:

        # update dt in description for this run
        description['level_params']['dt'] = dt

        # run the simulation and get the error
        _, _, err = run_simulation(description, controller_params, Tend)

        # store the error in the results dictionary
        results[dt] = err

    # errors in the correct order
    err_list = [results[dt] for dt in results.keys()]

    # global slope (fit in log-log)
    order, _ = np.polyfit(np.log(dt_list), np.log(err_list), 1)

    return results, order


def plot_accuracy(results, p=3):  # pragma: no cover
    """
    Routine to visualize the errors as well as the expected errors

    Args:
    -----
    results: dict
        The dictionary containing the errors for each dt, as well as the list of dts for easier access.
    p: int
        The expected order of accuracy
    """

    # get suffix for order (e.g. 1st, 2nd, 3rd, 4th, etc.)
    suffix = "th" if 10 <= p % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(p % 10, "th")

    # get the list of dt and errors from the results dictionary
    dt_list = sorted(results.keys())
    err_list = [results[dt] for dt in dt_list]

    # Set up plotting parameters
    params = {
        'legend.fontsize': 20,
        'figure.figsize': (12, 8),
        'axes.labelsize': 20,
        'axes.titlesize': 20,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'lines.linewidth': 3,
    }
    plt.rcParams.update(params)

    # create new figure
    plt.figure()
    # take x-axis limits from dt_list + some spacning left and right
    plt.xlim([min(dt_list) / 1.5, max(dt_list) * 1.5])
    plt.xlabel('dt')
    plt.ylabel('abs. error')
    plt.grid()

    # assemble optimal errors for 3rd order method and plot
    order_guide = [err_list[np.argmax(dt_list)] / (dt_list[np.argmax(dt_list)] / dt) ** p for dt in dt_list]

    plt.loglog(dt_list, order_guide, color='k', ls='--', label=f"{p}{suffix} order")
    plt.loglog(dt_list, err_list, ls=' ', marker='o', markersize=10, label='experiment')

    # adjust y-axis limits, add legend
    plt.ylim([min(order_guide) / 2, max(order_guide) * 2])
    plt.legend(loc=2, ncol=1, numpoints=1)

    plt.grid(True, which="minor", ls="--", color='0.8')
    plt.grid(True, which="major", ls="-", color='0.001')

    # get the data directory
    import os

    path = f"{os.path.dirname(__file__)}/../data/heat_equation/"

    # if it does not exist, create the 'data' directory at the specified path, including any necessary parent directories
    Path(path).mkdir(parents=True, exist_ok=True)
    fname = path + f'heat_equation_{p}{suffix}_order_time_FEniCS.png'
    plt.savefig(fname, bbox_inches='tight')

    return None


def main():

    # parameters for 3rd order accuracy test
    c_nvars = 64  # spatial resolution
    num_nodes = 2  # number of collocation nodes in time
    p = 2 * num_nodes - 1  # expected order of accuracy

    # run the simulation and get the results
    results, _ = run_accuracy(c_nvars, num_nodes)
    plot_accuracy(results, p)


if __name__ == "__main__":
    main()
