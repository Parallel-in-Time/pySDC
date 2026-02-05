from pathlib import Path
import dolfin as df
import numpy as np
import matplotlib

from collections import namedtuple
from pySDC.projects.StroemungsRaum.problem_classes.HeatEquation_2D_FEniCS import (
    fenics_heat2D_mass,
    fenics_heat2D_mass_timebc,
    fenics_heat2D,
)
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.sweeper_classes.imex_1st_order_mass import imex_1st_order_mass
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.transfer_classes.BaseTransfer_mass import base_transfer_mass
from pySDC.implementations.transfer_classes.TransferFenicsMesh import mesh_to_mesh_fenics
from pySDC.helpers.stats_helper import get_sorted
from pySDC.implementations.hooks.log_solution import LogSolution

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D


def setup(t0=None):
    """
    Helper routine to set up parameters

    Args:
        t0 (float): initial time
        ml (bool): use single or multiple levels

    Returns:
        description and controller_params parameter dictionaries
    """

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-12
    # level_params['dt'] = 0.2

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 20

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = [2]

    problem_params = dict()
    problem_params['nu'] = 0.1
    problem_params['t0'] = t0  # ugly, but necessary to set up this ProblemClass
    problem_params['c_nvars'] = [64]
    problem_params['family'] = 'CG'
    problem_params['order'] = [2]
    problem_params['refinements'] = [1]

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 20
    controller_params['hook_class'] = LogSolution

    # base_transfer_params = dict()
    # base_transfer_params['finter'] = True

    # Fill description dictionary for easy hierarchy creation
    description = dict()

    description['problem_params'] = problem_params
    description['problem_class'] = fenics_heat2D_mass_timebc
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params
    description['sweeper_class'] = imex_1st_order_mass
    description['space_transfer_class'] = mesh_to_mesh_fenics
    # description['base_transfer_params'] = base_transfer_params
    description['base_transfer_class'] = base_transfer_mass

    return description, controller_params


def run_accuracy():

    # setup id for gathering the results (will sort by dt)
    ID = namedtuple('ID', 'dt')

    Tend = 1.0
    t0 = 0.0

    description, controller_params = setup(t0=t0)

    # assemble list of dt
    dt_list = [0.2 / 2**p for p in range(0, 4)]

    results = {}
    # loop over all nvars
    for dt in dt_list:
        description['level_params']['dt'] = dt
        # print(description['level_params'])

        controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

        # get initial values on finest level
        P = controller.MS[0].levels[0].prob
        uinit = P.u_exact(0.0)

        # call main function to get things done...
        uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

        # compute exact solution and compare
        uex = P.u_exact(Tend)
        err = abs(uex - uend) / abs(uex)

        # get id for this dt and store error in results
        id = ID(dt=dt)
        results[id] = err

    # add list of dt to results for easier access
    results['dt_list'] = dt_list
    print(results)
    return results


def plot_accuracy(results):
    """
    Routine to visualize the errors as well as the expected errors

    Args:
        results: the dictionary containing the errors
    """

    # setup id for gathering the results (will sort by dt)
    ID = namedtuple('ID', 'dt')

    # retrieve the list of nvars from results
    assert 'dt_list' in results, 'ERROR: expecting the list of dts in the results dictionary'
    dt_list = sorted(results['dt_list'])

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
    # take x-axis limits from nvars_list + some spacning left and right
    plt.xlim([min(dt_list) / 1.5, max(dt_list) * 1.5])
    plt.xlabel('dt')
    plt.ylabel('abs. error')
    plt.grid()

    # get guide for the order of accuracy, i.e. the errors to expect
    # get error for last entry in nvars_list
    id = ID(dt=dt_list[-1])
    base_error = results[id]

    # assemble optimal errors for 5th order method and plot
    order_guide_space = [base_error / (2 ** (3 * i)) for i in range(0, len(dt_list))]
    order_guide_space = sorted(order_guide_space)
    plt.loglog(dt_list, order_guide_space, color='k', ls='--', label='3rd order')

    min_err = 1e99
    max_err = 0e00
    err_list = []
    # loop over nvars, get errors and find min/max error for y-axis limits
    for dt in dt_list:
        id = ID(dt=dt)
        err = results[id]
        # min_err = min(err, min_err)
        # max_err = max(err, max_err)
        err_list.append(err)
    plt.loglog(dt_list, err_list, ls=' ', marker='o', markersize=10, label='experiment')

    min_err = min(order_guide_space)
    max_err = max(order_guide_space)

    # adjust y-axis limits, add legend
    plt.ylim([min_err / 2, max_err * 2])
    plt.legend(loc=2, ncol=1, numpoints=1)
    # plt.grid(True, which="both")
    plt.grid(True, which="minor", ls="--", color='0.8')
    plt.grid(True, which="major", ls="-", color='0.001')

    # Get the data directory
    path = "data/"

    # If it does not exist, create the 'data' directory at the specified path, including any necessary parent directories
    Path(path).mkdir(parents=True, exist_ok=True)

    # save plot as PDF, beautify
    fname = path + 'heat_equation_3rd_order_time_FEniCS.png'
    plt.savefig(fname, bbox_inches='tight')

    # plt.show()

    return None


def main():
    # Run the simulation
    results = run_accuracy()
    plot_accuracy(results)


if __name__ == "__main__":
    main()
