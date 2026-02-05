import numpy as np
import json
from pathlib import Path

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
import pySDC.helpers.plot_helper as plt_helper

import dolfin as df
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D


def run_simulation(ml=None, mass=None):

    t0 = 0
    dt = 0.2
    Tend = 1.0

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-10
    level_params['dt'] = dt

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 20

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = [2]

    problem_params = dict()
    problem_params['nu'] = 0.1
    problem_params['t0'] = t0  # ugly, but necessary to set up ProblemClass
    problem_params['c_nvars'] = [64]
    problem_params['family'] = 'CG'
    problem_params['order'] = [2]
    problem_params['c'] = 0.0
    problem_params['refinements'] = [1]

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 20
    controller_params['hook_class'] = LogSolution

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    if mass:
        description['problem_class'] = fenics_heat2D_mass_timebc
        description['sweeper_class'] = imex_1st_order_mass
        description['base_transfer_class'] = base_transfer_mass
    else:
        description['problem_class'] = fenics_heat2D
        description['sweeper_class'] = imex_1st_order  # pass sweeper (see part B)

    description['problem_params'] = problem_params
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params
    description['space_transfer_class'] = mesh_to_mesh_fenics

    # quickly generate block of steps
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    errors = get_sorted(stats, type='error', sortby='iter')
    residuals = get_sorted(stats, type='residual', sortby='iter')

    uex = P.u_exact(Tend)

    Rel_err = abs(uex - uend) / abs(uex)
    print('The relative error at time ', Tend, 'is ', Rel_err)

    # Get the data directory
    path = "data/"

    # If it does not exist, create the 'data' directory at the specified path, including any necessary parent directories
    Path(path).mkdir(parents=True, exist_ok=True)

    # Save parameters
    parameters = description['problem_params']
    parameters.update(description['level_params'])
    parameters['Tend'] = Tend
    json.dump(parameters, open(path + "heat_equation_FEniCS_parameters.json", 'w'))

    # Create XDMF file for visualization output
    xdmffile_u = df.XDMFFile(path + 'heat_equation_FEniCS_Temperature.xdmf')

    # Get the solution at every time step sorted by time
    Solutions = get_sorted(stats, type='u', sortby='time')

    for i in range(int(Tend / description['level_params']['dt'])):
        time = (i + 1) * description['level_params']['dt']
        #
        un = Solutions[i][1]
        ux = P.u_exact(time)
        #
        xdmffile_u.write_checkpoint(un.values, "un", time, df.XDMFFile.Encoding.HDF5, True)
        xdmffile_u.write_checkpoint(ux.values, "ux", time, df.XDMFFile.Encoding.HDF5, True)
        #
    xdmffile_u.close()

    return errors, residuals


if __name__ == "__main__":

    # errors_sdc_noM, _ = run_simulation(mass=False)
    errors_sdc_M, _ = run_simulation(mass=True)
