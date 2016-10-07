import numpy as np

from examples.heat1d.ProblemClass import heat1d
from implementations.problem_classes.HeatEquation_1D_FD_forced import heat1d_forced
from implementations.transfer_classes.TransferMesh_1D_IMEX import mesh_to_mesh_1d_dirichlet
# from examples.heat1d.TransferClass import mesh_to_mesh_1d_dirichlet
from implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from implementations.collocation_classes.gauss_legendre import CollGaussLegendre
from implementations.collocation_classes.gauss_lobatto import CollGaussLobatto
# from implementations.collocation_classes.equidistant_spline_right import EquidistantSpline_Right
from implementations.controller_classes.allinclusive_multigrid_nonMPI import allinclusive_multigrid_nonMPI
from implementations.controller_classes.allinclusive_classic_nonMPI import allinclusive_classic_nonMPI
from implementations.datatype_classes.mesh import mesh, rhs_imex_mesh
from implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC import Log
# from pySDC.Stats import grep_stats, sort_stats
import logging


if __name__ == "__main__":

    num_procs = 1

    # This comes as read-in for the level class  (this is optional!)
    lparams = {}
    lparams['restol'] = 1E-10
    lparams['dt'] = 0.1

    # This comes as read-in for the controller
    cparams = {}
    cparams['fine_comm'] = True
    cparams['predict'] = True
    cparams['logger_level'] = 20

    # This comes as read-in for the problem class
    pparams = {}
    pparams['nu'] = 1.0
    pparams['nvars'] = [63]
    pparams['freq'] = 1

    # This comes as read-in for the base_transfer operations (this is optional!)
    tparams = {}
    tparams['iorder'] = 6
    tparams['rorder'] = 2

    # This comes as read-in for the sweeper class
    swparams = {}
    swparams['collocation_class'] = CollGaussRadau_Right
    # swparams['collocation_class'] = CollGaussLobatto
    # swparams['collocation_class'] = CollGaussLegendre
    swparams['num_nodes'] = [5]
    swparams['do_LU'] = True

    # Step parameters
    sparams = {}
    sparams['maxiter'] = 20

    # Fill description dictionary for easy hierarchy creation
    description = {}
    description['problem_class'] = heat1d_forced
    description['problem_params'] = pparams
    description['dtype_u'] = mesh
    description['dtype_f'] = rhs_imex_mesh
    description['sweeper_class'] = imex_1st_order
    description['sweeper_params'] = swparams
    description['level_params'] = lparams
    description['step_params'] = sparams
    description['space_transfer_class'] = mesh_to_mesh_1d_dirichlet
    description['space_transfer_params'] = tparams

    # initialize controller
    controller = allinclusive_multigrid_nonMPI(num_procs=num_procs, controller_params=cparams, description=description)
    # controller = allinclusive_classic_nonMPI(num_procs=num_procs, controller_params=cparams, description=description)

    # setup parameters "in time"
    t0 = 0
    Tend = 0.1

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    # compute exact solution and compare
    uex = P.u_exact(Tend)

    print('error at time %s: %s' %(Tend,np.linalg.norm(uex.values-uend.values,np.inf)/np.linalg.norm(
        uex.values,np.inf)))
