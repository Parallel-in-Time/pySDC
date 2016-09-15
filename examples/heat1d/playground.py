import numpy as np

from pySDC.controller_classes.allinclusive_blockwise_nonMPI import allinclusive_blockwise_nonMPI
from pySDC.controller_classes.allinclusive_stepwise_nonMPI import allinclusive_stepwise_nonMPI
from examples.heat1d.ProblemClass import heat1d
from examples.heat1d.TransferClass import mesh_to_mesh_1d
from pySDC import CollocationClasses as collclass
from pySDC import Log
from pySDC.datatype_classes.mesh import mesh, rhs_imex_mesh
from pySDC.sweeper_classes.imex_1st_order import imex_1st_order
# from pySDC.Stats import grep_stats, sort_stats


if __name__ == "__main__":

    # set global logger (remove this if you do not want the output at all)
    logger = Log.setup_custom_logger('root')

    num_procs = 3

    # This comes as read-in for the level class  (this is optional!)
    lparams = {}
    lparams['restol'] = 1E-10

    # This comes as read-in for the step class (this is optional!)
    sparams = {}
    sparams['maxiter'] = 20
    sparams['fine_comm'] = True
    sparams['predict'] = True

    # This comes as read-in for the problem class
    pparams = {}
    pparams['nu'] = 1.0
    pparams['nvars'] = [63, 31]

    # This comes as read-in for the transfer operations (this is optional!)
    tparams = {}
    tparams['finter'] = False
    tparams['iorder'] = 6
    tparams['rorder'] = 2

    # This comes as read-in for the sweeper class
    swparams = {}
    swparams['collocation_class'] = collclass.CollGaussRadau_Right
    swparams['num_nodes'] = 5

    # Fill description dictionary for easy hierarchy creation
    description = {}
    description['problem_class'] = heat1d
    description['problem_params'] = pparams
    description['dtype_u'] = mesh
    description['dtype_f'] = rhs_imex_mesh
    description['sweeper_class'] = imex_1st_order
    description['sweeper_params'] = swparams
    description['level_params'] = lparams
    description['transfer_class'] = mesh_to_mesh_1d
    description['transfer_params'] = tparams

    # initialize controller
    controller = allinclusive_blockwise_nonMPI(num_procs=num_procs, step_params=sparams, description=description)
    # controller = allinclusive_stepwise_nonMPI(num_procs=num_procs, step_params=sparams, description=description)

    # setup parameters "in time"
    t0 = 0
    dt = 0.12
    Tend = 3*dt

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, dt=dt, Tend=Tend)

    # compute exact solution and compare
    uex = P.u_exact(Tend)

    print('error at time %s: %s' %(Tend,np.linalg.norm(uex.values-uend.values,np.inf)/np.linalg.norm(
        uex.values,np.inf)))
