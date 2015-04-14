
from pySDC import CollocationClasses as collclass

import numpy as np

from ProblemClass import acoustic_1d_imex
#from examples.sharpclaw_burgers1d.TransferClass import mesh_to_mesh_1d
from pySDC.datatype_classes.mesh import mesh, rhs_imex_mesh
from pySDC.sweeper_classes.imex_1st_order import imex_1st_order
import pySDC.Methods as mp
from pySDC import Log
from pySDC.Stats import grep_stats, sort_stats

# Sharpclaw imports
from clawpack import pyclaw
from clawpack import riemann
from matplotlib import pyplot as plt


if __name__ == "__main__":

    # set global logger (remove this if you do not want the output at all)
    logger = Log.setup_custom_logger('root')

    num_procs = 1

    # This comes as read-in for the level class
    lparams = {}
    lparams['restol'] = 1E-10

    sparams = {}
    sparams['maxiter'] = 20

    # setup parameters "in time"
    t0 = 0
    dt = 0.01
    Tend = 20*dt

    # This comes as read-in for the problem class
    pparams = {}
    pparams['nvars'] = [(2,100)]
    pparams['cadv']  = 0.0
    pparams['cs']    = 1.0
    pparams['order_adv'] = 2
    
    # This comes as read-in for the transfer operations
    tparams = {}
    tparams['finter'] = True

    # Fill description dictionary for easy hierarchy creation
    description = {}
    description['problem_class']     = acoustic_1d_imex
    description['problem_params']    = pparams
    description['dtype_u']           = mesh
    description['dtype_f']           = rhs_imex_mesh
    description['collocation_class'] = collclass.CollGaussLobatto
    description['num_nodes']         = 2
    description['sweeper_class']     = imex_1st_order
    description['level_params']      = lparams
    #description['transfer_class'] = mesh_to_mesh_1d
    #description['transfer_params'] = tparams

    # quickly generate block of steps
    MS = mp.generate_steps(num_procs,sparams,description)

    # get initial values on finest level
    P = MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend,stats = mp.run_pfasst_serial(MS,u0=uinit,t0=t0,dt=dt,Tend=Tend)

    # compute exact solution and compare
    uex = P.u_exact(Tend)

    print('error at time %s: %s' %(Tend,np.linalg.norm(uex.values-uend.values,np.inf)/np.linalg.norm(uex.values,np.inf)))

    fig = plt.figure(figsize=(8,8))

    plt.plot(P.state.grid.x.centers, uex.values[0,:],  '+', color='b', label='u (exact)')
    plt.plot(P.state.grid.x.centers, uend.values[0,:], '-', color='b', label='u (SDC)')
    plt.plot(P.state.grid.x.centers, uex.values[1,:],  '+', color='r', label='p (exact)')
    plt.plot(P.state.grid.x.centers, uend.values[1,:], '-', color='r', label='p (SDC)')
    plt.legend()
    plt.xlim([0, 1])
    plt.ylim([-1, 1])
    plt.show()
