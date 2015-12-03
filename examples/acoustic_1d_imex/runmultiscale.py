
from pySDC import CollocationClasses as collclass

import numpy as np

from ProblemClass_multiscale import acoustic_1d_imex
from examples.acoustic_1d_imex.HookClass import plot_solution

from pySDC.datatype_classes.mesh import mesh, rhs_imex_mesh
from pySDC.sweeper_classes.imex_1st_order import imex_1st_order
import pySDC.PFASST_stepwise as mp
from pySDC import Log
from pySDC.Stats import grep_stats, sort_stats

# Sharpclaw imports
#from clawpack import pyclaw
#from clawpack import riemann
from matplotlib import pyplot as plt


if __name__ == "__main__":

    # set global logger (remove this if you do not want the output at all)
    logger = Log.setup_custom_logger('root')

    num_procs = 1

    # This comes as read-in for the level class
    lparams = {}
    lparams['restol'] = 1E-10

    sparams = {}
    sparams['maxiter'] = 4

    # setup parameters "in time"
    t0   = 0.0
    Tend = 3.0
    dt = Tend/float(154)
    
    # This comes as read-in for the problem class
    pparams = {}
    pparams['nvars']      = [(2,512)]
    pparams['cadv']       = 0.05
    pparams['cs']         = 1.0
    pparams['order_adv']  = 5
    pparams['multiscale'] = True

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
    description['num_nodes']         = 3
    description['sweeper_class']     = imex_1st_order
    description['level_params']      = lparams
    description['hook_class']        = plot_solution
    #description['transfer_class'] = mesh_to_mesh_1d
    #description['transfer_params'] = tparams

    # quickly generate block of steps
    MS = mp.generate_steps(num_procs,sparams,description)

    # get initial values on finest level
    P = MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend,stats = mp.run_pfasst(MS,u0=uinit,t0=t0,dt=dt,Tend=Tend)

    # compute exact solution and compare
    uex = P.u_exact(Tend)

    fig = plt.figure(figsize=(8,8))

    sigma_0 = 0.1
    k       = 7.0*2.0*np.pi
    x_0     = 0.75
    x_1     = 0.25

    #plt.plot(P.mesh, uex.values[0,:],  '+', color='b', label='u (exact)')
    plt.plot(P.mesh, uend.values[1,:], '-', color='b', label='SDC')
    #plt.plot(P.mesh, uex.values[1,:],  '+', color='r', label='p (exact)')
    #plt.plot(P.mesh, uend.values[1,:], '-', color='b', linewidth=2.0, label='p (SDC)')
    p_slow = np.exp(-np.square(P.mesh-x_0-pparams['cadv']*Tend)/(sigma_0*sigma_0))
    plt.plot(P.mesh, p_slow, '-', color='r', markersize=4, label='slow mode')
    plt.legend(loc=2)
    plt.xlim([0, 1])
    plt.ylim([-0.1, 1.1])
    fig.gca().grid()
    plt.show()
    #plt.gcf().savefig('fwsw-sdc-K'+str(sparams['maxiter'])+'-M'+str(description['num_nodes'])+'.pdf', bbox_inches='tight')
