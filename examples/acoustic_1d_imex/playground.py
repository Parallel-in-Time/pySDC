import numpy as np
import pySDC.deprecated.PFASST_stepwise as mp
from matplotlib import pyplot as plt

from ProblemClass import acoustic_1d_imex
from examples.acoustic_1d_imex.HookClass import plot_solution
from implementations.datatype_classes import mesh, rhs_imex_mesh
from implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC import CollocationClasses as collclass
from pySDC import Log

if __name__ == "__main__":

    # set global logger (remove this if you do not want the output at all)
    logger = Log.setup_custom_logger('root')

    num_procs = 1

    # This comes as read-in for the level class
    lparams = {}
    lparams['restol'] = 1E-10

    sparams = {}
    sparams['maxiter'] = 2

    # setup parameters "in time"
    t0 = 0.0
    Tend = 3.0
    dt = Tend/float(154)
    
    # This comes as read-in for the problem class
    pparams = {}
    pparams['nvars'] = [(2,512)]
    pparams['cadv']  = 0.1
    pparams['cs']    = 1.0
    pparams['order_adv'] = 5
    
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

    print('error at time %s: %s' %(Tend,np.linalg.norm(uex.values-uend.values,np.inf)/np.linalg.norm(uex.values,np.inf)))

    fig = plt.figure(figsize=(8,8))

    sigma_0 = 0.1
    x_0     = 0.75

    #plt.plot(P.mesh, uex.values[0,:],  '+', color='b', label='u (exact)')
    plt.plot(P.mesh, uend.values[1,:], '-', color='b', label='SDC')
    #plt.plot(P.mesh, uex.values[1,:],  '+', color='r', label='p (exact)')
    #plt.plot(P.mesh, uend.values[1,:], '-', color='b', linewidth=2.0, label='p (SDC)')
    p_slow = np.exp(-np.square(P.mesh-x_0)/(sigma_0*sigma_0))
    #plt.plot(P.mesh, p_slow, '-', color='r', markersize=4, label='slow mode')
    plt.legend(loc=2)
    plt.xlim([0, 1])
    plt.ylim([-0.1, 1.1])
    fig.gca().grid()
    #plt.show()
    plt.gcf().savefig('fwsw-sdc-K'+str(sparams['maxiter'])+'-M'+str(description['num_nodes'])+'.pdf', bbox_inches='tight')
