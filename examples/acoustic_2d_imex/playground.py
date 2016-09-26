import numpy as np
import pySDC.deprecated.PFASST_stepwise as mp
from matplotlib import pyplot as plt

from ProblemClass import acoustic_2d_imex
from examples.acoustic_2d_imex.HookClass import plot_solution
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
    lparams['restol'] = 3E-11

    sparams = {}
    sparams['maxiter'] = 8

    # setup parameters "in time"
    t0     = 0
    Tend   = 50.0
    Nsteps = 2000
    dt = Tend/float(Nsteps)

    # This comes as read-in for the problem class
    pparams = {}
    pparams['nvars'] = [(3, 50,25)]
    pparams['u_adv'] = 1.0
    pparams['c_s']   = 0.0
    pparams['x_bounds'] = [(-1.0, 1.0)]
    pparams['z_bounds'] = [( 0.0, 1.0)]

    # This comes as read-in for the transfer operations
    #tparams = {}
    #tparams['finter'] = False

    # Fill description dictionary for easy hierarchy creation
    description = {}
    description['problem_class']     = acoustic_2d_imex
    description['problem_params']    = pparams
    description['dtype_u']           = mesh
    description['dtype_f']           = rhs_imex_mesh
    description['collocation_class'] = collclass.CollGaussLobatto
    description['num_nodes']         = 4
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

    print('error at time %s: %9.5e' %(Tend,np.linalg.norm(uex.values[2,:,:].flatten()-uend.values[2,:,:].flatten(),np.inf)/np.linalg.norm(
        uex.values.flatten(),np.inf)))

    plt.show()

    # extract_stats = grep_stats(stats,iter=-1,type='residual')
    # sortedlist_stats = sort_stats(extract_stats,sortby='step')
    # print(extract_stats,sortedlist_stats)