import numpy as np
import pySDC.deprecated.PFASST_stepwise as mp

from ProblemClass import boussinesq_2d_imex
from examples.boussinesq_2d_imex.HookClass import plot_solution
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
    lparams['restol'] = 1E-8
    
    swparams = {}
    swparams['collocation_class'] = collclass.CollGaussLobatto
    swparams['num_nodes'] = 3
    swparams['do_LU'] = False

    sparams = {}
    sparams['maxiter'] = 12

    # setup parameters "in time"
    t0     = 0
    Tend   = 3000
    Nsteps =  500
    #Tend   = 30
    #Nsteps =  5
    dt = Tend/float(Nsteps)

    # This comes as read-in for the problem class
    pparams = {}
    pparams['nvars']    = [(4,450,30)]
    pparams['u_adv']    = 0.02
    pparams['c_s']      = 0.3
    pparams['Nfreq']    = 0.01
    pparams['x_bounds'] = [(-150.0, 150.0)]
    pparams['z_bounds'] = [(   0.0,  10.0)]
    pparams['order']    = [4] # [fine_level, coarse_level]
    pparams['order_upw'] = [5]
    pparams['gmres_maxiter'] = [50]
    pparams['gmres_restart'] = [20]
    pparams['gmres_tol_limit'] = [1e-5]
    pparams['gmres_tol_factor'] = [0.1]

    # This comes as read-in for the transfer operations
    tparams = {}
    tparams['finter'] = False

    # Fill description dictionary for easy hierarchy creation
    description = {}
    description['problem_class']     = boussinesq_2d_imex
    description['problem_params']    = pparams
    description['dtype_u']           = mesh
    description['dtype_f']           = rhs_imex_mesh
    description['sweeper_params']    = swparams
    description['sweeper_class']     = imex_1st_order
    description['level_params']      = lparams
    description['hook_class']        = plot_solution
  #  description['transfer_class']    = mesh_to_mesh_2d
  #  description['transfer_params']   = tparams

    # quickly generate block of steps
    MS = mp.generate_steps(num_procs,sparams,description)

    # get initial values on finest level
    P = MS[0].levels[0].prob
    uinit = P.u_exact(t0)
    
    cfl_advection = pparams['u_adv']*dt/P.h[0]
    cfl_acoustic_hor = pparams['c_s']*dt/P.h[0]
    cfl_acoustic_ver = pparams['c_s']*dt/P.h[1]
    print ("CFL number of advection: %4.2f" % cfl_advection)
    print ("CFL number of acoustics (horizontal): %4.2f" % cfl_acoustic_hor)
    print ("CFL number of acoustics (vertical):   %4.2f" % cfl_acoustic_ver)

    # call main function to get things done...
    uend,stats = mp.run_pfasst(MS,u0=uinit,t0=t0,dt=dt,Tend=Tend)

    # compute exact solution and compare
    uex = P.u_exact(Tend)

    print('error at time %s: %9.5e' %(Tend,np.linalg.norm(uex.values[2,:,:].flatten()-uend.values[2,:,:].flatten(),np.inf)/np.linalg.norm(
        uex.values.flatten(),np.inf)))
    
    P.report_log()

    #plt.show()

    #extract_stats = grep_stats(stats,iter=-1,type='residual')
    #sortedlist_stats = sort_stats(extract_stats,sortby='step')
    #print(extract_stats,sortedlist_stats)
