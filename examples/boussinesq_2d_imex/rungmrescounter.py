
from pySDC import CollocationClasses as collclass

import numpy as np

from ProblemClass import boussinesq_2d_imex
from examples.boussinesq_2d_imex.TransferClass import mesh_to_mesh_2d
from examples.boussinesq_2d_imex.HookClass import plot_solution

from pySDC.datatype_classes.mesh import mesh, rhs_imex_mesh
from pySDC.sweeper_classes.imex_1st_order import imex_1st_order
import pySDC.PFASST_stepwise as mp
from pySDC import Log
from pySDC.Stats import grep_stats, sort_stats

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from unflatten import unflatten

from standard_integrators import dirk

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
    swparams['do_LU'] = True

    sparams = {}
    sparams['maxiter'] = 4

    dirk_order = 4

    # setup parameters "in time"
    t0     = 0
    Tend   = 3000
    Nsteps =  500
    #Tend   = 30
    #Nsteps =  5
    dt = Tend/float(Nsteps)

    # This comes as read-in for the problem class
    pparams = {}
    #pparams['nvars']    = [(4,450,30)]
    pparams['nvars']    = [(4,150,10)]
    pparams['u_adv']    = 0.02
    pparams['c_s']      = 0.3
    pparams['Nfreq']    = 0.01
    pparams['x_bounds'] = [(-150.0, 150.0)]
    pparams['z_bounds'] = [(   0.0,  10.0)]
    pparams['order']    = [4] # [fine_level, coarse_level]
    pparams['order_upw'] = [5]
    pparams['gmres_maxiter'] = [50]
    pparams['gmres_restart'] = [10]
    pparams['gmres_tol']     = [1e-6]

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

    # quickly generate block of steps
    MS = mp.generate_steps(num_procs,sparams,description)

    # get initial values on finest level
    P = MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    dirk = dirk(P, dirk_order)
    u0 = uinit.values.flatten()

    for i in range(0,Nsteps):
      u0 = dirk.timestep(u0, dt)
    
    cfl_advection    = pparams['u_adv']*dt/P.h[0]
    cfl_acoustic_hor = pparams['c_s']*dt/P.h[0]
    cfl_acoustic_ver = pparams['c_s']*dt/P.h[1]
    print ("CFL number of advection: %4.2f" % cfl_advection)
    print ("CFL number of acoustics (horizontal): %4.2f" % cfl_acoustic_hor)
    print ("CFL number of acoustics (vertical):   %4.2f" % cfl_acoustic_ver)

    # call main function to get things done...
    uend,stats = mp.run_pfasst(MS,u0=uinit,t0=t0,dt=dt,Tend=Tend)

    u0 = unflatten(u0, 4, P.N[0], P.N[1])
    fig = plt.figure()

    plt.plot(P.xx[:,5], uend.values[2,:,5], color='r', marker='+', markevery=3)
    plt.plot(P.xx[:,5], u0[2,:,5], color='b', markevery=5)
    plt.show()

    print " #### Logging report for DIRK #### "
    print "Number of calls to implicit solver: %5i" % dirk.logger.solver_calls
    print "Total number of GMRES iterations: %5i" % dirk.logger.iterations
    print "Average number of iterations per call: %6.3f" % (float(dirk.logger.iterations)/float(dirk.logger.solver_calls))
    
    print " #### Logging report for SDC #### "
    print "Number of calls to implicit solver: %5i" % P.logger.solver_calls
    print "Total number of GMRES iterations: %5i" % P.logger.iterations
    print "Average number of iterations per call: %6.3f" % (float(P.logger.iterations)/float(P.logger.solver_calls))
