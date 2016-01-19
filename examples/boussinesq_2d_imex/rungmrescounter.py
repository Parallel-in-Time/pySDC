
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
from pylab import rcParams

from unflatten import unflatten

from standard_integrators import dirk, bdf2, trapezoidal, rk_imex

if __name__ == "__main__":

    # set global logger (remove this if you do not want the output at all)
    logger = Log.setup_custom_logger('root')

    num_procs = 1

    # This comes as read-in for the level class
    lparams = {}
    lparams['restol'] = 1E-15
    
    swparams = {}
    swparams['collocation_class'] = collclass.CollGaussLegendre
    swparams['num_nodes'] = 3
    swparams['do_LU'] = False

    sparams = {}
    sparams['maxiter'] = 4

    dirk_order = 3

    # setup parameters "in time"
    t0     = 0
    Tend   = 3000
    Nsteps =  100
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
    pparams['gmres_maxiter'] = [500]
    pparams['gmres_restart'] = [10]
    pparams['gmres_tol']     = [1e-5]

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

    cfl_advection    = pparams['u_adv']*dt/P.h[0]
    cfl_acoustic_hor = pparams['c_s']*dt/P.h[0]
    cfl_acoustic_ver = pparams['c_s']*dt/P.h[1]
    print "Horizontal resolution: %4.2f" % P.h[0]
    print "Vertical resolution:   %4.2f" % P.h[1]
    print ("CFL number of advection: %4.2f" % cfl_advection)
    print ("CFL number of acoustics (horizontal): %4.2f" % cfl_acoustic_hor)
    print ("CFL number of acoustics (vertical):   %4.2f" % cfl_acoustic_ver)

    dirkp = dirk(P, dirk_order)
    u0 = uinit.values.flatten()
    udirk = u0
    for i in range(0,Nsteps):
      udirk = dirkp.timestep(udirk, dt)  

    dirkref = dirk(P, 4)
    uref    = u0
    dt_ref  = dt/10.0
    for i in range(0,10*Nsteps):
      uref = dirkref.timestep(uref, dt_ref)

    rkimex = rk_imex(P, 3)
    uimex  = u0
    dt_imex = dt
    for i in range(0,Nsteps):
      uimex = rkimex.timestep(uimex, dt_imex)
  
    # call main function to get things done...
    uend,stats = mp.run_pfasst(MS,u0=uinit,t0=t0,dt=dt,Tend=Tend)
    udirk = unflatten(udirk, 4, P.N[0], P.N[1])
    uimex = unflatten(uimex, 4, P.N[0], P.N[1])
    uref  = unflatten(uref,  4, P.N[0], P.N[1])

    np.save('xaxis', P.xx)
    np.save('sdc', uend.values)
    np.save('dirk', udirk)
    np.save('rkimex', uimex)
    np.save('uref', uref)
    
    print " #### Logging report for DIRK-%1i #### " % dirkp.order
    print "Number of calls to implicit solver: %5i" % dirkp.logger.solver_calls
    print "Total number of GMRES iterations: %5i" % dirkp.logger.iterations
    print "Average number of iterations per call: %6.3f" % (float(dirkp.logger.iterations)/float(dirkp.logger.solver_calls))
    print " "
    print " #### Logging report for RK-IMEX-%1i #### " % rkimex.order
    print "Number of calls to implicit solver: %5i" % rkimex.logger.solver_calls
    print "Total number of GMRES iterations: %5i" % rkimex.logger.iterations
    print "Average number of iterations per call: %6.3f" % (float(rkimex.logger.iterations)/float(rkimex.logger.solver_calls))
    print " "
    print " #### Logging report for SDC-(%1i,%1i) #### " % (swparams['num_nodes'], sparams['maxiter'])
    print "Number of calls to implicit solver: %5i" % P.logger.solver_calls
    print "Total number of GMRES iterations: %5i" % P.logger.iterations
    print "Average number of iterations per call: %6.3f" % (float(P.logger.iterations)/float(P.logger.solver_calls))  

