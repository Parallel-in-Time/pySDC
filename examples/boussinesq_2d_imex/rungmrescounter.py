import numpy as np
import pySDC.core.deprecated.PFASST_stepwise as mp

from ProblemClass import boussinesq_2d_imex
from examples.boussinesq_2d_imex.HookClass import plot_solution
from pySDC.implementations.datatype_classes import mesh, rhs_imex_mesh
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.core import CollocationClasses as collclass
from standard_integrators import dirk, rk_imex, SplitExplicit
from unflatten import unflatten

if __name__ == "__main__":

    # set global logger (remove this if you do not want the output at all)
    # logger = Log.setup_custom_logger('root')

    num_procs = 1

    # This comes as read-in for the level class
    lparams = {}
    lparams['restol'] = 1E-15
    
    swparams = {}
    
    ### SET TYPE OF QUADRATURE NODES ###
    #swparams['collocation_class'] = collclass.CollGaussLobatto
    swparams['collocation_class'] = collclass.CollGaussLegendre
    #swparams['collocation_class'] = collclass.CollGaussRadau_Right
    
    ### SET NUMBER OF NODES ###
    swparams['num_nodes'] = 3
    
    swparams['do_LU'] = False

    sparams = {}
    
    ### SET NUMBER OF ITERATIONS ###
    sparams['maxiter'] = 4

    ### ORDER OF DIRK/IMEX EQUAL TO NUMBER OF SDC ITERATIONS AND THUS SDC ORDER ###
    dirk_order = sparams['maxiter']

    # setup parameters "in time"
    t0     = 0

    Tend   = 3000   
    Nsteps =  100

    dt = Tend/float(Nsteps)

    # This comes as read-in for the problem class
    pparams = {}
    pparams['nvars']    = [(4,300,30)]
    pparams['u_adv']    = 0.02
    pparams['c_s']      = 0.3
    pparams['Nfreq']    = 0.01
    pparams['x_bounds'] = [(-150.0, 150.0)]
    pparams['z_bounds'] = [(   0.0,  10.0)]
    pparams['order']    = [4] # [fine_level, coarse_level]
    pparams['order_upw'] = [5]
    pparams['gmres_maxiter'] = [500]
    pparams['gmres_restart'] = [10]
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

    # quickly generate block of steps
    MS = mp.generate_steps(num_procs,sparams,description)

    # get initial values on finest level
    P = MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    cfl_advection    = pparams['u_adv']*dt/P.h[0]
    cfl_acoustic_hor = pparams['c_s']*dt/P.h[0]
    cfl_acoustic_ver = pparams['c_s']*dt/P.h[1]
    print("Horizontal resolution: %4.2f" % P.h[0])
    print("Vertical resolution:   %4.2f" % P.h[1])
    print("CFL number of advection: %4.2f" % cfl_advection)
    print("CFL number of acoustics (horizontal): %4.2f" % cfl_acoustic_hor)
    print("CFL number of acoustics (vertical):   %4.2f" % cfl_acoustic_ver)

    method_split = 'MIS4_4'
#   method_split = 'RK3'
    splitp = SplitExplicit(P, method_split, pparams) 

    dirkp = dirk(P, dirk_order)

    u0 = uinit.values.flatten()
    usplit = np.copy(u0)
#   for i in range(0,Nsteps):
    print(np.linalg.norm(usplit))  
    for i in range(0,2*Nsteps):
      usplit = splitp.timestep(usplit, dt/2)  
    print(np.linalg.norm(usplit))

    dirkp = dirk(P, dirk_order)
    udirk = np.copy(u0)
    print("Running DIRK ....")
    print(np.linalg.norm(udirk))  
    for i in range(0,Nsteps):
      udirk = dirkp.timestep(udirk, dt)  
    print np.linalg.norm(udirk)

    rkimex = rk_imex(P, dirk_order)
    uimex  = np.copy(u0)
    dt_imex = dt
    print("Running RK-IMEX ....")
    for i in range(0,Nsteps):
#     print("Running RK-IMEWX Step:  %4.2f" % dt_imex)
      uimex = rkimex.timestep(uimex, dt_imex)
    print(np.linalg.norm(uimex))  

    # call main function to get things done...
    print("Running SDC...")
    uend,stats = mp.run_pfasst(MS,u0=uinit,t0=t0,dt=dt,Tend=Tend)

    # For reference solution, increase GMRES tolerance
    P.gmres_tol_limit = 1e-10
    rkimexref = rk_imex(P, 5)
    uref      = np.copy(u0)
    dt_ref    = dt/10.0
    print("Running RK-IMEX reference....")
    for i in range(0,10*Nsteps):
      uref = rkimexref.timestep(uref, dt_ref)
  
    usplit = unflatten(usplit, 4, P.N[0], P.N[1])
    udirk = unflatten(udirk, 4, P.N[0], P.N[1])
    uimex = unflatten(uimex, 4, P.N[0], P.N[1])
    uref  = unflatten(uref,  4, P.N[0], P.N[1])

    np.save('xaxis', P.xx)
    np.save('sdc', uend.values)
    np.save('dirk', udirk)
    np.save('rkimex', uimex)
    np.save('split', usplit)
    np.save('uref', uref)

    print "diff split  ",np.linalg.norm(uref-usplit)
    print "diff dirk   ",np.linalg.norm(uref-udirk)
    print "diff rkimex ",np.linalg.norm(uref-uimex)
    print "diff sdc    ",np.linalg.norm(uref-uend.values)
    
    print(" #### Logging report for Split    #### " )
    print("Total number of matrix multiplcations: %5i" % splitp.logger.nsmall)

    print(" #### Logging report for DIRK-%1i #### " % dirkp.order)
    print("Number of calls to implicit solver: %5i" % dirkp.logger.solver_calls)
    print("Total number of GMRES iterations: %5i" % dirkp.logger.iterations)
    print("Average number of iterations per call: %6.3f" % (float(dirkp.logger.iterations)/float(dirkp.logger.solver_calls)))
    print(" ")
    print(" #### Logging report for RK-IMEX-%1i #### " % rkimex.order)
    print("Number of calls to implicit solver: %5i" % rkimex.logger.solver_calls)
    print("Total number of GMRES iterations: %5i" % rkimex.logger.iterations)
    print("Average number of iterations per call: %6.3f" % (float(rkimex.logger.iterations)/float(rkimex.logger.solver_calls)))
    print(" ")
    print(" #### Logging report for SDC-(%1i,%1i) #### " % (swparams['num_nodes'], sparams['maxiter']))
    print("Number of calls to implicit solver: %5i" % P.logger.solver_calls)
    print("Total number of GMRES iterations: %5i" % P.logger.iterations)
    print("Average number of iterations per call: %6.3f" % (float(P.logger.iterations)/float(P.logger.solver_calls)))

