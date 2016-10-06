from subprocess import call

import numpy as np
import pySDC.deprecated.PFASST_stepwise as mp
from matplotlib import pyplot as plt
from pylab import rcParams

from ProblemClass_multiscale import acoustic_1d_imex
from examples.acoustic_1d_imex.HookClass import plot_solution
from implementations.datatype_classes import mesh, rhs_imex_mesh
from implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC import CollocationClasses as collclass
from pySDC import Log
from standard_integrators import bdf2, dirk, trapezoidal, rk_imex

fs = 8

if __name__ == "__main__":

    # set global logger (remove this if you do not want the output at all)
    logger = Log.setup_custom_logger('root')

    num_procs = 1

    # This comes as read-in for the level class
    lparams = {}
    lparams['restol'] = 1E-10

    sparams = {}
    
    ### SET NUMBER OF ITERATIONS ###
    sparams['maxiter'] = 2

    # setup parameters "in time"
    t0   = 0.0
    Tend = 3.0
    nsteps = 154 # 154 is value in Vater et al.
    dt = Tend/float(nsteps)
    
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
    
    ### SET TYPE OF QUADRATURE NODES ###
    #description['collocation_class'] = collclass.CollGaussLobatto
    #description['collocation_class'] = collclass.CollGaussLegendre
    description['collocation_class'] = collclass.CollGaussRadau_Right
    
    ### SET NUMBER OF NODES ###
    description['num_nodes']         = 2
    
    description['sweeper_class']     = imex_1st_order
    description['level_params']      = lparams
    description['hook_class']        = plot_solution
    #description['transfer_class'] = mesh_to_mesh_1d_dirichlet
    #description['transfer_params'] = tparams

    # quickly generate block of steps
    MS = mp.generate_steps(num_procs,sparams,description)

    # get initial values on finest level
    P = MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend,stats = mp.run_pfasst(MS,u0=uinit,t0=t0,dt=dt,Tend=Tend)

    # instantiate standard integrators to be run for comparison
    trap  = trapezoidal( (P.A+P.Dx).astype('complex'), 0.5 )
    bdf2  = bdf2( P.A+P.Dx)
    dirk  = dirk( (P.A+P.Dx).astype('complex'), sparams['maxiter'])
    rkimex = rk_imex(P.A.astype('complex'), P.Dx.astype('complex'), sparams['maxiter'])
    
    y0_tp = np.concatenate( (uinit.values[0,:], uinit.values[1,:]) )
    y0_bdf  = y0_tp
    y0_dirk = y0_tp.astype('complex')
    y0_imex = y0_tp.astype('complex')
    
    # Perform time steps with standard integrators
    for i in range(0,nsteps):  

      # trapezoidal rule step
      ynew_tp = trap.timestep(y0_tp, dt)

      # BDF-2 scheme
      if i==0:
        ynew_bdf = bdf2.firsttimestep( y0_bdf, dt)
        ym1_bdf = y0_bdf
      else:
        ynew_bdf = bdf2.timestep( y0_bdf, ym1_bdf, dt)

      # DIRK scheme
      ynew_dirk = dirk.timestep(y0_dirk, dt)

      # IMEX scheme
      ynew_imex = rkimex.timestep(y0_imex, dt)
      
      y0_tp   = ynew_tp
      ym1_bdf = y0_bdf
      y0_bdf  = ynew_bdf
      y0_dirk = ynew_dirk
      y0_imex = ynew_imex
      
    # Finished running standard integrators
      unew_tp, pnew_tp     = np.split(ynew_tp, 2)
      unew_bdf, pnew_bdf   = np.split(ynew_bdf, 2)
      unew_dirk, pnew_dirk = np.split(ynew_dirk, 2)
      unew_imex, pnew_imex = np.split(ynew_imex, 2)

    rcParams['figure.figsize'] = 2.5, 2.5
    fig = plt.figure()

    sigma_0 = 0.1
    k       = 7.0*2.0*np.pi
    x_0     = 0.75
    x_1     = 0.25

    print ('Maximum pressure in SDC: %5.3e' % np.linalg.norm(uend.values[1,:], np.inf))
    print ('Maximum pressure in DIRK: %5.3e' % np.linalg.norm(pnew_dirk, np.inf))
    print ('Maximum pressure in RK-IMEX: %5.3e' % np.linalg.norm(pnew_imex, np.inf))

    #plt.plot(P.mesh, pnew_tp,  '-', color='c', label='Trapezoidal')
    if dirk.order==2:
      plt.plot(P.mesh, pnew_bdf, 'd-', color='c', label='BDF-2',markevery=(50,75))
    p_slow = np.exp(-np.square( np.mod( P.mesh-pparams['cadv']*Tend, 1.0 ) -x_0 )/(sigma_0*sigma_0))
    plt.plot(P.mesh, p_slow, '--', color='k', markersize=fs-2, label='Slow mode', dashes=(10,2))
    if np.linalg.norm(pnew_imex, np.inf)<=2:
      plt.plot(P.mesh, pnew_imex,  '+-', color='r', label='IMEX('+str(rkimex.order)+')',markevery=(1,75),mew=1.0)
    plt.plot(P.mesh, uend.values[1,:], 'o-', color='b', label='SDC('+str(sparams['maxiter'])+')',markevery=(25,75))
    plt.plot(P.mesh, pnew_dirk, '-', color='g', label='DIRK('+str(dirk.order)+')')
    #plt.plot(P.mesh, uex.values[1,:],  '+', color='r', label='p (exact)')
    #plt.plot(P.mesh, uend.values[1,:], '-', color='b', linewidth=2.0, label='p (SDC)')

    plt.xlabel('x', fontsize=fs, labelpad=0)
    plt.ylabel('Pressure', fontsize=fs, labelpad=0)
    fig.gca().set_xlim([0, 1.0])
    fig.gca().set_ylim([-0.5, 1.1])
    fig.gca().tick_params(axis='both', labelsize=fs)
    plt.legend(loc='upper left', fontsize=fs, prop={'size':fs}, handlelength=3)
    fig.gca().grid()
    #plt.show()
    filename = 'multiscale-K'+str(sparams['maxiter'])+'-M'+str(description['num_nodes'])+'.pdf'
    plt.gcf().savefig(filename, bbox_inches='tight')
    call(["pdfcrop", filename, filename])

    #plt.plot(P.mesh, uend.values[1,:], '-', color='b', linewidth=2.0, label='p (SDC)')

    fig = plt.figure()
    p_slow = np.exp(-np.square( np.mod( P.mesh-pparams['cadv']*Tend, 1.0 ) -x_0 )/(sigma_0*sigma_0))
    plt.plot(P.mesh, uinit.values[1,:], '-', color='b')
    plt.xlabel('x', fontsize=fs, labelpad=0)
    plt.ylabel('Pressure', fontsize=fs, labelpad=0)
    fig.gca().set_xlim([0, 1.0])
    fig.gca().set_ylim([-0.5, 1.1])
    fig.gca().tick_params(axis='both', labelsize=fs)
    fig.gca().grid()
    #plt.show()
    filename = 'multiscale-initial.pdf'
    plt.gcf().savefig(filename, bbox_inches='tight')
    call(["pdfcrop", filename, filename])
