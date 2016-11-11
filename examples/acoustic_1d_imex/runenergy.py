from subprocess import call

import numpy as np
import pySDC_core.deprecated.PFASST_stepwise as mp

from ProblemClass_conv import acoustic_1d_imex
from examples.acoustic_1d_imex.HookClass import plot_solution
from pySDC_implementations.datatype_classes import mesh, rhs_imex_mesh
from pySDC_implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC_core import CollocationClasses as collclass
from pySDC_core import Log
from standard_integrators import dirk, rk_imex

if __name__ == "__main__":

    # set global logger (remove this if you do not want the output at all)
    logger = Log.setup_custom_logger('root')

    num_procs = 1

    # This comes as read-in for the level class
    lparams = {}
    lparams['restol'] = 1E-14

    sparams = {}
  
    # This comes as read-in for the problem class
    pparams = {}
    pparams['cadv']      = 0.1
    pparams['cs']        = 1.00
    pparams['order_adv'] = 5
    pparams['waveno']    = 5

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
    
    description['sweeper_class']     = imex_1st_order
    description['do_coll_update']    = True
    description['level_params']      = lparams
    description['hook_class']        = plot_solution
    
    nsteps = 20
    
    for order in [3, 4, 5]:
    
      # setup parameters "in time"
      t0   = 0
      Tend = 1.0
      
      ### SET NUMBER OF NODES DEPENDING ON REQUESTED ORDER ###
      if order==3:
        description['num_nodes'] = 3
      elif order==4:
        description['num_nodes'] = 3
      elif order==5:
        description['num_nodes'] = 3
      
      sparams['maxiter'] = order
      
      pparams['nvars'] = [(2,160)]
        
      # quickly generate block of steps
      MS = mp.generate_steps(num_procs,sparams,description)
        
      dt = Tend/float(nsteps)

      # get initial values on finest level
      P = MS[0].levels[0].prob
      uinit = P.u_exact(t0)
      file = open('energy-exact.txt', 'w')
      E = np.sum(np.square(uinit.values[0,:]) + np.square(uinit.values[1,:]))
      file.write('%30.20f\n' % E)
      file.write('%30.20f\n' % float(Tend))
      file.write('%30.20f' % float(nsteps))
      file.close()
      
      print "Time step: %4.2f" % dt
      print "Fast CFL number: %4.2f" % (pparams['cs']*dt/P.dx)
      print "Slow CFL number: %4.2f" % (pparams['cadv']*dt/P.dx)
      
      ### Run standard integrators first
      _dirk = dirk( (P.A+P.Dx).astype('complex'), sparams['maxiter'])
      _rkimex = rk_imex(P.A.astype('complex'), P.Dx.astype('complex'), sparams['maxiter'])
      y_dirk = np.concatenate( (uinit.values[0,:], uinit.values[1,:]) )
      y_imex   = np.concatenate( (uinit.values[0,:], uinit.values[1,:]) )
      y_dirk = y_dirk.astype('complex')
      y_imex = y_imex.astype('complex')
      file_dirk = open('energy-dirk-'+str(sparams['maxiter'])+'.txt', 'w')
      file_imex = open('energy-imex-'+str(sparams['maxiter'])+'.txt', 'w')
      
      for nn in range(nsteps):
        y_dirk = _dirk.timestep(y_dirk, dt)
        y_imex = _rkimex.timestep(y_imex, dt)
        y_e    = np.split(y_dirk, 2)
        E      = np.sum(np.square(y_e[0]) + np.square(y_e[1]))
        file_dirk.write('%30.20f\n' % E)
        y_e    = np.split(y_imex, 2)
        E      = np.sum(np.square(y_e[0]) + np.square(y_e[1]))
        file_imex.write('%30.20f\n' % E)
      
      ### Run SDC

      # call main function to get things done...
      uend,stats = mp.run_pfasst(MS, u0=uinit, t0=t0, dt=dt, Tend=Tend)

      # rename file with energy data to indicate M and K
      filename = 'energy-sdc-K'+str(sparams['maxiter'])+'-M'+str(description['num_nodes'])+'.txt'
      call(['mv', 'energy-sdc.txt', filename])
