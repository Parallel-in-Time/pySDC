from subprocess import call

import numpy as np
import pySDC.deprecated.PFASST_stepwise as mp
from matplotlib import pyplot as plt
from pylab import rcParams

from ProblemClass_conv import acoustic_1d_imex
from examples.acoustic_1d_imex.HookClass import plot_solution
from implementations.datatype_classes import mesh, rhs_imex_mesh
from implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC import CollocationClasses as collclass
from pySDC import Log

if __name__ == "__main__":

    fs = 8

    # set global logger (remove this if you do not want the output at all)
    logger = Log.setup_custom_logger('root')

    num_procs = 1

    # This comes as read-in for the level class
    lparams = {}
    lparams['restol'] = 1E-10

    sparams = {}
  
    cs_v = [0.5, 1.0, 1.5, 5.0]
    sparams['maxiter'] = 15
    
    ### SET NUMBER OF NODES ###
    nodes_v = [3]
      
    residual = np.zeros((np.size(cs_v), np.size(nodes_v), sparams['maxiter'])) 
    convrate = np.zeros((np.size(cs_v), np.size(nodes_v), sparams['maxiter']-1))
    lastiter = np.zeros(( np.size(cs_v), np.size(nodes_v) )) + sparams['maxiter']
    avg_convrate = np.zeros(( np.size(cs_v), np.size(nodes_v) ))

    for cs_ind in range(0,np.size(cs_v)):

      # This comes as read-in for the problem class
      pparams = {}
      pparams['nvars']     = [(2,300)]
      pparams['cadv']      = 0.1
      pparams['cs']        = cs_v[cs_ind]
      pparams['order_adv'] = 5
      pparams['waveno']    = 5

      # This comes as read-in for the transfer operations
      tparams = {}
      tparams['finter'] = True

      # Fill description dictionary for easy hierarchy creation
      description = {}
      description['problem_class']     = acoustic_1d_imex
      description['problem_params']    = pparams
      description['dtype_u']           = mesh
      description['dtype_f']           = rhs_imex_mesh
      
      ### SELECT TYPE OF QUADRATURE NODES ###
      #description['collocation_class'] = collclass.CollGaussLobatto
      #description['collocation_class'] = collclass.CollGaussLegendre
      description['collocation_class'] = collclass.CollGaussRadau_Right
      
      description['sweeper_class']     = imex_1st_order
      description['level_params']      = lparams
      description['hook_class']        = plot_solution
      #description['transfer_class'] = mesh_to_mesh_dirichlet
      #description['transfer_params'] = tparams
      
      for nodes_ind in np.arange(np.size(nodes_v)):
        # setup parameters "in time"
        t0   = 0
        Tend = 0.025
        description['num_nodes'] = nodes_v[nodes_ind]

         # quickly generate block of steps
        MS = mp.generate_steps(num_procs,sparams,description)
      
        dt = Tend

        # get initial values on finest level
        P = MS[0].levels[0].prob
        uinit = P.u_exact(t0)
        
        print "Fast CFL number: %4.2f" % (pparams['cs']*dt/P.dx) 
        print "Slow CFL number: %4.2f" % (pparams['cadv']*dt/P.dx) 

        # call main function to get things done...
        uend,stats = mp.run_pfasst(MS, u0=uinit, t0=t0, dt=dt, Tend=Tend)

        for k,v in stats.items():
            iter = getattr(k,'iter')
            if iter is not -1:
                residual[cs_ind, nodes_ind, iter-1] = v   

        # Compute convergence rates
        for iter in range(0,sparams['maxiter']-1):
            if residual[cs_ind, nodes_ind, iter]< lparams['restol']:
              lastiter[cs_ind,nodes_ind] = iter
            else:
              convrate[cs_ind, nodes_ind, iter] = residual[cs_ind, nodes_ind, iter+1]/residual[cs_ind, nodes_ind, iter]

        # Compute estimate
        #lambda_fast = LA.eigsh(P.A,  k=1, which='LM')
        #lambda_slow = LA.eigsh(P.Dx, k=1, which='LM')
        avg_convrate[cs_ind, nodes_ind] = np.sum(convrate[cs_ind, nodes_ind, :])/float(lastiter[cs_ind,nodes_ind])

  #### end of for loops ####

    color = [ 'r', 'b', 'g', 'c' ]
    shape = ['o-', 'd-', 's-', '>-']
    rcParams['figure.figsize'] = 2.5, 2.5
    fig = plt.figure()
    for ii in range(0,np.size(cs_v)):
      x = np.arange(1,lastiter[ii,0])
      y = convrate[ii, 0, 0:lastiter[ii,0]-1]
      plt.plot(x, y, shape[ii], markersize=fs-2, color=color[ii], label=r'$C_{\rm fast}$=%4.2f' % (cs_v[ii]*dt/P.dx))       
      #plt.plot(x, 0.0*y+avg_convrate[ii,0], '--', color=color[ii])

    plt.legend(loc='upper right', fontsize=fs, prop={'size':fs-2})
    plt.xlabel('Iteration', fontsize=fs)
    plt.ylabel(r'$|| r^{k+1} ||_{\infty}/|| r^k ||_{\infty}$', fontsize=fs, labelpad=2)
    plt.xlim([0, sparams['maxiter']])
    plt.ylim([0, 1.0])
    plt.yticks(fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.show()
    filename = 'iteration.pdf'
    fig.savefig(filename,bbox_inches='tight')
    call(["pdfcrop", filename, filename])


