from pySDC import Level as lvl
from pySDC import Hooks as hookclass
from pySDC import CollocationClasses as collclass
from pySDC import Step as stepclass

from pySDC.datatype_classes.complex_mesh import mesh, rhs_imex_mesh
from pySDC.sweeper_classes.imex_1st_order import imex_1st_order as imex
from examples.SWFW.ProblemClass import swfw_scalar 
import numpy as np

from matplotlib import pyplot as plt
from pylab import rcParams
from subprocess import call

if __name__ == "__main__":

    fs = 8

    pparams = {}
    pparams['lambda_s'] = np.array([1.0*1j], dtype='complex')
    pparams['lambda_f'] = np.array([50.0*1j, 100.0*1j], dtype='complex')
    pparams['u0'] = 1.0
    swparams = {}
    #
    #
    #
    #swparams['collocation_class'] = collclass.CollGaussLobatto
    #swparams['collocation_class'] = collclass.CollGaussLegendre
    swparams['collocation_class'] = collclass.CollGaussRadau_Right

    nodes_v = np.arange(2,10)
    specrad = np.zeros((3,np.size(nodes_v)))
    norm    = np.zeros((3,np.size(nodes_v)))
    for i in range(0,np.size(nodes_v)):
      swparams['num_nodes'] = nodes_v[i]
      #
      # ...this is functionality copied from test_imexsweeper. Ideally, it should be available in one place.
      #
      step = stepclass.step(params={})
      L = lvl.level(problem_class=swfw_scalar, problem_params=pparams, dtype_u=mesh, dtype_f=rhs_imex_mesh, sweeper_class=imex, sweeper_params=swparams, level_params={}, hook_class=hookclass.hooks, id="stability")
      step.register_level(L)
      step.status.dt   = 1.0
      step.status.time = 0.0
      u0 = step.levels[0].prob.u_exact(step.status.time)
      step.init_step(u0)
      nnodes  = step.levels[0].sweep.coll.num_nodes
      level   = step.levels[0]
      problem = level.prob
      QE = level.sweep.QE[1:,1:]
      QI = level.sweep.QI[1:,1:]
      Q  = level.sweep.coll.Qmat[1:,1:]

      dt = step.status.dt
      
      for j in range(0,2):
        LHS = np.eye(nnodes) - step.status.dt*( problem.lambda_f[j]*QI + problem.lambda_s[0]*QE )
        RHS = step.status.dt*( (problem.lambda_f[j]+problem.lambda_s[0])*Q - (problem.lambda_f[j]*QI + problem.lambda_s[0]*QE) )
        evals, evecs = np.linalg.eig( np.linalg.inv(LHS).dot(RHS) )
        specrad[j+1,i] = np.linalg.norm( evals, np.inf )
        norm[j+1,i]    = np.linalg.norm( np.linalg.inv(LHS).dot(RHS), np.inf )

      if swparams['collocation_class']==collclass.CollGaussLobatto:
        # For Lobatto nodes, first column and row are all zeros, since q_1 = q_0; hence remove them
        QI = QI[1:,1:]
        Q  = Q[1:,1:]
        # Eigenvalue of error propagation matrix in stiff limit: E = I - inv(QI)*Q
        evals, evecs = np.linalg.eig( np.eye(nnodes-1) - np.linalg.inv(QI).dot(Q) )
        norm[0,i] = np.linalg.norm( np.eye(nnodes-1) - np.linalg.inv(QI).dot(Q), np.inf )
      else:
        evals, evecs = np.linalg.eig( np.eye(nnodes) - np.linalg.inv(QI).dot(Q) )
        norm[0,i] = np.linalg.norm( np.eye(nnodes) - np.linalg.inv(QI).dot(Q), np.inf )
      specrad[0,i] = np.linalg.norm( evals, np.inf )
      

  ### Plot result
    rcParams['figure.figsize'] = 2.5, 2.5
    fig = plt.figure()
    plt.plot(nodes_v, specrad[0,:], 'rd-', markersize=fs-2, label=r'$\lambda_{\rm fast} = \infty$')
    plt.plot(nodes_v, specrad[1,:], 'bo-', markersize=fs-2, label=r'$\lambda_{\rm fast} = %2.0f $' % problem.lambda_f[0].imag)
    plt.plot(nodes_v, specrad[2,:], 'gs-', markersize=fs-2, label=r'$\lambda_{\rm fast} = %2.0f $' % problem.lambda_f[1].imag)
    plt.xlabel(r'Number of nodes $M$', fontsize=fs)
    plt.ylabel(r'Spectral radius  $\sigma\left( \mathbf{E} \right)$', fontsize=fs, labelpad=2)
    #plt.title(r'$\Delta t \left| \lambda_{\rm slow} \right|$ = %2.1f' % step.status.dt*abs(problem.lambda_s[0]), fontsize=fs)
    plt.legend(loc='lower right', fontsize=fs, prop={'size':fs})
    plt.xlim([np.min(nodes_v), np.max(nodes_v)])
    plt.ylim([0, 1.0])
    plt.yticks(fontsize=fs)
    plt.xticks(fontsize=fs)
    #plt.show()
    filename = 'stifflimit-specrad.pdf'
    fig.savefig(filename,bbox_inches='tight')
    call(["pdfcrop", filename, filename])

    fig = plt.figure()
    plt.plot(nodes_v, norm[0,:], 'rd-', markersize=fs-2, label=r'$\lambda_{\rm fast} = \infty$')
    plt.plot(nodes_v, norm[1,:], 'bo-', markersize=fs-2, label=r'$\lambda_{\rm fast} = %2.0f $' % problem.lambda_f[0].imag)
    plt.plot(nodes_v, norm[2,:], 'gs-', markersize=fs-2, label=r'$\lambda_{\rm fast} = %2.0f $' % problem.lambda_f[1].imag)
    plt.xlabel(r'Number of nodes $M$', fontsize=fs)
    plt.ylabel(r'Norm  $\left|| \mathbf{E} \right||_{\infty}$', fontsize=fs, labelpad=2)
    #plt.title(r'$\Delta t \left| \lambda_{\rm slow} \right|$ = %2.1f' % step.status.dt*abs(problem.lambda_s[0]), fontsize=fs)
    plt.legend(loc='lower right', fontsize=fs, prop={'size':fs})
    plt.xlim([np.min(nodes_v), np.max(nodes_v)])
    #plt.ylim([0, 1.0])
    plt.yticks(fontsize=fs)
    plt.xticks(fontsize=fs)
    #plt.show()
    filename = 'stifflimit-norm.pdf'
    fig.savefig(filename,bbox_inches='tight')
    call(["pdfcrop", filename, filename])
