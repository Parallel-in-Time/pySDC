from subprocess import call

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from pylab import rcParams

from examples.fwsw.ProblemClass import swfw_scalar
from implementations.datatype_classes import mesh, rhs_imex_mesh
from implementations.sweeper_classes.imex_1st_order import imex_1st_order as imex
from pySDC import CollocationClasses as collclass
from pySDC import Hooks as hookclass
from pySDC import Level as lvl
from pySDC import Step as stepclass

if __name__ == "__main__":

    N_s = 100
    N_f = 400
    
    lam_s_max = 5.0
    lam_f_max = 12.0
    lambda_s = 1j*np.linspace(0.0, lam_s_max, N_s)
    lambda_f = 1j*np.linspace(0.0, lam_f_max, N_f)

    pparams = {}
    # the following are not used in the computation
    pparams['lambda_s'] = np.array([0.0])
    pparams['lambda_f'] = np.array([0.0])
    pparams['u0'] = 1.0
    swparams = {}
    ### SET TYPE OF QUADRATURE NODES ###
    swparams['collocation_class'] = collclass.CollGaussLobatto
    #swparams['collocation_class'] = collclass.CollGaussLegendre
    #swparams['collocation_class'] = collclass.CollGaussRadau_Right
    
    ### SET NUMBER OF QUADRATURE NODES ###
    swparams['num_nodes'] = 4
    
    ### SET NUMBER OF ITERATIONS - SET K=0 FOR COLLOCATION SOLUTION ###
    K = 5
    
    do_coll_update = True
    
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
    Q  = level.sweep.coll.Qmat[1:,1:]

    stab = np.zeros((N_f, N_s), dtype='complex')

    for i in range(0,N_s):
      for j in range(0,N_f):
        lambda_fast = lambda_f[j]
        lambda_slow = lambda_s[i]
        if K is not 0:
          lambdas = [ lambda_fast, lambda_slow ]
          LHS, RHS = level.sweep.get_scalar_problems_sweeper_mats( lambdas = lambdas )
          Mat_sweep = level.sweep.get_scalar_problems_manysweep_mat( nsweeps = K, lambdas = lambdas )
        else:
          # Compute stability function of collocation solution
          Mat_sweep = np.linalg.inv(np.eye(nnodes)-step.status.dt*(lambda_fast + lambda_slow)*Q)
        if do_coll_update:
          stab_fh = 1.0 + (lambda_fast + lambda_slow)*level.sweep.coll.weights.dot(Mat_sweep.dot(np.ones(nnodes)))
        else:
          q = np.zeros(nnodes)
          q[nnodes-1] = 1.0
          stab_fh = q.dot(Mat_sweep.dot(np.ones(nnodes)))
          #
        stab[j,i] = stab_fh

    ###
    rcParams['figure.figsize'] = 1.5, 1.5
    fs = 8
    fig  = plt.figure()
    #pcol = plt.pcolor(lambda_s.imag, lambda_f.imag, np.absolute(stab), vmin=0.99, vmax=2.01)
    #pcol.set_edgecolor('face')
    levels = np.array([0.25, 0.5, 0.75, 0.9, 1.1])
#    levels = np.array([1.0])
    CS1 = plt.contour(lambda_s.imag, lambda_f.imag, np.absolute(stab), levels, colors='k', linestyles='dashed')
    CS2 = plt.contour(lambda_s.imag, lambda_f.imag, np.absolute(stab), [1.0],  colors='k')
    # Set markers at points used in plot_stab_vs_k
    plt.plot(4, 10, 'x', color='k', markersize=fs-4)
    plt.plot(1, 10, 'x', color='k', markersize=fs-4)
    plt.clabel(CS1, inline=True, fmt='%3.2f', fontsize=fs-2)
    manual_locations = [(1.5, 2.5)]
    if K>0: # for K=0 and no 1.0 isoline, this crashes Matplotlib for somer reason
      plt.clabel(CS2, inline=True, fmt='%3.2f', fontsize=fs-2, manual=manual_locations)
    plt.gca().add_patch(Polygon([[0, 0], [lam_s_max,0], [lam_s_max,lam_s_max]], visible=True, fill=True, facecolor='.75',edgecolor='k', linewidth=1.0,  zorder=11))
    #plt.plot([0, 2], [0, 2], color='k', linewidth=1, zorder=12)
    plt.gca().set_xticks(np.arange(0, int(lam_s_max)+1))
    plt.gca().set_yticks(np.arange(0, int(lam_f_max)+2, 2))
    plt.gca().tick_params(axis='both', which='both', labelsize=fs)
    plt.xlim([0.0, lam_s_max])
    plt.ylim([0.0, lam_f_max])
    plt.xlabel('$\Delta t \lambda_{slow}$', fontsize=fs, labelpad=0.0)
    plt.ylabel('$\Delta t \lambda_{fast}$', fontsize=fs, labelpad=0.0)
    plt.title(r'$M=%1i$, $K=%1i$' % (swparams['num_nodes'],K), fontsize=fs)
    #plt.show()
    filename = 'stability-K'+str(K)+'-M'+str(swparams['num_nodes'])+'.pdf'
    fig.savefig(filename, bbox_inches='tight')
    call(["pdfcrop", filename, filename])

