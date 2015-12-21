from pySDC import Level as lvl
from pySDC import Hooks as hookclass
from pySDC import CollocationClasses as collclass
from pySDC import Step as stepclass

from pySDC.datatype_classes.complex_mesh import mesh, rhs_imex_mesh
from pySDC.sweeper_classes.imex_1st_order import imex_1st_order as imex
from examples.fwsw.ProblemClass import swfw_scalar 
import numpy as np

from pylab import rcParams
import matplotlib.pyplot as plt
from subprocess import call

if __name__ == "__main__":

    N_s = 100
    N_f = 400

    lambda_s = 1j*np.linspace(0.0, 2.0, N_s)
    lambda_f = 1j*np.linspace(2.0, 8.0, N_f)

    pparams = {}
    # the following are not used in the computation
    pparams['lambda_s'] = np.array([0.0])
    pparams['lambda_f'] = np.array([0.0])
    pparams['u0'] = 1.0
    swparams = {}
    swparams['collocation_class'] = collclass.CollGaussLobatto
    swparams['num_nodes'] = 2
    K = 2
    
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

    stab = np.zeros((N_f, N_s), dtype='complex')

    for i in range(0,N_s):
      for j in range(0,N_f):
        lambda_fast = lambda_f[j]
        lambda_slow = lambda_s[i]
        LHS = np.eye(nnodes) - step.status.dt*( lambda_fast*QI + lambda_slow*QE )
        RHS = step.status.dt*( (lambda_fast+lambda_slow)*Q - (lambda_fast*QI + lambda_slow*QE) )

        Pinv = np.linalg.inv(LHS)
        Mat_sweep = np.linalg.matrix_power(Pinv.dot(RHS), K)
        for k in range(0,K):
          Mat_sweep = Mat_sweep + np.linalg.matrix_power(Pinv.dot(RHS),k).dot(Pinv)
          stab_fh = 1.0 + (lambda_fast + lambda_slow)*level.sweep.coll.weights.dot(Mat_sweep.dot(np.ones(nnodes)))
        stab[j,i] = stab_fh

    ###
    rcParams['figure.figsize'] = 2.5, 2.5
    fs = 8
    fig  = plt.figure()
    #pcol = plt.pcolor(lambda_s.imag, lambda_f.imag, np.absolute(stab), vmin=0.99, vmax=2.01)
    #pcol.set_edgecolor('face')
    levels = np.array([0.95, 0.99, 1.01, 1.05])
#    levels = np.array([1.0])
    CS1 = plt.contour(lambda_s.imag, lambda_f.imag, np.absolute(stab), levels, colors='k',linestyles='dashed')
    CS2 = plt.contour(lambda_s.imag, lambda_f.imag, np.absolute(stab), [1.0], colors='k')
    plt.clabel(CS1, fontsize=fs-2)
    plt.clabel(CS2, fontsize=fs-2)
    #plt.plot([0, 2], [0, 2], color='k', linewidth=1)
    plt.gca().set_xticks([0.0, 1.0, 2.0, 3.0])
    plt.gca().tick_params(axis='both', which='both', labelsize=fs)
    plt.xlim([np.min(lambda_s.imag), np.max(lambda_s.imag)])
    plt.xlabel('$\Delta t \lambda_{slow}$', fontsize=fs, labelpad=0.0)
    plt.ylabel('$\Delta t \lambda_{fast}$', fontsize=fs)
    filename = 'sdc-fwsw-stability-K'+str(K)+'-M'+str(swparams['num_nodes'])+'.pdf'
    fig.savefig(filename, bbox_inches='tight')
    call(["pdfcrop", filename, filename])

