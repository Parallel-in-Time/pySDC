from subprocess import call

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter
from pylab import rcParams

from pySDC_implementations.datatype_classes import mesh, rhs_imex_mesh
from pySDC_implementations.problem_classes.FastWaveSlowWave_Scalar import swfw_scalar
from pySDC_implementations.sweeper_classes.imex_1st_order import imex_1st_order as imex
from pySDC_core import CollocationClasses as collclass
from pySDC_core import Hooks as hookclass
from pySDC_core import Level as lvl
from pySDC_core import Step as stepclass

if __name__ == "__main__":
  mvals = [2, 3, 4]
  kvals = np.arange(1,10)
  lambda_fast = 10j

  ### PLOT EITHER FOR lambda_slow = 1 (resolved) OR lambda_slow = 4 (unresolved)
  slow_resolved = False
  if slow_resolved:
    lambda_slow = 1j
  else:
    lambda_slow = 4j
  stabval = np.zeros((np.size(mvals), np.size(kvals)))
  
  for i in range(0,np.size(mvals)):
    pparams = {}
    # the following are not used in the computation
    pparams['lambda_s'] = np.array([0.0])
    pparams['lambda_f'] = np.array([0.0])
    pparams['u0'] = 1.0
    swparams = {}
#    swparams['collocation_class'] = collclass.CollGaussLobatto
#    swparams['collocation_class'] = collclass.CollGaussLegendre
    swparams['collocation_class'] = collclass.CollGaussRadau_Right
    swparams['num_nodes'] = mvals[i]
    do_coll_update = True  

    #
    # ...this is functionality copied from test_imexsweeper. Ideally, it should be available in one place.
    #
    step = stepclass.step(params={})
    L = lvl.level(problem_class=swfw_scalar, problem_params=pparams, dtype_u=mesh, dtype_f=rhs_imex_mesh, sweeper_class=imex, sweeper_params=swparams, level_params={}, hook_class=hookclass.hooks, id="stability")
    step.register_level(L)
    step.status.dt   = 1.0 # Needs to be 1.0, change dt through lambdas
    step.status.time = 0.0
    u0 = step.levels[0].prob.u_exact(step.status.time)
    step.init_step(u0)
    nnodes  = step.levels[0].sweep.coll.num_nodes
    level   = step.levels[0]
    problem = level.prob
  
    QE = level.sweep.QE[1:,1:]
    QI = level.sweep.QI[1:,1:]
    Q  = level.sweep.coll.Qmat[1:,1:]

    LHS, RHS = level.sweep.get_scalar_problems_sweeper_mats( lambdas = [ lambda_fast, lambda_slow ] )

    for k in range(0, np.size(kvals)):
      Kmax = kvals[k]
      Mat_sweep = level.sweep.get_scalar_problems_manysweep_mat( nsweeps = Kmax, lambdas = [ lambda_fast, lambda_slow ] )
      if do_coll_update:
        stab_fh = 1.0 + (lambda_fast + lambda_slow)*level.sweep.coll.weights.dot(Mat_sweep.dot(np.ones(nnodes)))
      else:
        q = np.zeros(nnodes)
        q[nnodes-1] = 1.0
        stab_fh = q.dot(Mat_sweep.dot(np.ones(nnodes)))
      stabval[i,k] = np.absolute(stab_fh)

  rcParams['figure.figsize'] = 2.5, 2.5
  fig = plt.figure()
  fs = 8
  plt.plot(kvals, stabval[0,:], 'o-', color='b', label=("M=%2i" % mvals[0]), markersize=fs-2)
  plt.plot(kvals, stabval[1,:], 's-', color='r', label=("M=%2i" % mvals[1]), markersize=fs-2)
  plt.plot(kvals, stabval[2,:], 'd-', color='g', label=("M=%2i" % mvals[2]), markersize=fs-2)
  plt.plot(kvals, 1.0+0.0*kvals, '--', color='k')
  plt.xlabel('Number of iterations K', fontsize=fs)
  plt.ylabel(r'Modulus of stability function $\left| R \right|$', fontsize=fs)
  plt.ylim([0.0, 1.2])
  if slow_resolved:
    plt.legend(loc='upper right', fontsize=fs, prop={'size':fs})
  else:
    plt.legend(loc='lower left', fontsize=fs, prop={'size':fs})

  plt.gca().get_xaxis().get_major_formatter().labelOnlyBase = False
  plt.gca().get_xaxis().set_major_formatter(ScalarFormatter())
  #plt.show()
  if slow_resolved:
    filename = 'stab_vs_k_resolved.pdf'
  else:
    filename = 'stab_vs_k_unresolved.pdf'
  
  fig.savefig(filename, bbox_inches='tight')
  call(["pdfcrop", filename, filename])
