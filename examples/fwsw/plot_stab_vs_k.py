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
from matplotlib.patches import Polygon
from subprocess import call
from matplotlib.ticker import ScalarFormatter

if __name__ == "__main__":
  mvals = [2, 3, 4]
  kvals = np.arange(2,10)
  lambda_fast = 15j
  lambda_slow = 3j
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
  plt.plot(kvals, stabval[0,:], 'o-', color='b', label=("M=%2i" % mvals[0]))
  plt.plot(kvals, stabval[1,:], 's-', color='r', label=("M=%2i" % mvals[1]))
  plt.plot(kvals, stabval[2,:], 'd-', color='g', label=("M=%2i" % mvals[2]))
  plt.plot(kvals, 1.0+0.0*kvals, '--', color='k')
  plt.xlabel('Number of iterations K', fontsize=fs)
  plt.ylabel(r'Modulus of stability function $\left| R \right|$', fontsize=fs)
  plt.ylim([0.0, 1.2])
  plt.legend(loc='lower left', fontsize=fs, prop={'size':fs})
  plt.gca().get_xaxis().get_major_formatter().labelOnlyBase = False
  plt.gca().get_xaxis().set_major_formatter(ScalarFormatter())
  plt.show()

#  filename = 'stablimit-M'+str(mvals[0])+'.pdf'
#  fig.savefig(filename, bbox_inches='tight')
#  call(["pdfcrop", filename, filename])
