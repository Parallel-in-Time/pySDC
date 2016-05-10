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

def get_stab_function(LHS, RHS, Kmax, lambd, do_coll_update):
  Pinv = np.linalg.inv(LHS)
  Mat_sweep = np.linalg.matrix_power(Pinv.dot(RHS), Kmax)
  for l in range(0,Kmax):
    Mat_sweep = Mat_sweep + np.linalg.matrix_power(Pinv.dot(RHS),k).dot(Pinv)
  if do_coll_update:
    stab_fh = 1.0 + lambd*level.sweep.coll.weights.dot(Mat_sweep.dot(np.ones(nnodes)))
  else:
    q = np.zeros(nnodes)
    q[nnodes-1] = 1.0
    stab_fh = q.dot(Mat_sweep.dot(np.ones(nnodes)))
  return stab_fh

if __name__ == "__main__":
  mvals = [2, 4]
  kvals = np.arange(1,15)
  lambdaratio = [2, 10, 50]
  stabval = np.zeros((np.size(mvals), np.size(lambdaratio), np.size(kvals)))
  
  for i in range(0,np.size(mvals)):
    pparams = {}
    # the following are not used in the computation
    pparams['lambda_s'] = np.array([0.0])
    pparams['lambda_f'] = np.array([0.0])
    pparams['u0'] = 1.0
    swparams = {}
    swparams['collocation_class'] = collclass.CollGaussRadau_Right
    swparams['num_nodes'] = mvals[i]
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
  
    QE = level.sweep.QE[1:,1:]
    QI = level.sweep.QI[1:,1:]
    Q  = level.sweep.coll.Qmat[1:,1:]

    for j in range(0,np.size(lambdaratio)):
      lambda_slow = 1j
      lambda_fast = lambdaratio[j]*lambda_slow

      LHS = np.eye(nnodes) - step.status.dt*( lambda_fast*QI + lambda_slow*QE )
      RHS = step.status.dt*( (lambda_fast+lambda_slow)*Q - (lambda_fast*QI + lambda_slow*QE) )

      for k in range(0, np.size(kvals)):
        Kmax = kvals[k]
        stab_fh = get_stab_function(LHS, RHS, kvals[k], lambda_fast+lambda_slow, do_coll_update)
        stabval[i,j,k] = np.absolute(stab_fh)

  fig = plt.figure()
  plt.plot(kvals, stabval[0,0,:], '-', color='b')
  plt.plot(kvals, stabval[0,1,:], '-', color='r')
  plt.plot(kvals, stabval[0,2,:], '-', color='g')
  #plt.plot(kvals, stabval[1,0,:], '-',  color='r')
  #plt.plot(kvals, stabval[1,1,:], '--', color='r')
  #plt.plot(kvals, stabval[1,2,:], '-.', color='r')
  plt.show()
