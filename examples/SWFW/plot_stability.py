from pySDC import Level as lvl
from pySDC import Hooks as hookclass
from pySDC import CollocationClasses as collclass
from pySDC import Step as stepclass

from pySDC.datatype_classes.complex_mesh import mesh, rhs_imex_mesh
from pySDC.sweeper_classes.imex_1st_order import imex_1st_order as imex
from examples.SWFW.ProblemClass import swfw_scalar 
import numpy as np

if __name__ == "__main__":

    pparams = {}
    pparams['lambda_s'] = np.array([-0.0], dtype='complex')
    pparams['lambda_f'] = np.array([-1.0], dtype='complex')
    pparams['u0'] = 1.0
    swparams = {}
    swparams['collocation_class'] = collclass.CollGaussLobatto
    swparams['num_nodes'] = 3
    K = 1
    

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
    LHS = np.eye(nnodes) - step.status.dt*( problem.lambda_f[0]*QI + problem.lambda_s[0]*QE )
    RHS = step.status.dt*( (problem.lambda_f[0]+problem.lambda_s[0])*Q - (problem.lambda_f[0]*QI + problem.lambda_s[0]*QE) )

    Pinv = np.linalg.inv(LHS)
    Mat_sweep = np.linalg.matrix_power(Pinv.dot(RHS), K)
    for i in range(0,K):
      Mat_sweep = Mat_sweep + np.linalg.matrix_power(Pinv.dot(RHS),i).dot(Pinv)
    stab_fh = 1.0 + (pparams['lambda_s'][0] + pparams['lambda_f'][0])*level.sweep.coll.weights.dot(Mat_sweep.dot(np.ones(nnodes)))
    print abs(stab_fh)
