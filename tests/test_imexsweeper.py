from pySDC import Level as lvl
from pySDC import Hooks as hookclass
from pySDC import CollocationClasses as collclass
from pySDC import Step as stepclass

from pySDC.datatype_classes.complex_mesh import mesh, rhs_imex_mesh
from pySDC.sweeper_classes.imex_1st_order import imex_1st_order as imex
from examples.SWFW.ProblemClass import swfw_scalar 

from nose.tools import *
import unittest
import numpy as np

class TestImexSweeper(unittest.TestCase):

  #
  #
  #
  def setUp(self):
    self.pparams = {}
    self.pparams['lambda_s'] = np.array([-0.1*1j], dtype='complex')
    self.pparams['lambda_f'] = np.array([-1.0*1j], dtype='complex')
    self.pparams['u0'] = 1.0
    self.swparams = {}
    self.swparams['collocation_class'] = collclass.CollGaussLobatto
    self.swparams['num_nodes'] = 2

  #
  #
  #
  def test_caninstantiate(self):
    L = lvl.level(problem_class=swfw_scalar, problem_params=self.pparams, dtype_u=mesh, dtype_f=rhs_imex_mesh, sweeper_class=imex, sweeper_params=self.swparams, level_params={}, hook_class=hookclass.hooks, id="imextest")

  #
  #
  #
  def test_canregisterlevel(self):
    step = stepclass.step(params={})
    L = lvl.level(problem_class=swfw_scalar, problem_params=self.pparams, dtype_u=mesh, dtype_f=rhs_imex_mesh, sweeper_class=imex, sweeper_params=self.swparams, level_params={}, hook_class=hookclass.hooks, id="imextest")
    step.register_level(L)

  #
  #
  #
  def test_canrunsweep(self):

    step = stepclass.step(params={})
    L = lvl.level(problem_class=swfw_scalar, problem_params=self.pparams, dtype_u=mesh, dtype_f=rhs_imex_mesh, sweeper_class=imex, sweeper_params=self.swparams, level_params={}, hook_class=hookclass.hooks, id="imextest")
    step.register_level(L)
    step.status.dt   = 1.0
    step.status.time = 0.0
    nnodes  = step.levels[0].sweep.coll.num_nodes
    u0 = step.levels[0].prob.u_exact(step.status.time)
    step.init_step(u0)
    step.levels[0].sweep.predict()
    u0full = np.array([ step.levels[0].u[l].values.flatten() for l in range(1,nnodes+1) ])

    step.levels[0].sweep.update_nodes()
    assert step.levels[0].uend is None, "uend should be None previous to running compute_end_point"
    step.levels[0].sweep.compute_end_point()
    #print "Sweep: %s" % step.levels[0].uend.values

  #
  # Make sure a sweep in matrix form is equal to a sweep in node-to-node form
  #
  def test_sweepequalmatrix(self):

    step = stepclass.step(params={})
    L = lvl.level(problem_class=swfw_scalar, problem_params=self.pparams, dtype_u=mesh, dtype_f=rhs_imex_mesh, sweeper_class=imex, sweeper_params=self.swparams, level_params={}, hook_class=hookclass.hooks, id="imextest")
    step.register_level(L)
    step.status.dt   = 1.0
    step.status.time = 0.0
    nnodes  = step.levels[0].sweep.coll.num_nodes
    u0 = step.levels[0].prob.u_exact(step.status.time)
    step.init_step(u0)
    step.levels[0].sweep.predict()
    u0full = np.array([ step.levels[0].u[l].values.flatten() for l in range(1,nnodes+1) ])

    # Perform node-to-node SDC sweep
    step.levels[0].sweep.update_nodes()

    # Build SDC sweep matrix
    level   = step.levels[0]
    problem = level.prob
    QE = level.sweep.QE[1:,1:]
    QI = level.sweep.QI[1:,1:]
    Q  = level.sweep.coll.Qmat[1:,1:]
    dt = step.status.dt
    LHS = np.eye(nnodes) - step.status.dt*( problem.lambda_f[0]*QI + problem.lambda_s[0]*QE )
    RHS = step.status.dt*( (problem.lambda_f[0]+problem.lambda_s[0])*Q - (problem.lambda_f[0]*QI + problem.lambda_s[0]*QE) )
    unew = np.linalg.inv(LHS).dot( u0full + RHS.dot(u0full) )
    usweep = np.array([ level.u[l].values.flatten() for l in range(1,nnodes+1) ])
    assert np.linalg.norm(unew - usweep, np.infty)<1e-14, "Single SDC sweeps in matrix and node-to-node formulation yield different results"        

  #
  #
  #
  def test_updateformula(self):

    step = stepclass.step(params={})
    L = lvl.level(problem_class=swfw_scalar, problem_params=self.pparams, dtype_u=mesh, dtype_f=rhs_imex_mesh, sweeper_class=imex, sweeper_params=self.swparams, level_params={}, hook_class=hookclass.hooks, id="imextest")
    step.register_level(L)
    step.status.dt   = 1.0
    step.status.time = 0.0
    nnodes  = step.levels[0].sweep.coll.num_nodes
    u0 = step.levels[0].prob.u_exact(step.status.time)
    step.init_step(u0)
    step.levels[0].sweep.predict()
    u0full = np.array([ step.levels[0].u[l].values.flatten() for l in range(1,nnodes+1) ])

    # Build SDC sweep matrix
    level   = step.levels[0]
    problem = level.prob
    QE = level.sweep.QE[1:,1:]
    QI = level.sweep.QI[1:,1:]
    Q  = level.sweep.coll.Qmat[1:,1:]

    # Perform update step in sweeper
    step.levels[0].sweep.update_nodes()
    ustages = np.array([ step.levels[0].u[l].values.flatten() for l in range(1,nnodes+1) ])

    step.levels[0].sweep.compute_end_point()
    uend_sweep = step.levels[0].uend.values
    uend_mat   = u0.values + step.status.dt*step.levels[0].sweep.coll.weights.dot(ustages*(problem.lambda_s[0] + problem.lambda_f[0]))
    assert np.linalg.norm(uend_sweep - uend_mat, np.infty)<1e-14, "Update formula in sweeper gives different result than matrix update formula"

  #
  # Compute the exact collocation solution by matrix inversion and make sure it is a fixed point
  #
  def test_collocationinvariant(self):

    step = stepclass.step(params={})
    L = lvl.level(problem_class=swfw_scalar, problem_params=self.pparams, dtype_u=mesh, dtype_f=rhs_imex_mesh, sweeper_class=imex, sweeper_params=self.swparams, level_params={}, hook_class=hookclass.hooks, id="imextest")
    step.register_level(L)
    step.status.dt   = 1.0
    step.status.time = 0.0
    nnodes  = step.levels[0].sweep.coll.num_nodes
    u0 = step.levels[0].prob.u_exact(step.status.time)
    step.init_step(u0)
    step.levels[0].sweep.predict()
    u0full = np.array([ step.levels[0].u[l].values.flatten() for l in range(1,nnodes+1) ])

    # Build SDC sweep matrix
    level   = step.levels[0]
    problem = level.prob
    QE = level.sweep.QE[1:,1:]
    QI = level.sweep.QI[1:,1:]
    Q  = level.sweep.coll.Qmat[1:,1:]
    
    Mcoll = np.eye(nnodes) - step.status.dt*Q*(problem.lambda_s[0] + problem.lambda_f[0])
    ucoll = np.linalg.inv(Mcoll).dot(u0full)
    for l in range(0,nnodes):
      step.levels[0].u[l+1].values = ucoll[l]
      step.levels[0].f[l+1].impl.values = problem.lambda_f[0]*ucoll[l]
      step.levels[0].f[l+1].expl.values = problem.lambda_s[0]*ucoll[l]

    # Perform node-to-node SDC sweep
    step.levels[0].sweep.update_nodes()

    LHS = np.eye(nnodes) - step.status.dt*( problem.lambda_f[0]*QI + problem.lambda_s[0]*QE )
    RHS = step.status.dt*( (problem.lambda_f[0]+problem.lambda_s[0])*Q - (problem.lambda_f[0]*QI + problem.lambda_s[0]*QE) )
    unew = np.linalg.inv(LHS).dot( u0full + RHS.dot(ucoll) )
    assert np.linalg.norm( unew - ucoll, np.infty )<1e-14, "Collocation solution not invariant under matrix SDC sweep"
    unew_sweep = np.array([ step.levels[0].u[l].values.flatten() for l in range(1,nnodes+1) ])
    assert np.linalg.norm( unew_sweep - ucoll, np.infty )<1e-14, "Collocation solution not invariant under node-to-node sweep"

  #
  #
  #
  def test_canrunmatrixsweep(self):
    step = stepclass.step(params={})
    L = lvl.level(problem_class=swfw_scalar, problem_params=self.pparams, dtype_u=mesh, dtype_f=rhs_imex_mesh, sweeper_class=imex, sweeper_params=self.swparams, level_params={}, hook_class=hookclass.hooks, id="imextest")
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
    
    P  = np.eye(nnodes) - step.status.dt*problem.lambda_s[0]*QE - step.status.dt*problem.lambda_f[0]*QI

    Pinv = np.linalg.inv(P)
    M  = np.eye(nnodes) - step.status.dt*( problem.lambda_s[0] + problem.lambda_f[0] )*Q
    #M = step.status.dt*( (problem.lambda_s[0]+problem.lambda_f[0])*Q - problem.lambda_f[0]*QI - problem.lambda_s[0]*QE )
    #print QI
    #print P    
    #print Pinv
    #print M
    #level.sweep.predict()
    #u0full = np.array([ level.u[l].values.flatten() for l in range(1,nnodes+1) ])
    #ufull  = u0full + Pinv.dot( u0full ) - Pinv.dot( M.dot(u0full) )
    #print u0.values
    #print Pinv.dot(u0full)
    #print Pinv.dot(M)
    #print Pinv.dot(M.dot(u0full))
    #ufull = Pinv.dot(M.dot(u0full)) + Pinv.dot(u0full)
    #ufull  = np.linalg.inv(M).dot(u0full)
    #print ufull
    #uend   = u0.values + step.status.dt*level.sweep.coll.weights.dot( (problem.lambda_f[0]+problem.lambda_s[0])*ufull )
    #print "Matrix: %s" % uend
