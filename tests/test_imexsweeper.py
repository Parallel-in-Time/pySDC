from pySDC import Level as lvl
from pySDC import Hooks as hookclass
from pySDC import CollocationClasses as collclass
from pySDC import Step as stepclass

from pySDC.datatype_classes.complex_mesh import mesh, rhs_imex_mesh
from pySDC.sweeper_classes.imex_1st_order import imex_1st_order as imex
from examples.fwsw.ProblemClass import swfw_scalar

from nose.tools import *
import unittest
import numpy as np

class TestImexSweeper(unittest.TestCase):

  #
  # Some auxiliary functions which are not tests themselves
  #
  def setupLevelStepProblem(self):
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
    return step, level, problem, nnodes
  
  def setupQMatrices(self, level):
    QE = level.sweep.QE[1:,1:]
    QI = level.sweep.QI[1:,1:]
    Q  = level.sweep.coll.Qmat[1:,1:]
    return QE, QI, Q

  def setupSweeperMatrices(self, step, level, problem):
    nnodes  = step.levels[0].sweep.coll.num_nodes
    # Build SDC sweep matrix
    QE, QI, Q = self.setupQMatrices(level)
    dt = step.status.dt
    LHS = np.eye(nnodes) - step.status.dt*( problem.lambda_f[0]*QI + problem.lambda_s[0]*QE )
    RHS = step.status.dt*( (problem.lambda_f[0]+problem.lambda_s[0])*Q - (problem.lambda_f[0]*QI + problem.lambda_s[0]*QE) )
    return LHS, RHS

  #
  # General setUp function used by all tests
  #
  def setUp(self):
    self.pparams = {}
    self.pparams['lambda_s'] = np.array([-0.1*1j], dtype='complex')
    self.pparams['lambda_f'] = np.array([-1.0*1j], dtype='complex')
    self.pparams['u0'] = np.random.rand()
    self.swparams = {}
    self.swparams['collocation_class'] = collclass.CollGaussLobatto
    self.swparams['num_nodes'] = 2+np.random.randint(5)

  # ***************
  # **** TESTS ****
  # ***************

  #
  # Check that a level object can be instantiated
  #
  def test_caninstantiate(self):
    L = lvl.level(problem_class=swfw_scalar, problem_params=self.pparams, dtype_u=mesh, dtype_f=rhs_imex_mesh, sweeper_class=imex, sweeper_params=self.swparams, level_params={}, hook_class=hookclass.hooks, id="imextest")
    assert isinstance(L.sweep, imex), "sweeper in generated level is not an object of type imex"

  #
  # Check that a level object can be registered in a step object (needed as prerequiste to execute update_nodes
  #
  def test_canregisterlevel(self):
    step = stepclass.step(params={})
    L = lvl.level(problem_class=swfw_scalar, problem_params=self.pparams, dtype_u=mesh, dtype_f=rhs_imex_mesh, sweeper_class=imex, sweeper_params=self.swparams, level_params={}, hook_class=hookclass.hooks, id="imextest")
    step.register_level(L)
    # At this point, it should not be possible to actually execute functions of the sweeper because the parameters set in setupLevelStepProblem are not yet initialised
    with self.assertRaises(Exception):
      step.sweep.predict()
    with self.assertRaises(Exception):
      step.sweep.update_nodes()
    with self.assertRaises(Exception):
      step.sweep.compute_end_point()

  #
  # Check that the sweeper functions update_nodes and compute_end_point can be executed
  #
  def test_canrunsweep(self):

    # After running setupLevelStepProblem, the functions predict, update_nodes and compute_end_point should run
    step, level, problem, nnodes = self.setupLevelStepProblem()   
    assert level.u[0] is not None, "After init_step, level.u[0] should no longer be of type None" 
    assert level.u[1] is None, "Before predict, level.u[1] and following should be of type None"
    level.sweep.predict()
    # Should now be able to run update nodes
    level.sweep.update_nodes()
    assert level.uend is None, "uend should be None previous to running compute_end_point"
    level.sweep.compute_end_point()
    assert level.uend is not None, "uend still None after running compute_end_point"

  #
  # Make sure a sweep in matrix form is equal to a sweep in node-to-node form
  #
  def test_sweepequalmatrix(self):

    step, level, problem, nnodes = self.setupLevelStepProblem()
    step.levels[0].sweep.predict()
    u0full = np.array([ level.u[l].values.flatten() for l in range(1,nnodes+1) ])

    # Perform node-to-node SDC sweep
    level.sweep.update_nodes()

    LHS, RHS = self.setupSweeperMatrices(step, level, problem)

    unew = np.linalg.inv(LHS).dot( u0full + RHS.dot(u0full) )
    usweep = np.array([ level.u[l].values.flatten() for l in range(1,nnodes+1) ])
    assert np.linalg.norm(unew - usweep, np.infty)<1e-14, "Single SDC sweeps in matrix and node-to-node formulation yield different results"        

  #
  # Make sure the implemented update formula matches the matrix update formula
  #
  def test_updateformula(self):

    if (self.swparams['collocation_class']==collclass.CollGaussLobatto):
      raise unittest.SkipTest("Needs fix of issue #52 before passing for Gauss Lobatto nodes")

    step, level, problem, nnodes = self.setupLevelStepProblem()
    level.sweep.predict()
    u0full = np.array([ level.u[l].values.flatten() for l in range(1,nnodes+1) ])

    # Perform update step in sweeper
    level.sweep.update_nodes()
    ustages = np.array([ level.u[l].values.flatten() for l in range(1,nnodes+1) ])

    # Compute end value through provided function
    level.sweep.compute_end_point()
    uend_sweep = level.uend.values
    # Compute end value from matrix formulation
    uend_mat   = self.pparams['u0'] + step.status.dt*level.sweep.coll.weights.dot(ustages*(problem.lambda_s[0] + problem.lambda_f[0]))
    assert np.linalg.norm(uend_sweep - uend_mat, np.infty)<1e-14, "Update formula in sweeper gives different result than matrix update formula"

  #
  # Compute the exact collocation solution by matrix inversion and make sure it is a fixed point
  #
  def test_collocationinvariant(self):

    step, level, problem, nnodes = self.setupLevelStepProblem()
    level.sweep.predict()
    u0full = np.array([ level.u[l].values.flatten() for l in range(1,nnodes+1) ])
    
    QE, QI, Q = self.setupQMatrices(level)

    # Build collocation matrix
    Mcoll = np.eye(nnodes) - step.status.dt*Q*(problem.lambda_s[0] + problem.lambda_f[0])

    # Solve collocation problem directly
    ucoll = np.linalg.inv(Mcoll).dot(u0full)
    
    # Put stages of collocation solution into level
    for l in range(0,nnodes):
      level.u[l+1].values = ucoll[l]
      level.f[l+1].impl.values = problem.lambda_f[0]*ucoll[l]
      level.f[l+1].expl.values = problem.lambda_s[0]*ucoll[l]

    # Perform node-to-node SDC sweep
    level.sweep.update_nodes()

    # Build matrices for matrix formulation of sweep
    LHS = np.eye(nnodes) - step.status.dt*( problem.lambda_f[0]*QI + problem.lambda_s[0]*QE )
    RHS = step.status.dt*( (problem.lambda_f[0]+problem.lambda_s[0])*Q - (problem.lambda_f[0]*QI + problem.lambda_s[0]*QE) )
    # Make sure both matrix and node-to-node sweep leave collocation unaltered
    unew = np.linalg.inv(LHS).dot( u0full + RHS.dot(ucoll) )
    assert np.linalg.norm( unew - ucoll, np.infty )<1e-14, "Collocation solution not invariant under matrix SDC sweep"
    unew_sweep = np.array([ level.u[l].values.flatten() for l in range(1,nnodes+1) ])
    assert np.linalg.norm( unew_sweep - ucoll, np.infty )<1e-14, "Collocation solution not invariant under node-to-node sweep"


  #
  # Make sure that K node-to-node sweeps give the same result as K sweeps in matrix form and the single matrix formulation for K sweeps
  #
  def test_manysweepsequalmatrix(self):
    step, level, problem, nnodes = self.setupLevelStepProblem()
    step.levels[0].sweep.predict()
    u0full = np.array([ level.u[l].values.flatten() for l in range(1,nnodes+1) ])

    # Perform K node-to-node SDC sweep
    K = 1 + np.random.randint(6)
    for i in range(0,K):
      level.sweep.update_nodes()
    usweep = np.array([ level.u[l].values.flatten() for l in range(1,nnodes+1) ])

    LHS, RHS = self.setupSweeperMatrices(step, level, problem)
    unew = u0full
    for i in range(0,K):
      unew = np.linalg.inv(LHS).dot( u0full + RHS.dot(unew) )
    
    assert np.linalg.norm(unew - usweep, np.infty)<1e-14, "Doing multiple node-to-node sweeps yields different result than same number of matrix-form sweeps"   
    
    # Build single matrix representing K sweeps    
    Pinv = np.linalg.inv(LHS)
    Mat_sweep = np.linalg.matrix_power(Pinv.dot(RHS), K)
    for i in range(0,K):
      Mat_sweep = Mat_sweep + np.linalg.matrix_power(Pinv.dot(RHS),i).dot(Pinv)
    usweep_onematrix = Mat_sweep.dot(u0full)
    assert np.linalg.norm( usweep_onematrix - usweep, np.infty )<1e-14, "Single-matrix multiple sweep formulation yields different result than multiple sweeps in node-to-node or matrix form form"
    
  #
  # Make sure that update function for K sweeps computed from K-sweep matrix gives same result as K sweeps in node-to-node form plus compute_end_point
  #
  def test_maysweepupdate(self):

    if (self.swparams['collocation_class']==collclass.CollGaussLobatto):
      raise unittest.SkipTest("Needs fix of issue #52 before passing for Gauss Lobatto nodes")

    step, level, problem, nnodes = self.setupLevelStepProblem()
    step.levels[0].sweep.predict()
    u0full = np.array([ level.u[l].values.flatten() for l in range(1,nnodes+1) ])

    # Perform K node-to-node SDC sweep
    K = 1 + np.random.randint(6)
    for i in range(0,K):
      level.sweep.update_nodes()
    # Fetch final value
    level.sweep.compute_end_point()
    uend_sweep = level.uend.values

    LHS, RHS = self.setupSweeperMatrices(step, level, problem)
    # Build single matrix representing K sweeps    
    Pinv = np.linalg.inv(LHS)
    Mat_sweep = np.linalg.matrix_power(Pinv.dot(RHS), K)
    for i in range(0,K):
      Mat_sweep = Mat_sweep + np.linalg.matrix_power(Pinv.dot(RHS),i).dot(Pinv)
    # Now build update function
    update = 1.0 + (problem.lambda_s[0] + problem.lambda_f[0])*level.sweep.coll.weights.dot(Mat_sweep.dot(np.ones(nnodes)))
    # Multiply u0 by value of update function to get end value directly
    uend_matrix = update*self.pparams['u0']
    assert abs(uend_matrix - uend_sweep)<1e-14, "Node-to-node sweep plus update yields different result than update function computed through K-sweep matrix"
