import numpy as np
import math
import scipy.sparse.linalg as LA
import scipy.sparse as sp
from ProblemClass import Callback, logging, boussinesq_2d_imex

#
# A diagonally implicit Runge-Kutta method of order 2, 3 or 4
#
class dirk:

  def __init__(self, problem, order):

    assert isinstance(problem, boussinesq_2d_imex), "problem is wrong type of object"
    self.Ndof = np.shape(problem.M)[0]
    self.order = order
    self.logger = logging()
    self.problem = problem

    assert self.order in [2,22,3,4], 'Order must be 2,22,3,4'
    
    if (self.order==2):
      self.nstages = 1
      self.A       = np.zeros((1,1))
      self.A[0,0]  = 0.5
      self.tau     = [0.5]
      self.b       = [1.0]
    
    if (self.order==22):
      self.nstages = 2
      self.A       = np.zeros((2,2))
      self.A[0,0]  = 1.0/3.0
      self.A[1,0]  = 1.0/2.0
      self.A[1,1]  = 1.0/2.0
      
      self.tau     = np.zeros(2)
      self.tau[0]  = 1.0/3.0
      self.tau[1]  = 1.0

      self.b       = np.zeros(2)
      self.b[0]    = 3.0/4.0
      self.b[1]    = 1.0/4.0
    
    
    if (self.order==3):
      self.nstages = 2 
      self.A       = np.zeros((2,2))
      self.A[0,0]  = 0.5 + 1.0/( 2.0*math.sqrt(3.0) )
      self.A[1,0] = -1.0/math.sqrt(3.0)
      self.A[1,1] = self.A[0,0]
      
      self.tau    = np.zeros(2)
      self.tau[0] = 0.5 + 1.0/( 2.0*math.sqrt(3.0) )
      self.tau[1] = 0.5 - 1.0/( 2.0*math.sqrt(3.0) )
      
      self.b     = np.zeros(2)
      self.b[0]  = 0.5
      self.b[1]  = 0.5
      
    if (self.order==4):
      self.nstages = 3
      alpha = 2.0*math.cos(math.pi/18.0)/math.sqrt(3.0)
      
      self.A      = np.zeros((3,3))
      self.A[0,0] = (1.0 + alpha)/2.0
      self.A[1,0] = -alpha/2.0
      self.A[1,1] = self.A[0,0]
      self.A[2,0] = (1.0 + alpha)
      self.A[2,1] =  -(1.0 + 2.0*alpha)
      self.A[2,2] = self.A[0,0]
      
      self.tau    = np.zeros(3)
      self.tau[0] = (1.0 + alpha)/2.0
      self.tau[1] = 1.0/2.0
      self.tau[2] = (1.0 - alpha)/2.0
      
      self.b      = np.zeros(3)
      self.b[0]   = 1.0/(6.0*alpha*alpha)
      self.b[1]   = 1.0 - 1.0/(3.0*alpha*alpha)
      self.b[2]   = 1.0/(6.0*alpha*alpha)
       
    self.stages  = np.zeros((self.nstages,self.Ndof))

  def timestep(self, u0, dt):
    
      uend           = u0
      for i in range(0,self.nstages):  
        
        b = u0
        
        # Compute right hand side for this stage's implicit step
        for j in range(0,i):
          b = b + self.A[i,j]*dt*self.f(self.stages[j,:])
        
        # Implicit solve for current stage    
        #if i==0:
        self.stages[i,:] = self.f_solve( b, dt*self.A[i,i] , u0 )
        #else:
        #  self.stages[i,:] = self.f_solve( b, dt*self.A[i,i] , self.stages[i-1,:] )
        
        # Add contribution of current stage to final value
        uend = uend + self.b[i]*dt*self.f(self.stages[i,:])
        
      return uend
      
  # 
  # Returns f(u) = c*u
  #  
  def f(self,u):
    return self.problem.D_upwind.dot(u)+self.problem.M.dot(u)
    
  
  #
  # Solves (Id - alpha*c)*u = b for u
  #  
  def f_solve(self, b, alpha, u0):
    cb = Callback()
    sol, info = LA.gmres( self.problem.Id - alpha*(self.problem.D_upwind + self.problem.M), b, x0=u0, tol=self.problem.gmres_tol, restart=self.problem.gmres_restart, maxiter=self.problem.gmres_maxiter, callback=cb)
    if alpha!=0.0:
      print "DIRK: Number of GMRES iterations: %3i --- Final residual: %6.3e" % ( cb.getcounter(), cb.getresidual() )
      self.logger.add(cb.getcounter())    
    return sol
