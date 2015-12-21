import numpy as np
import math
import scipy.sparse.linalg as LA
import scipy.sparse as sp

#
# Trapezoidal rule
#
class trapezoidal:
 
  def __init__(self, M, alpha=0.5):
    assert np.shape(M)[0]==np.shape(M)[1], "Matrix M must be quadratic"
    self.Ndof = np.shape(M)[0]
    self.M = M
    self.alpha = alpha

  def timestep(self, u0, dt):
    M_trap   = sp.eye(self.Ndof) - self.alpha*dt*self.M
    B_trap   = sp.eye(self.Ndof) + (1.0-self.alpha)*dt*self.M
    b = B_trap.dot(u0)
    return LA.spsolve(M_trap, b)
#
# A BDF-2 implicit two-step method
#
class bdf2:

  def __init__(self, M):
    assert np.shape(M)[0]==np.shape(M)[1], "Matrix M must be quadratic"
    self.Ndof = np.shape(M)[0]
    self.M = M

  def firsttimestep(self, u0, dt):
    b = u0
    L = sp.eye(self.Ndof) - dt*self.M
    return LA.spsolve(L, b)

  def timestep(self, u0, um1, dt):
    b = (4.0/3.0)*u0 - (1.0/3.0)*um1
    L = sp.eye(self.Ndof) - (2.0/3.0)*dt*self.M
    return LA.spsolve(L, b)
#
# A diagonally implicit Runge-Kutta method of order 2, 3 or 4
#
class dirk:

  def __init__(self, M, order):

    assert np.shape(M)[0]==np.shape(M)[1], "Matrix M must be quadratic"
    self.Ndof = np.shape(M)[0]
    self.M = M
    self.order = order

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
       
    self.stages  = np.zeros((self.nstages,self.Ndof), dtype='complex')

  def timestep(self, u0, dt):
    
      uend           = u0
      for i in range(0,self.nstages):  
        
        b = u0
        
        # Compute right hand side for this stage's implicit step
        for j in range(0,i):
          b = b + self.A[i,j]*dt*self.f(self.stages[j,:])
        
        # Implicit solve for current stage    
        self.stages[i,:] = self.f_solve( b, dt*self.A[i,i] )
        
        # Add contribution of current stage to final value
        uend = uend + self.b[i]*dt*self.f(self.stages[i,:])
        
      return uend
      
  # 
  # Returns f(u) = c*u
  #  
  def f(self,u):
    return self.M.dot(u)
    
  
  #
  # Solves (Id - alpha*c)*u = b for u
  #  
  def f_solve(self, b, alpha):
    L = sp.eye(self.Ndof) - alpha*self.M
    return LA.spsolve(L, b)
