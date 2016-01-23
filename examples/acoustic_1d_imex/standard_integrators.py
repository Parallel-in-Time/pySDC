import numpy as np
import math
import scipy.sparse.linalg as LA
import scipy.sparse as sp

#
# Runge-Kutta IMEX methods of order 1 to 3
#
class rk_imex:
  
  def __init__(self, M_fast, M_slow, order):
    assert np.shape(M_fast)[0]==np.shape(M_fast)[1], "A_fast must be square"
    assert np.shape(M_slow)[0]==np.shape(M_slow)[1], "A_slow must be square"
    assert np.shape(M_fast)[0]==np.shape(M_slow)[0], "A_fast and A_slow must be of the same size"

    assert order in [1,2,3,4], "Order must be 1, 2, 3 or 4"
    self.order = order

    if self.order==1:
      self.A      = np.array([[0,0],[0,1]])
      self.A_hat  = np.array([[0,0],[1,0]])
      self.b      = np.array([0,1])
      self.b_hat  = np.array([1,0])
      self.nstages = 2

    elif self.order==2:
      self.A      = np.array([[0,0],[0,0.5]])
      self.A_hat  = np.array([[0,0],[0.5,0]])
      self.b      = np.array([0,1])
      self.b_hat  = np.array([0,1])
      self.nstages = 2

    elif self.order==3:
      # parameter from Pareschi and Russo, J. Sci. Comp. 2005
      alpha = 0.24169426078821
      beta  = 0.06042356519705
      eta   = 0.12915286960590
      self.A_hat   = np.array([ [0,0,0,0], [0,0,0,0], [0,1.0,0,0], [0, 1.0/4.0, 1.0/4.0, 0] ])
      self.A       = np.array([ [alpha, 0, 0, 0], [-alpha, alpha, 0, 0], [0, 1.0-alpha, alpha, 0], [beta, eta, 0.5-beta-eta-alpha, alpha] ])
      self.b_hat   = np.array([0, 1.0/6.0, 1.0/6.0, 2.0/3.0])
      self.b       = self.b_hat
      self.nstages = 4

    elif self.order==4:

      self.A_hat = np.array([ [0,0,0,0,0,0],
                    [1./2,0,0,0,0,0],
                    [13861./62500.,6889./62500.,0,0,0,0],
                    [-116923316275./2393684061468.,-2731218467317./15368042101831.,9408046702089./11113171139209.,0,0,0],
                    [-451086348788./2902428689909.,-2682348792572./7519795681897.,12662868775082./11960479115383.,3355817975965./11060851509271.,0,0],
                 [647845179188./3216320057751.,73281519250./8382639484533.,552539513391./3454668386233.,3354512671639./8306763924573.,4040./17871.,0]])
      self.A = np.array([[0,0,0,0,0,0],
                  [1./4,1./4,0,0,0,0],
                  [8611./62500.,-1743./31250.,1./4,0,0,0],
                  [5012029./34652500.,-654441./2922500.,174375./388108.,1./4,0,0],
                  [15267082809./155376265600.,-71443401./120774400.,730878875./902184768.,2285395./8070912.,1./4,0],
                  [82889./524892.,0,15625./83664.,69875./102672.,-2260./8211,1./4]])
      self.b = np.array([82889./524892.,0,15625./83664.,69875./102672.,-2260./8211,1./4])
      self.b_hat = np.array([4586570599./29645900160.,0,178811875./945068544.,814220225./1159782912.,-3700637./11593932.,61727./225920.])
      self.nstages = 6
    
    self.M_fast = sp.csc_matrix(M_fast)
    self.M_slow = sp.csc_matrix(M_slow)
    self.ndof   = np.shape(M_fast)[0]

    self.stages = np.zeros((self.nstages, self.ndof), dtype='complex')

  def timestep(self, u0, dt):

    # Solve for stages
    for i in range(0,self.nstages):

      # Construct RHS
      rhs = np.copy(u0)
      for j in range(0,i):
        rhs += dt*self.A_hat[i,j]*(self.f_slow(self.stages[j,:])) + dt*self.A[i,j]*(self.f_fast(self.stages[j,:]))

      # Solve for stage i
      if self.A[i,i] == 0:
        # Avoid call to spsolve with identity matrix
        self.stages[i,:] = np.copy(rhs)
      else:
        self.stages[i,:] = self.f_fast_solve( rhs, dt*self.A[i,i] )
    
    # Update 
    for i in range(0,self.nstages):
      u0 += dt*self.b_hat[i]*(self.f_slow(self.stages[i,:])) + dt*self.b[i]*(self.f_fast(self.stages[i,:]))

    return u0

  def f_slow(self, u):
    return self.M_slow.dot(u)

  def f_fast(self, u):
    return self.M_fast.dot(u)

  def f_fast_solve(self, rhs, alpha):
    L = sp.eye(self.ndof) - alpha*self.M_fast
    return LA.spsolve(L, rhs)

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
