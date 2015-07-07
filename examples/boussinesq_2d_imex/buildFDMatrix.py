import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
# Only for periodic BC because we have advection only in x direction
def getUpwindMatrix(N, dx):
     
  #stencil    = [-1.0, 1.0]
  #zero_pos = 2
  #coeff      = 1.0
  
  #stencil    = [1.0, -4.0, 3.0]
  #coeff      = 1.0/2.0
  #zero_pos   = 3
  
  #stencil    = [1.0, -6.0, 3.0, 2.0]
  #coeff      = 1.0/6.0
  #zero_pos   = 3
  
  #stencil  = [-5.0, 30.0, -90.0, 50.0, 15.0]
  #coeff    = 1.0/60.0
  #zero_pos = 4
  
  stencil = [3.0, -20.0, 60.0, -120.0, 65.0, 12.0]
  coeff   = 1.0/60.0
  zero_pos = 5
  
  first_col = np.zeros(N)
  
  # Because we need to specific first column (not row) in circulant, flip stencil array
  first_col[0:np.size(stencil)] = np.flipud(stencil)

  # Circulant shift of coefficient column so that entry number zero_pos becomes first entry
  first_col = np.roll(first_col, -np.size(stencil)+zero_pos, axis=0)

  return sp.csc_matrix( coeff*(1.0/dx)*la.circulant(first_col) )

def getMatrix(N, dx, bc_left, bc_right):
  stencil = [1.0, -8.0, 0.0, 8.0, -1.0]
  range   = [ -2,   -1,   0,   1,    2]
  A       = sp.diags(stencil, range, shape=(N,N))
  A       = sp.lil_matrix(A)

  assert bc_left in ['periodic','neumann','dirichlet'], "Unknown type of BC"
  if bc_left in ['periodic']:
    assert bc_right in ['periodic'], "Periodic BC can only be selected for both sides simultaneously"
  
  if bc_left in ['periodic']:
    A[0,N-2] = stencil[0]
    A[0,N-1] = stencil[1]
    A[1,N-1] = stencil[0]

  if bc_right in ['periodic']:
    A[N-2,0] = stencil[4]
    A[N-1,0] = stencil[3]
    A[N-1,1] = stencil[4]

  if bc_left in ['neumann']:
    A[0,:] = np.zeros(N)
    A[0,0] = -8.0
    A[0,1] =  8.0
    A[1,0] = -8.0 + 4.0/3.0
    A[1,1] = -1.0/3.0

  if bc_right in ['neumann']:
    A[N-1,:]   = np.zeros(N)
    A[N-2,N-1] = 8.0 - 4.0/3.0
    A[N-2,N-2] = 1.0/3.0
    A[N-1,N-1] =  8.0
    A[N-1,N-2] = -8.0

  if bc_left in ['dirichlet']:
    A[0,:] = np.zeros(N)
    A[0,1] = 6.0

  if bc_right in ['dirichlet']:
    A[N-1,:]   = np.zeros(N)
    A[N-1,N-2] = -6.0

  A = 1.0/(12.0*dx)*A

  return sp.csc_matrix(A)

def getBCLeft(value, N, dx, type):

  assert type in ['periodic','neumann','dirichlet'], "Unknown type of BC"

  b = np.zeros(N)
  if type in ['dirichlet']:
    b[0] = -6.0*value
    b[1] =  1.0*value
  
  if type in ['neumann']:
    b[0] = 4.0*dx*value
    b[1] = -(2.0/3.0)*dx*value

  return (1.0/(12.0*dx))*b

def getBCRight(value, N, dx, type):

  assert type in ['periodic','neumann','dirichlet'], "Unknown type of BC"

  b = np.zeros(N)
  if type in ['dirichlet']:
    b[N-2] = -1.0*value
    b[N-1] =  6.0*value
  
  if type in ['neumann']:
    b[N-2] = -(2.0/3.0)*dx*value
    b[N-1] =  4.0*dx*value

  return (1.0/(12.0*dx))*b
