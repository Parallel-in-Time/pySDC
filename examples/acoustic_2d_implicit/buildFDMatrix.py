import numpy as np
import scipy.sparse as sp

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
