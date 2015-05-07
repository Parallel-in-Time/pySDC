import numpy as np
import scipy.sparse as sp

#####################################
def getFDMatrix(N, dx, bc_periodic):
  stencil = [1.0, -8.0, 0.0, 8.0, -1.0]
  range   = [ -2,   -1,   0,   1,    2]
  A       = sp.diags(stencil, range, shape=(N,N))
  A       = sp.lil_matrix(A)
  
  if not bc_periodic:
  
    # Lower order stencil for u[1]
    A[1,:] = np.zeros((1,N))
    A[1,0] = -6.0
    A[1,2] =  6.0
    
    # Lower order stencil for u[N-2]
    A[N-2,:] = np.zeros((1,N))
    A[N-2,N-3] = -6.0
    A[N-2,N-1] =  6.0
  
  else:
    A = modify_periodic(A, dx, 'both')
  
  A = 1.0/(12.0*dx)*A

  return sp.csr_matrix(A)

#####################################
def modify_periodic(A, dx, pos):

  A = sp.lil_matrix(A)
  shp = np.shape(A)
  assert shp[0]==shp[1], 'Matrix A must be quadratic'
  N = shp[0]
  
  stencil = np.array([1.0, -8.0, 0.0, 8.0, -1.0])
  
  assert pos in ['left', 'right', 'both'], "Unknown value for pos: "+pos

  if pos in ['left','both']:
    A[0,N-2] = stencil[0]
    A[0,N-1] = stencil[1]
    A[1,N-1] = stencil[0]

  if pos in ['right','both']:
    A[N-2,0] = stencil[4]
    A[N-1,0] = stencil[3]
    A[N-1,1] = stencil[4]

  return A

#####################################
def modify_dirichlet(A, pos):
  
  A = sp.lil_matrix(A)
  shp = np.shape(A)
  assert shp[0]==shp[1], 'Matrix A must be quadratic'
  N = shp[0]

  assert pos in ['left', 'right', 'both'], "Unknown value for pos: "+pos

  if pos in ['left','both']:
    # Dirichlet BC at left
    A[0,:] = np.zeros((1,N))
    A[0,0] = 1.0

  if pos in ['right','both']:
    # Dirichlet BC at right
    A[N-1,:]   = np.zeros((1,N))
    A[N-1,N-1] = 1.0

  return sp.csr_matrix(A)

#####################################
def modify_neumann(A, dx, pos):

  A = sp.lil_matrix(A)
  shp = np.shape(A)
  assert shp[0]==shp[1], 'Matrix A must be quadratic'
  N = shp[0]

  assert pos in ['left', 'right', 'both'], "Unknown value for pos: "+pos

  if pos in ['left','both']:
    # Low order Neumann BC at left
    A[0,:] = np.zeros((1,N))
    A[0,0]    = -1.0/dx
    A[0,1]    =  1.0/dx

  if pos in ['right','both']:
    # Low order Neumann BC at right
    A[N-1,:]   = np.zeros((1,N))
    A[N-1,N-2] = -1.0/dx
    A[N-1,N-1] =  1.0/dx

  return sp.csr_matrix(A)

#####################################
def modify_delete(A, pos):

  A = sp.lil_matrix(A)
  shp = np.shape(A)
  assert shp[0]==shp[1], 'Matrix A must be quadratic'
  N = shp[0]

  assert pos in ['left', 'right', 'both'], "Unknown value for pos: "+pos

  if pos in ['left','both']:
    A[0,:] = np.zeros((1,N))

  if pos in ['right','both']:
    A[N-1,:] = np.zeros((1,N))

  return sp.csr_matrix(A)