import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as LA

def getFDMatrix(N, p, dx):

  if (p==2):
    stencil = [-0.5, 0.0, 0.5]
    range   = [-1,0,1]
  elif (p==4):
    stencil = [1.0/12.0, -2.0/3.0, 0.0, 2.0/3.0, -1.0/12.0]
    range  = [-2,-1,0,1,2]
  elif (p==6):
    stencil = [-1.0/60.0, 3.0/20.0, -3.0/4.0, 0.0, 3.0/4.0, -3.0/20.0, 1.0/60.0]
    range = [-3,-2,-1,0,1,2,3]
  else:
    print ("Do not have order %1i implemented." % p)

  A = sp.diags(stencil, range, shape=(N,N))
  A = sp.lil_matrix(A)
  # Now insert periodic BC manually.. I am sure there is a cleverer way to do this ... but meh, can't be bothered to find out
  if (p==2):
    A[0,N-1] = stencil[0]
    A[N-1,0] = stencil[2]
  elif (p==4):
    A[0,N-2] = stencil[0]
    A[0,N-1] = stencil[1]
    A[1,N-1] = stencil[0]
    A[N-2,0] = stencil[4]
    A[N-1,0] = stencil[3]
    A[N-1,1] = stencil[4]
  elif (p==6):
    A[0,N-3] = stencil[0]
    A[0,N-2] = stencil[1]
    A[0,N-1] = stencil[2]
    A[1,N-2] = stencil[0]
    A[1,N-1] = stencil[1]
    A[2,N-1] = stencil[0]
    A[N-3,0] = stencil[6]
    A[N-2,0] = stencil[5]
    A[N-2,1] = stencil[6]
    A[N-1,0] = stencil[4]
    A[N-1,1] = stencil[5]
    A[N-1,2] = stencil[6]
  A *= (1.0/dx)
  return sp.csr_matrix(A)