import numpy as np
import scipy.sparse as sp
from build2DFDMatrix import get2DMatrix, getBCHorizontal, getBCVertical, get2DUpwindMatrix

def getBoussinesq2DUpwindMatrix(N, dx):

  Dx   = get2DUpwindMatrix(N, dx)
  
  Zero = np.zeros((N[0]*N[1],N[0]*N[1]))
  M1 = sp.hstack((Dx ,  Zero,   Zero, Zero), format="csr")
  M2 = sp.hstack((Zero,   Dx,   Zero, Zero), format="csr")
  M3 = sp.hstack((Zero, Zero,     Dx, Zero), format="csr")
  M4 = sp.hstack((Zero, Zero,   Zero,   Dx), format="csr")
  M  = sp.vstack((M1,M2,M3,M4), format="csr")
  
  return sp.csc_matrix(M)
  
def getBoussinesq2DMatrix(N, h, bc_hor, bc_ver, Nfreq):
  Dx_u, Dz_u = get2DMatrix(N, h, bc_hor[0], bc_ver[0])
  Dx_w, Dz_w = get2DMatrix(N, h, bc_hor[1], bc_ver[1])
  Dx_b, Dz_b = get2DMatrix(N, h, bc_hor[2], bc_ver[2])
  Dx_p, Dz_p = get2DMatrix(N, h, bc_hor[3], bc_ver[3])
  Id_N = sp.eye(N[0]*N[1])

  Zero = np.zeros((N[0]*N[1],N[0]*N[1]))
  Id_w = -sp.eye(N[0]*N[1])
  Id_b = Nfreq**2*sp.eye(N[0]*N[1])
  M1 = sp.hstack((Zero, Zero, Zero, Dx_p), format="csr")
  M2 = sp.hstack((Zero, Zero, Id_w, Dz_p), format="csr")
  M3 = sp.hstack((Zero, Id_b, Zero, Zero), format="csr")
  M4 = sp.hstack((Dx_u, Dz_w, Zero, Zero), format="csr")
  M  = sp.vstack((M1,M2,M3,M4), format="csr")

  Id = sp.eye(4*N[0]*N[1])

  return sp.csc_matrix(Id), sp.csc_matrix(M)

def getBoussinesqBCHorizontal(value, N, dx, bc_hor):
  
  bu_left, bu_right = getBCHorizontal( value[0], N, dx, bc_hor[0] )
  bw_left, bw_right = getBCHorizontal( value[1], N, dx, bc_hor[1] )
  bb_left, bb_right = getBCHorizontal( value[2], N, dx, bc_hor[2] )
  bp_left, bp_right = getBCHorizontal( value[3], N, dx, bc_hor[3] )
  
  b_left  = np.concatenate(( bp_left,  bp_left,  bu_left + bw_left))
  b_right = np.concatenate(( bp_right, bp_right, bu_right + bw_right))
  return b_left, b_right

def getBoussinesqBCVertical():
  return 0.0