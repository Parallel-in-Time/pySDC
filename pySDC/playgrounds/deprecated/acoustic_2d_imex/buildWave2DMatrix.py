import numpy as np
import scipy.sparse as sp
from build2DFDMatrix import get2DMatrix, getBCHorizontal, get2DUpwindMatrix

def getWave2DUpwindMatrix(N, dx):

  Dx   = get2DUpwindMatrix(N, dx)
  
  Zero = np.zeros((N[0]*N[1],N[0]*N[1]))
  M1 = sp.hstack((Dx ,  Zero,   Zero), format="csr")
  M2 = sp.hstack((Zero,   Dx,   Zero), format="csr")
  M3 = sp.hstack((Zero, Zero,     Dx), format="csr")
  M  = sp.vstack((M1,M2,M3), format="csr")
  
  return sp.csc_matrix(M)
  
def getWave2DMatrix(N, h, bc_hor, bc_ver):
  Dx_u, Dz_u = get2DMatrix(N, h, bc_hor[0], bc_ver[0])
  Dx_w, Dz_w = get2DMatrix(N, h, bc_hor[1], bc_ver[1])
  Dx_p, Dz_p = get2DMatrix(N, h, bc_hor[2], bc_ver[2])

  Id_N = sp.eye(N[0]*N[1])

  Zero = np.zeros((N[0]*N[1],N[0]*N[1]))
  M1 = sp.hstack((Zero, Zero, Dx_p), format="csr")
  M2 = sp.hstack((Zero, Zero, Dz_p), format="csr")
  M3 = sp.hstack((Dx_u, Dz_w, Zero), format="csr")
  M  = sp.vstack((M1,M2,M3), format="csr")

  Id = sp.eye(3*N[0]*N[1])

  return sp.csc_matrix(Id), sp.csc_matrix(M)

def getWaveBCHorizontal(value, N, dx, bc_hor):
  
  bu_left, bu_right = getBCHorizontal( value[0], N, dx, bc_hor[0] )
  bw_left, bw_right = getBCHorizontal( value[1], N, dx, bc_hor[1] )
  bp_left, bp_right = getBCHorizontal( value[2], N, dx, bc_hor[2] )
  
  b_left  = np.concatenate(( bp_left, bp_left, bu_left + bw_left))
  b_right = np.concatenate(( bp_right, bp_right, bu_right + bw_right))
  return b_left, b_right

def getWaveBCVertical():
  return 0.0