import numpy as np
import scipy.sparse as sp
from examples.boussinesq_2d_imex.build2DFDMatrix import get2DMatrix, getBCHorizontal, getBCVertical, get2DUpwindMatrix

def getBoussinesq2DUpwindMatrix(N, dx, u_adv, order):

  Dx   = get2DUpwindMatrix(N, dx, order)
  
  # Note: In the equations it is u_t + u_adv* D_x u = ... so in order to comply with the form u_t = M u,
  # add a minus sign in front of u_adv
  
  Zero = np.zeros((N[0]*N[1], N[0]*N[1]))
  M1 = sp.hstack((-u_adv*Dx,        Zero,      Zero,      Zero), format="csr")
  M2 = sp.hstack((     Zero,   -u_adv*Dx,      Zero,      Zero), format="csr")
  M3 = sp.hstack((     Zero,        Zero, -u_adv*Dx,      Zero), format="csr")
  M4 = sp.hstack((     Zero,        Zero,      Zero, -u_adv*Dx), format="csr")
  M  = sp.vstack((M1,M2,M3,M4), format="csr")
  
  return sp.csc_matrix(M)
  
def getBoussinesq2DMatrix(N, h, bc_hor, bc_ver, c_s, Nfreq, order):

  Dx_u, Dz_u = get2DMatrix(N, h, bc_hor[0], bc_ver[0], order)
  Dx_w, Dz_w = get2DMatrix(N, h, bc_hor[1], bc_ver[1], order)
  Dx_b, Dz_b = get2DMatrix(N, h, bc_hor[2], bc_ver[2], order)
  Dx_p, Dz_p = get2DMatrix(N, h, bc_hor[3], bc_ver[3], order)

  Id_N = sp.eye(N[0]*N[1])

  Zero = np.zeros((N[0]*N[1],N[0]*N[1]))
  Id_w = sp.eye(N[0]*N[1])

  # Note: Bring all terms to right hand side, therefore a couple of minus signs
  # are needed

  M1 = sp.hstack((        Zero,          Zero,  Zero, -Dx_p), format="csr")
  M2 = sp.hstack((        Zero,          Zero,  Id_w, -Dz_p), format="csr")
  M3 = sp.hstack((        Zero, -Nfreq**2*Id_w,  Zero, Zero), format="csr")
  M4 = sp.hstack((-c_s**2*Dx_u,   -c_s**2*Dz_w,  Zero, Zero), format="csr")
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
