import sys
sys.path.append('../')
import numpy as np
import scipy.sparse as sp
from buildFDMatrix import getMatrix, getHorizontalDx, getBCLeft, getBCRight

wave_order = 6

def getWave1DMatrix(N, dx, bc_left, bc_right):
  
  Id = sp.eye(2*N)

  D_u  = getMatrix(N, dx, bc_left[0], bc_right[0], wave_order)
  D_p  = getMatrix(N, dx, bc_left[1], bc_right[1], wave_order)
  Zero = np.zeros((N,N))
  M1 = sp.hstack((Zero, D_p), format="csc")
  M2 = sp.hstack((D_u, Zero), format="csc")
  M   = sp.vstack((M1, M2),  format="csc")
  return sp.csc_matrix(Id), sp.csc_matrix(M)

def getWave1DAdvectionMatrix(N, dx, order):
  Dx   = getHorizontalDx(N, dx, order)
  Zero = np.zeros((N,N))
  M1   = sp.hstack((Dx, Zero), format="csc")
  M2   = sp.hstack((Zero, Dx), format="csc")
  M    = sp.vstack((M1, M2), format="csc")
  return sp.csc_matrix(M)

def getWaveBCLeft(value, N, dx, bc_left):
  bu = getBCLeft(value[0],  N, dx, bc_left[0], wave_order)
  bp = getBCLeft(value[1],  N, dx, bc_left[1], wave_order)
  return np.concatenate((bp, bu))

def getWaveBCRight(value, N, dx, bc_right):
  bu = getBCRight(value[0], N, dx, bc_right[0], wave_order)
  bp = getBCRight(value[1], N, dx, bc_right[1], wave_order)
  return np.concatenate((bp, bu))