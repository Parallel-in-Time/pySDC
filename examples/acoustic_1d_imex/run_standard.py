import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as LA
import os
from matplotlib import pyplot as plt
from buildWave1DMatrix import getWave1DMatrix, getWave1DAdvectionMatrix

sigma_0 = 0.1
k       = 7.0*2*np.pi
x_0     = 0.75
x_1     = 0.25

nvars = 512
cs    = 1.0
cadv  = 0.0
order = 4

def u(x,t, multiscale):
  u0 = np.exp(-np.square( np.mod( mesh- cs*t, 1.0 ) -x_0 )/(sigma_0*sigma_0)) + multiscale*np.exp(-np.square( np.mod( mesh -cs*t, 1.0 ) -x_1 )/(sigma_0*sigma_0))*np.cos(k*( np.mod( mesh-cs*t, 1.0 ))/sigma_0)
  p0 = u0
  return u0, p0

Tend   = 3.0
Nsteps = 154
dt = Tend/float(Nsteps)

mesh   = np.linspace(0.0, 1.0, nvars, endpoint=False)
dx     = mesh[1] - mesh[0]
Dx     = -cadv*getWave1DAdvectionMatrix(nvars, dx, order)
Id, A  = getWave1DMatrix(nvars, dx, ['periodic','periodic'], ['periodic','periodic'])
A      = -cs*A

M_ieuler = Id - dt*(A + Dx)

M_bdf    = Id - (2.0/3.0)*dt*(A + Dx)

alpha    = 0.5
M_trap   = Id - alpha*dt*(A+Dx)
B_trap   = Id + (1-alpha)*dt*(A+Dx)
 
u0, p0 = u(mesh, 0.0, 1.0)
y0_ie  = np.concatenate( (u0, p0) )
y0_tp  = y0_ie

y0_bdf = y0_ie

fig = plt.figure(figsize=(8,8))

for i in range(0,Nsteps):

  # implicit Euler step
  ynew_ie = LA.spsolve(M_ieuler, y0_ie)

  # trapezoidal rule step
  b_tp    = B_trap.dot(y0_tp)
  ynew_tp = LA.spsolve(M_trap, b_tp)

  # BDF-2 scheme
  if i==0:
    ynew_bdf = LA.spsolve(M_ieuler, y0_bdf)
  else:
    b_bdf    = (4.0/3.0)*y0_bdf - (1.0/3.0)*ym1_bdf
    ynew_bdf = LA.spsolve(M_bdf, b_bdf)

  unew_ie, pnew_ie = np.split(ynew_ie, 2)
  unew_tp, pnew_tp = np.split(ynew_tp, 2)
  unew_bdf, pnew_bdf = np.split(ynew_bdf, 2)
  uex, pex = u(mesh, float(i+1)*dt, 0.0)

  fig.gca().clear()
  #plt.plot(mesh, pnew_bdf, 'b', label='BDF-2')
  plt.plot(mesh, pnew_tp, 'r', label='Trapezoidal')
  plt.plot(mesh, pex, 'k', label='Slow Mode')
  fig.gca().set_xlim([0, 1.0])
  fig.gca().set_ylim([-0.5, 1.1])
  fig.gca().legend(loc=3)
  fig.gca().grid()
  #plt.draw()
  plt.pause(0.0001)  
  #if i==0:
    #plt.gcf().savefig('initial.pdf', bbox_inches='tight')
  filename = 'images/standard'+"%03d" % i
  plt.gcf().savefig(filename+'.png', bbox_inches='tight')
  os.system('convert -quality 100 '+filename+'.png '+filename+'.jpeg')
  os.system('rm '+filename+'.png')

  y0_ie   = ynew_ie
  y0_tp   = ynew_tp
  ym1_bdf = y0_bdf
  y0_bdf  = ynew_bdf
#plt.show()
#lt.gcf().savefig('final.pdf', bbox_inches='tight')
os.system('ffmpeg -r 25 -i images/standard-%03d.jpeg -vcodec libx264 -crf 25 movie.avi')

