import math

import numpy as np
import numpy.linalg as LA
from matplotlib import pyplot as plt

from pySDC.playgrounds.deprecated.advection_1d_implicit.getFDMatrix import getFDMatrix


def u_function(x):
  u = np.zeros(np.size(x))
  for i in range(0,np.size(x)):
    u[i] = math.cos(2.0*math.pi*x[i])
  return u

def u_x_function(x):
  u = np.zeros(np.size(x))
  for i in range(0,np.size(x)):
    u[i] = -2.0*math.pi*math.sin(2.0*math.pi*x[i])
  return u

p = [2, 4, 6]
Nv = [10, 25, 50, 75, 100, 125, 150]

error = np.zeros([np.size(p),np.size(Nv)])
orderline = np.zeros([np.size(p),np.size(Nv)])

for j in range(0,np.size(Nv)):
  x = np.linspace(0, 1, num=Nv[j], endpoint=False)
  dx = x[1]-x[0]
  for i in range(0,np.size(p)):
    A = getFDMatrix(Nv[j],p[i], dx)
    u = u_function(x)
    u_x_fd = A.dot(u)
    u_x_ex = u_x_function(x)
    error[i,j] = LA.norm(u_x_fd - u_x_ex, np.inf)/LA.norm(u_x_ex, np.inf)
    orderline[i,j] = (float(Nv[0])/float(Nv[j]))**p[i]*error[i,0]

fig = plt.figure(figsize=(8,8))
plt.loglog(Nv, error[0,:], 'o', label='p=2', markersize=12, color='b')
plt.loglog(Nv, orderline[0,:], color='b')
plt.loglog(Nv, error[1,:], 'o', label='p=4', markersize=12, color='r')
plt.loglog(Nv, orderline[1,:], color='r')
plt.loglog(Nv, error[2,:], 'o', label='p=6', markersize=12, color='g')
plt.loglog(Nv, orderline[2,:], color='g')
plt.legend()
plt.xlim([Nv[0], Nv[np.size(Nv)-1]])
plt.xlabel(r'$N_x$')
plt.ylabel('Relative error')
plt.show()