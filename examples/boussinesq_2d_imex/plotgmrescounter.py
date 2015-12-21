import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from pylab import rcParams

from unflatten import unflatten

if __name__ == "__main__":
  xx = np.load('xaxis.npy')
  uend = np.load('sdc.npy')
  udirk2 = np.load('dirk2.npy')
  udirk4 = np.load('dirk4.npy')
  utrap   = np.load('trap.npy')

  fs = 8
  rcParams['figure.figsize'] = 5.0, 2.5
  fig = plt.figure()

  plt.plot(xx[:,5], uend[2,:,5], '-', color='b', label='SDC')
  plt.plot(xx[:,5], udirk4[2,:,5], '--', color='g', markersize=fs-2, label='DIRK(4)', dashes=(3,3))
  plt.plot(xx[:,5], udirk2[2,:,5], '--', color='r', markersize=fs-2, label='DIRK(2)', dashes=(3,3))
  #plt.plot(xx[:,5], utrap[2,:,5], '--', color='k', markersize=fs-2, label='Trap', dashes=(3,3))
  plt.legend(loc='lower left', fontsize=fs, prop={'size':fs})
  plt.yticks(fontsize=fs)
  plt.xticks(fontsize=fs)
  plt.xlabel('x [km]', fontsize=fs, labelpad=0)
  plt.ylabel('Bouyancy', fontsize=fs, labelpad=1)
  #plt.show()
  plt.savefig('boussinesq.pdf', bbox_inches='tight')

