import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from pylab import rcParams
from subprocess import call

from unflatten import unflatten

if __name__ == "__main__":
  xx    = np.load('xaxis.npy')
  uend  = np.load('sdc.npy')
  udirk = np.load('dirk.npy')
  uimex = np.load('rkimex.npy')
  uref  = np.load('uref.npy')

  print("Estimated discretisation error of DIRK: %5.3e" % ( np.linalg.norm(udirk.flatten() - uref.flatten(), np.inf)/np.linalg.norm(uref.flatten(),np.inf) ))
  print("Estimated discretisation error of SDC:  %5.3e" % ( np.linalg.norm(uend.flatten() - uref.flatten(), np.inf)/np.linalg.norm(uref.flatten(),np.inf) ))
  print("Estimated discretisation error of RK-IMEX:  %5.3e" % ( np.linalg.norm(uimex.flatten() - uref.flatten(), np.inf)/np.linalg.norm(uref.flatten(),np.inf) ))

  fs = 8
  rcParams['figure.figsize'] = 5.0, 2.5
  fig = plt.figure()

  plt.plot(xx[:,5], udirk[2,:,5], '--', color='r', markersize=fs-2, label='DIRK', dashes=(3,3))
  plt.plot(xx[:,5], uend[2,:,5], '-', color='b', label='SDC')
  plt.plot(xx[:,5], uimex[2,:,5], '--', color='g', markersize=fs-2, label='RK-IMEX', dashes=(3,3))
  #plt.plot(xx[:,5], utrap[2,:,5], '--', color='k', markersize=fs-2, label='Trap', dashes=(3,3))
  plt.legend(loc='lower left', fontsize=fs, prop={'size':fs})
  plt.yticks(fontsize=fs)
  plt.xticks(fontsize=fs)
  plt.xlabel('x [km]', fontsize=fs, labelpad=0)
  plt.ylabel('Bouyancy', fontsize=fs, labelpad=1)
  #plt.show()
  filename = 'sdc-fwsw-boussinesq.pdf'
  plt.savefig(filename, bbox_inches='tight')
  call(["pdfcrop", filename, filename])

