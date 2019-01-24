from subprocess import call

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
from pylab import rcParams

fs      = 8
params  = np.array([[3, 3], [3, 4], [3, 5]])
file = open('energy-exact.txt')
energy_exact = float(file.readline().strip())
Tend         = float(file.readline().strip())
nsteps       = float(file.readline().strip())
file.close()

dt = Tend/nsteps
taxis = np.linspace(dt, Tend, nsteps)

energy  = np.array([])
energy_imex = np.array([])
energy_dirk = np.array([])

for ii in range(3):

  filename = 'energy-sdc-K'+str(params[ii,1])+'-M'+str(params[ii,0])+'.txt'
  file = open(filename, 'r')
  while True:
    line = file.readline()
    if not line: break
    energy = np.append(energy, float(line.strip()))
  file.close()

  filename = 'energy-dirk-'+str(params[ii,1])+'.txt'
  file = open(filename, 'r')
  while True:
    line = file.readline()
    if not line: break
    energy_dirk = np.append(energy_dirk, float(line.strip()))
  file.close()

  filename = 'energy-imex-'+str(params[ii,1])+'.txt'
  file = open(filename, 'r')
  while True:
    line = file.readline()
    if not line: break
    energy_imex = np.append(energy_imex, float(line.strip()))
  file.close()

energy = np.split(energy, 3)
energy_dirk = np.split(energy_dirk, 3)
energy_imex = np.split(energy_imex, 3)
for ii in range(3):
  energy[ii][:] = energy[ii][:]/energy_exact
  energy_dirk[ii][:] = energy_dirk[ii][:]/energy_exact
  energy_imex[ii][:] = energy_imex[ii][:]/energy_exact

color = [ 'r', 'b', 'g' ]
shape = ['-', '-', '-']
rcParams['figure.figsize'] = 2.5, 2.5
fig = plt.figure()
for ii in range(0,3):
  plt.plot(taxis, energy[ii], shape[ii], markersize=fs, color=color[ii], label='p='+str(int(params[ii,1])))
  plt.plot(taxis, energy_dirk[ii], '--', color=color[ii], label='DIRK('+str(int(params[ii,1]))+')', dashes=(2.0, 1.0))
  plt.plot(taxis, energy_imex[ii], ':', color=color[ii], label='IMEX('+str(int(params[ii,1]))+')')
plt.legend(loc='lower left', fontsize=fs, prop={'size':fs}, ncol=2)
plt.xlabel('Time', fontsize=fs)
plt.ylabel('Normalised energy', fontsize=fs, labelpad=2)
plt.ylim([0.0, 1.1])
plt.yticks(fontsize=fs)
plt.xticks(fontsize=fs)
plt.gca().get_xaxis().get_major_formatter().labelOnlyBase = False
plt.gca().get_xaxis().set_major_formatter(ScalarFormatter())
#plt.show()
filename = 'energy.pdf'
fig.savefig(filename,bbox_inches='tight')
call(["pdfcrop", filename, filename])