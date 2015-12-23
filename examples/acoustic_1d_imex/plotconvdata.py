import numpy as np
from matplotlib import pyplot as plt
from pylab import rcParams
from matplotlib.ticker import ScalarFormatter
from subprocess import call

fs     = 8
order  = np.array([])
nsteps = np.array([])
error  = np.array([])

file = open('conv-data.txt', 'r')
while True:
  line = file.readline()
  if not line: break
  items  = str.split(line, "    ", 3)
  order  = np.append(order,  int(items[0]))
  nsteps = np.append(nsteps, int(items[1]))
  error  = np.append(error,  float(items[2]))

assert np.size(order)==np.size(nsteps), 'Found different number of entries in order and nsteps'
assert np.size(nsteps)==np.size(error), 'Found different number of entries in nsteps and error'

N = np.size(nsteps)/3
assert isinstance(N, int), 'Number of entries not a multiple of three'

error_plot  = np.zeros((3, N))
nsteps_plot = np.zeros((3, N))
convline    = np.zeros((3, N))
order_plot  = np.zeros(3)

for ii in range(0,3):
  order_plot[ii] = order[N*ii]
  for jj in range(0,N):
    error_plot[ii,jj]  = error[N*ii+jj]
    nsteps_plot[ii,jj] = nsteps[N*ii+jj]
    convline[ii,jj]    = error_plot[ii,0]*(float(nsteps_plot[ii,0])/float(nsteps_plot[ii,jj]))**order_plot[ii]

color = [ 'r', 'b', 'g' ]
shape = ['o', 'd', 's']
rcParams['figure.figsize'] = 2.5, 2.5
fig = plt.figure()
for ii in range(0,3):
  plt.loglog(nsteps_plot[ii,:], convline[ii,:], '-', color=color[ii])
  plt.loglog(nsteps_plot[ii,:], error_plot[ii,:], shape[ii], markersize=fs, color=color[ii], label='p='+str(int(order_plot[ii])))


plt.legend(loc='upper right', fontsize=fs, prop={'size':fs})
plt.xlabel('Number of time steps', fontsize=fs)
plt.ylabel('Relative error', fontsize=fs, labelpad=2)
plt.xlim([0.9*np.min(nsteps_plot), 1.1*np.max(nsteps_plot)])
plt.ylim([1e-7, 1e1])
plt.yticks([1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1],fontsize=fs)
plt.xticks([25, 50, 100], fontsize=fs)
plt.gca().get_xaxis().get_major_formatter().labelOnlyBase = False
plt.gca().get_xaxis().set_major_formatter(ScalarFormatter())
plt.show()
filename = 'sdc_fwsw_convergence.pdf'
fig.savefig(filename,bbox_inches='tight')
call(["pdfcrop", filename, filename])


