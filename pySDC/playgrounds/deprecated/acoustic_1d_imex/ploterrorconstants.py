from subprocess import call

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
from pylab import rcParams

fs = 8
order = np.array([])
nsteps = np.array([])
error = np.array([])

# load SDC data
file = open('conv-data.txt', 'r')
while True:
    line = file.readline()
    if not line:
        break
    items = str.split(line, "    ", 3)
    order = np.append(order, int(items[0]))
    nsteps = np.append(nsteps, int(float(items[1])))
    error = np.append(error, float(items[2]))
file.close()
assert np.size(order) == np.size(nsteps), 'Found different number of entries in order and nsteps'
assert np.size(nsteps) == np.size(error), 'Found different number of entries in nsteps and error'

N = np.size(nsteps) / 3
assert isinstance(N, int), 'Number of entries not a multiple of three'

# load Runge-Kutta data
order_rk = np.array([])
nsteps_rk = np.array([])
error_rk = np.array([])
file = open('conv-data-rk.txt', 'r')
while True:
    line = file.readline()
    if not line:
        break
    items = str.split(line, "    ", 3)
    order_rk = np.append(order_rk, int(items[0]))
    nsteps_rk = np.append(nsteps_rk, int(float(items[1])))
    error_rk = np.append(error_rk, float(items[2]))
file.close()
assert np.size(order_rk) == np.size(nsteps_rk), 'Found different number of entries in order and nsteps'
assert np.size(nsteps_rk) == np.size(error_rk), 'Found different number of entries in nsteps and error'

N = np.size(nsteps_rk) / 3
assert isinstance(N, int), 'Number of entries not a multiple of three'

### Compute and plot error constant ###
errconst_sdc = np.zeros((3, N))
errconst_rk = np.zeros((3, N))
nsteps_plot_sdc = np.zeros((3, N))
nsteps_plot_rk = np.zeros((3, N))
order_plot = np.zeros(3)

for ii in range(0, 3):
    order_plot[ii] = order[N * ii]
    for jj in range(0, N):
        p_sdc = order[N * ii + jj]
        err_sdc = error[N * ii + jj]
        nsteps_plot_sdc[ii, jj] = nsteps[N * ii + jj]
        dt_sdc = 1.0 / float(nsteps_plot_sdc[ii, jj])
        errconst_sdc[ii, jj] = err_sdc / dt_sdc ** float(p_sdc)

        p_rk = order_rk[N * ii + jj]
        err_rk = error_rk[N * ii + jj]
        nsteps_plot_rk[ii, jj] = nsteps_rk[N * ii + jj]
        dt_rk = 1.0 / float(nsteps_plot_rk[ii, jj])
        errconst_rk[ii, jj] = err_rk / dt_rk ** float(p_rk)

color = ['r', 'b', 'g']
shape_sdc = ['<', '^', '>']
shape_rk = ['o', 'd', 's']
rcParams['figure.figsize'] = 2.5, 2.5
fig = plt.figure()
for ii in range(0, 3):
    plt.semilogy(
        nsteps_plot_sdc[ii, :],
        errconst_sdc[ii, :],
        shape_sdc[ii],
        markersize=fs,
        color=color[ii],
        label='SDC(' + str(int(order_plot[ii])) + ')',
    )
    plt.semilogy(
        nsteps_plot_rk[ii, :],
        errconst_rk[ii, :],
        shape_rk[ii],
        markersize=fs - 2,
        color=color[ii],
        label='IMEX(' + str(int(order_plot[ii])) + ')',
    )

plt.legend(loc='lower left', fontsize=fs, prop={'size': fs - 1}, ncol=2)
plt.xlabel('Number of time steps', fontsize=fs)
plt.ylabel('Estimated error constant', fontsize=fs, labelpad=2)
plt.xlim([0.9 * np.min(nsteps_plot_sdc), 1.1 * np.max(nsteps_plot_sdc)])
plt.ylim([1e1, 1e6])
plt.yticks([1e1, 1e2, 1e3, 1e4, 1e5, 1e6], fontsize=fs)
plt.xticks([20, 30, 40, 60, 80, 100], fontsize=fs)
plt.gca().get_xaxis().get_major_formatter().labelOnlyBase = False
plt.gca().get_xaxis().set_major_formatter(ScalarFormatter())
# plt.show()
filename = 'error_constants.pdf'
fig.savefig(filename, bbox_inches='tight')
call(["pdfcrop", filename, filename])
