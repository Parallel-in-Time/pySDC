import matplotlib
import matplotlib.pylab as plt
import numpy as np
import scipy.optimize as opt

from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right


def main():
    def rho(x):
        return max(abs(np.linalg.eigvals(np.eye(M) - np.diag([x[i] for i in range(M)]).dot(coll.Qmat[1:, 1:]))))

    M = 2

    coll = CollGaussRadau_Right(M, 0, 1)

    x0 = np.ones(M)
    d = opt.minimize(rho, x0, method='Nelder-Mead')
    print(d)

    numsteps = 800
    xdim = np.linspace(0, 8, numsteps)
    ydim = np.linspace(0, 13, numsteps)

    minfield = np.zeros((len(xdim), len(ydim)))

    for idx, x in enumerate(xdim):
        for idy, y in enumerate(ydim):
            minfield[idx, idy] = max(abs(np.linalg.eigvals(np.eye(M) - np.diag([x, y]).dot(coll.Qmat[1:, 1:]))))

    # Set up plotting parameters
    params = {'legend.fontsize': 20,
              'figure.figsize': (12, 8),
              'axes.labelsize': 20,
              'axes.titlesize': 20,
              'xtick.labelsize': 16,
              'ytick.labelsize': 16,
              'lines.linewidth': 3
              }
    plt.rcParams.update(params)
    matplotlib.style.use('classic')

    plt.figure()
    plt.pcolor(xdim, ydim, minfield.T, cmap='Reds', vmin=0, vmax=1)
    plt.text(d.x[0], d.x[1], 'X', horizontalalignment='center', verticalalignment='center')
    plt.xlim((min(xdim), max(xdim)))
    plt.ylim((min(ydim), max(ydim)))
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    cbar = plt.colorbar()
    cbar.set_label('spectral radius')

    fname = 'data/parallelSDC_minimizer_full.png'
    plt.savefig(fname,  bbox_inches='tight')

    plt.figure()
    xdim_part = xdim[int(0.25 * numsteps):int(0.75 * numsteps) + 1]
    ydim_part = ydim[0:int(0.25 * numsteps)]
    minfield_part = minfield[int(0.25 * numsteps):int(0.75 * numsteps) + 1, 0:int(0.25 * numsteps)]
    plt.pcolor(xdim_part, ydim_part, minfield_part.T, cmap='Reds', vmin=0, vmax=1)
    plt.text(d.x[0], d.x[1], 'X', horizontalalignment='center', verticalalignment='center')
    plt.xlim((min(xdim_part), max(xdim_part)))
    plt.ylim((min(ydim_part), max(ydim_part)))
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    cbar = plt.colorbar()
    cbar.set_label('spectral radius')

    fname = 'data/parallelSDC_minimizer_zoom.png'
    plt.savefig(fname,  bbox_inches='tight')


if __name__ == "__main__":
    main()
