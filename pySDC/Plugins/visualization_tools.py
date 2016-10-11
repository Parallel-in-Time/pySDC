from pySDC.Plugins.stats_helper import filter_stats

import numpy as np

from matplotlib import rc
import matplotlib.pyplot as plt


def show_residual_across_simulation(stats,fname):

    extract_stats = filter_stats(stats,type='residual')

    maxsteps = 0
    maxiter = 0
    minres = 0
    maxres = -99
    for k,v in extract_stats.items():
        maxsteps = max(maxsteps,getattr(k,'step'))
        maxiter = max(maxiter,getattr(k,'iter'))
        minres = min(minres,np.log10(v))
        maxres = max(maxres,np.log10(v))
        # print(getattr(k,'step'),getattr(k,'iter'),v)

    # print(maxsteps,maxiter,minres,maxres)

    residual = np.zeros((maxiter,maxsteps+1))
    residual[:] = -99

    for k,v in extract_stats.items():
        step = getattr(k,'step')
        iter = getattr(k,'iter')
        if iter is not -1:
            residual[iter-1,step] = np.log10(v)

    # Set up latex stuff and fonts
    # rc('text', usetex=True)
    # rc('font', **{"sans-serif": ["Arial"], "size": 30})
    rc('font', family='serif',size=30)
    rc('legend', fontsize='small')
    rc('xtick', labelsize='small')
    rc('ytick', labelsize='small')

    fig, ax = plt.subplots(figsize=(15,10))

    cmap = plt.get_cmap('Reds')
    plt.pcolor(residual.T, cmap=cmap, vmin=minres, vmax=maxres)

    cax = plt.colorbar()
    cax.set_label('log10(residual)')

    ax.set_xlabel('iteration')
    ax.set_ylabel('step')

    ax.set_xticks(np.arange(maxiter)+0.5, minor=False)
    ax.set_yticks(np.arange(maxsteps+1)+0.5, minor=False)
    ax.set_xticklabels(np.arange(maxiter)+1, minor=False)
    ax.set_yticklabels(np.arange(maxsteps+1), minor=False)

    plt.tight_layout()

    plt.savefig(fname, rasterized=True, transparent=True, bbox_inches='tight')

    plt.show()

    pass