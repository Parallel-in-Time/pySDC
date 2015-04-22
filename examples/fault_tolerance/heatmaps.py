import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt
import os

rc('text', usetex=True)
rc("font", **{"sans-serif": ["Arial"], "size": 30})
# rc('font', family='serif',size=30)
rc('legend', fontsize='small')
rc('xtick', labelsize='small')
rc('ytick', labelsize='small')

fields = [('heat_results_hf_SPREAD.npz','hard restart'),('heat_results_hf_INTERP.npz','interpolation'),
          ('heat_results_hf_PREDICT.npz','predict')]
# fields = [('advection_results_hf_SPREAD.npz','hard restart'),('advection_results_hf_INTERP.npz','interpolation'),('advection_results_hf_PREDICT.npz','predict')]


vmin = 0
vmax = 0
for file,title in fields:

    infile = np.load(file)

    data = infile['iter_count'].T

    data = data-data[0,0]

    ft_iter = infile['ft_iter']
    ft_step = infile['ft_step']

    vmin = min(vmin,max(data.min(),0.0))
    vmax = max(vmax,data.max())

print(vmin,vmax)

for file,title in fields:

    infile = np.load(file)

    data = infile['iter_count'].T

    data = data-data[0,0]

    ft_iter = infile['ft_iter']
    ft_step = infile['ft_step']

    fig, ax = plt.subplots(figsize=(15,10))

    cmap = plt.get_cmap('Reds', vmax-vmin+1)
    plt.pcolor(data, cmap=cmap, vmin=vmin, vmax=vmax)

    plt.axis([ft_step[0],ft_step[-1]+1,ft_iter[0]-1,ft_iter[-1]])

    ticks = np.arange(vmin,vmax+1)
    tickpos = np.linspace(ticks[0]+0.5, ticks[-1]-0.5, len(ticks))
    cax = plt.colorbar(ticks=tickpos)
    cax.set_ticklabels(ticks)
    cax.set_label('additional iterations')

    plt.title(title)

    ax.set_xlabel('affected step')
    ax.set_ylabel('affected iteration')

    ax.set_xticks(np.arange(len(ft_step))+0.5, minor=False)
    ax.set_yticks(np.arange(len(ft_iter))+0.5, minor=False)
    ax.set_xticklabels(ft_step+1, minor=False)
    ax.set_yticklabels(ft_iter, minor=False)

    plt.tight_layout()

    fname = 'add_'+os.path.splitext(file)[0]+'.png'

    plt.savefig(fname, rasterized=True, transparent=True, bbox_inches='tight')
    # plt.show()


exit()