import numpy as np
import matplotlib.pyplot as plt
import os

axis_font = {'fontname':'Arial', 'size':'30', 'family':'serif'}

# setup = 'HEAT'
setup = 'ADVECTION'
fields = [(setup+'_results_hf_SPREAD.npz','SPREAD'),
          (setup+'_results_hf_INTERP.npz','INTERP'),
          (setup+'_results_hf_INTERP_PREDICT.npz','INTERP_PREDICT'),
          (setup+'_results_hf_SPREAD_PREDICT.npz','SPREAD_PREDICT')]

vmin = 99
vmax = 0
for file,strategy in fields:

    infile = np.load(file)

    data = infile['iter_count'].T

    # data = data-data[0,0]

    ft_iter = infile['ft_iter']
    ft_step = infile['ft_step']

    vmin = min(vmin,data.min())
    vmax = max(vmax,data.max())

print(vmin,vmax)

for file,strategy in fields:

    infile = np.load(file)

    data = infile['iter_count'].T

    # data = data-data[0,0]

    ft_iter = infile['ft_iter']
    ft_step = infile['ft_step']

    fig, ax = plt.subplots(figsize=(15,10))

    cmap = plt.get_cmap('Reds', vmax-vmin+1)
    pcol = plt.pcolor(data, cmap=cmap, vmin=vmin, vmax=vmax)
    pcol.set_edgecolor('face')

    plt.axis([ft_step[0],ft_step[-1]+1,ft_iter[0]-1,ft_iter[-1]])

    ticks = np.arange(vmin,vmax+1,2)
    tickpos = np.linspace(ticks[0]+0.5, ticks[-1]-0.5, len(ticks))
    cax = plt.colorbar(ticks=tickpos)
    
    plt.tick_params(axis='both', which='major', labelsize=20)

    cax.set_ticklabels(ticks)
    cax.set_label('number of iterations', **axis_font)

    ax.set_xlabel('affected step', **axis_font)
    ax.set_ylabel('affected iteration', **axis_font)

    ax.set_xticks(np.arange(len(ft_step))+0.5, minor=False)
    ax.set_yticks(np.arange(len(ft_iter))+0.5, minor=False)
    ax.set_xticklabels(ft_step+1, minor=False)
    ax.set_yticklabels(ft_iter, minor=False)

    ax.tick_params(pad=8)
    plt.tight_layout()

    #fname = setup+'_iteration_counts_hf_'+strategy+'.png'
    fname = setup+'_iteration_counts_hf_'+strategy+'.pdf'

    #plt.savefig(fname, rasterized=True, transparent=True, bbox_inches='tight')
    plt.savefig(fname, bbox_inches='tight')
    # plt.show()


exit()