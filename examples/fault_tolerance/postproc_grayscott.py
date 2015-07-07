import numpy as np
import math
import os
from matplotlib import rc
import matplotlib.pyplot as plt


if __name__ == "__main__":

    rc('font', family='serif',size=30)
    rc('legend', fontsize='small')
    rc('xtick', labelsize='small')
    rc('ytick', labelsize='small')

    # ref = 'GRAYSCOTT_stats_hf_NOFAULT.npz'
    ref = 'GRAYSCOTT_stats_hf_SPREAD.npz'

    list = [ ('GRAYSCOTT_stats_hf_SPREAD.npz','SPREAD','green','o'),
             ('GRAYSCOTT_stats_hf_INTERP.npz','INTERP','green','o'),
             ('GRAYSCOTT_stats_hf_INTERP_PREDICT.npz','INTERP_PREDICT','blue','v'),
             ('GRAYSCOTT_stats_hf_SPREAD_PREDICT.npz','SPREAD_PREDICT','red','d') ]

    nprocs = 16

    maxiter = 0
    nsteps = 0
    for file,label,color,marker in list:

        data = np.load(file)

        iter_count = data['iter_count']
        residual = data['residual']

        residual = np.where(residual > 0, np.log10(residual), -99)
        vmin = -9
        vmax = int(np.amax(residual))

        maxiter = max(maxiter,int(max(iter_count)))
        nsteps = max(nsteps,len(iter_count))


    data = np.load(ref)
    ref_iter_count = data['iter_count']

    fig, ax = plt.subplots(figsize=(20,7))

    plt.plot([0]*nsteps,'k-',linewidth=2)

    ymin = 99
    ymax = 0
    for file,label,color,marker in list:

        if not file is ref:
            data = np.load(file)
            iter_count = data['iter_count']

            ymin = min(ymin,min(ref_iter_count-iter_count))
            ymax = max(ymax,max(ref_iter_count-iter_count))

            plt.plot(ref_iter_count-iter_count,color=color,label=label,marker=marker,linestyle='',markersize=12)


    plt.xlabel('Step')
    plt.ylabel('Number of saved iterations')
    plt.xlim(-1,nsteps+1)
    plt.ylim(-1+ymin,ymax+1)
    plt.legend(loc=1,numpoints=1)

    plt.tight_layout()

    fname = 'GRAYSCOTT_saved_iteration_vs_SPREAD_hf.png'
    plt.savefig(fname, rasterized=True, transparent=True, bbox_inches='tight')

    for file,label,color,marker in list:

        data = np.load(file)

        iter_count = data['iter_count']
        residual = data['residual']
        stats = data['hard_stats']

        residual = np.where(residual > 0, np.log10(residual), -99)

        fig, ax = plt.subplots(figsize=(20,7))

        cmap = plt.get_cmap('Reds',vmax-vmin+1)
        plt.pcolor(residual,cmap=cmap,vmin=vmin,vmax=vmax)

        for item in stats:
            plt.text(item[0]+0.5,item[1]-1+0.5,'x',horizontalalignment='center',verticalalignment='center')

        plt.axis([0,nsteps,0,maxiter])

        ticks = np.arange(vmin,vmax+1,2)
        tickpos = np.linspace(ticks[0]+0.5, ticks[-1]-0.5, len(ticks))
        cax = plt.colorbar(ticks=tickpos)
        cax.set_ticklabels(ticks)
        cax.set_label('log10(residual)')

        ax.set_xlabel('step')
        ax.set_ylabel('iteration')

        ax.set_yticks(np.arange(1,maxiter,2)+0.5, minor=False)
        # ax.set_yticks(np.arange(nsteps+1)+0.5, minor=False)
        ax.set_yticklabels(np.arange(1,maxiter,2)+1, minor=False)
        # ax.set_yticklabels(np.arange(nsteps+1), minor=False)

        plt.tight_layout()

        fname = 'GRAYSCOTT_steps_vs_iteration_hf_'+label+'.png'
        plt.savefig(fname, rasterized=True, transparent=True, bbox_inches='tight')

    # plt.show()