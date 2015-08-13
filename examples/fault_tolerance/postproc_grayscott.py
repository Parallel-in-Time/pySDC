import numpy as np
import math
import os
from matplotlib import rc
import matplotlib.pyplot as plt


if __name__ == "__main__":

    # rc('text', usetex=True)
    rc("font", **{"sans-serif": ["Arial"], "size": 30})
    # rc('font', family='serif',size=30)
    rc('legend', fontsize='small')
    rc('xtick', labelsize='small')
    rc('ytick', labelsize='small')

    # set a reference file (either no fault or spread)
    # ref = 'PFASST_GRAYSCOTT_stats_hf_NOFAULT_new.npz'
    ref = 'PFASST_GRAYSCOTT_stats_hf_SPREAD_P32.npz'

    # list files, strategy names, colors, markers
    # list = [('PFASST_GRAYSCOTT_stats_hf_INTERP_PREDICT_P32.npz','2-sided+corr','green','o')]
    list = [ ('PFASST_GRAYSCOTT_stats_hf_SPREAD_P32.npz','SPREAD','1-sided','red','s'),
             ('PFASST_GRAYSCOTT_stats_hf_INTERP_P32.npz','INTERP','2-sided','orange','o'),
             ('PFASST_GRAYSCOTT_stats_hf_SPREAD_PREDICT_P32.npz','SPREAD_PREDICT','1-sided+corr','blue','^'),
             ('PFASST_GRAYSCOTT_stats_hf_INTERP_PREDICT_P32.npz','INTERP_PREDICT','2-sided+corr','green','d') ]

    # no very generic, but easier: set number of processes and xtick frequency
    nprocs = 32
    xtick_dist = 16

    # can choose only some parts of the big run
    minstep = 288
    maxstep = 384
    # minstep = 0
    # maxstep = 640

    # find boundaries
    nsteps = 0
    maxiter = 0
    for file,strategy,label,color,marker in list:

        data = np.load(file)

        iter_count = data['iter_count'][minstep:maxstep]
        residual = data['residual'][:,minstep:maxstep]

        residual = np.where(residual > 0, np.log10(residual), -99)
        vmin = -9
        vmax = int(np.amax(residual))

        maxiter = max(maxiter,int(max(iter_count)))
        nsteps = max(nsteps,len(iter_count))


    # generate comparison of iteration counts between reference run and recovery strategies

    data = np.load(ref)
    ref_iter_count = data['iter_count'][minstep:maxstep]

    fig, ax = plt.subplots(figsize=(20,7))

    # linewidth
    lw = 2

    plt.plot(range(minstep,maxstep),[0]*nsteps,'k-',linewidth=2)

    ymin = 99
    ymax = 0
    for file,strategy,label,color,marker in list:

        if not file is ref:
            data = np.load(file)
            iter_count = data['iter_count'][minstep:maxstep]

            ymin = min(ymin,min(ref_iter_count-iter_count))
            ymax = max(ymax,max(ref_iter_count-iter_count))

            plt.plot(range(minstep,maxstep),ref_iter_count-iter_count,color=color,label=label,marker=marker,linestyle='',linewidth=lw,markersize=12)


    plt.xlabel('step')
    plt.ylabel('saved iterations')
    plt.xlim(-1+minstep,maxstep+1)
    plt.ylim(-1+ymin,ymax+1)
    ax.set_xticks(np.arange(minstep,maxstep,xtick_dist)+0.5, minor=False)
    ax.set_xticklabels(np.arange(minstep,maxstep,xtick_dist), minor=False)
    plt.legend(loc=2,numpoints=1)

    plt.tight_layout()

    # fname = 'GRAYSCOTT_saved_iteration_vs_NOFAULT_hf.png'
    fname = 'GRAYSCOTT_saved_iteration_vs_SPREAD_hf.png'
    plt.savefig(fname, rasterized=True, transparent=True, bbox_inches='tight')

    # generate heatmaps steps vs. iterations with xxx for the faults
    for file,strategy,label,color,marker in list:

        data = np.load(file)

        iter_count = data['iter_count'][minstep:maxstep]
        residual = data['residual'][:,minstep:maxstep]
        stats = data['hard_stats']

        residual = np.where(residual > 0, np.log10(residual), -99)

        fig, ax = plt.subplots(figsize=(20,7))

        cmap = plt.get_cmap('Reds',vmax-vmin+1)
        plt.pcolor(residual,cmap=cmap,vmin=vmin,vmax=vmax)

        if not "NOFAULT" in strategy:
            for item in stats:
                if item[0] in range(minstep,maxstep):
                    plt.text(item[0]+0.5-(maxstep-nsteps),item[1]-1+0.5,'x',horizontalalignment='center',verticalalignment='center')

        plt.axis([0,nsteps,0,maxiter])

        ticks = np.arange(vmin,vmax+1,2)
        tickpos = np.linspace(ticks[0]+0.5, ticks[-1]-0.5, len(ticks))
        cax = plt.colorbar(ticks=tickpos)
        cax.set_ticklabels(ticks)
        cax.set_label('log10(residual)')

        ax.set_xlabel('step')
        ax.set_ylabel('iteration')

        ax.set_yticks(np.arange(1,maxiter,2)+0.5, minor=False)
        ax.set_xticks(np.arange(0,nsteps,xtick_dist)+0.5, minor=False)
        ax.set_yticklabels(np.arange(1,maxiter,2)+1, minor=False)
        ax.set_xticklabels(np.arange(minstep,maxstep,xtick_dist), minor=False)

        plt.tight_layout()

        fname = 'GRAYSCOTT_steps_vs_iteration_hf_'+strategy+'.png'
        plt.savefig(fname, rasterized=True, transparent=True, bbox_inches='tight')


    # generate iteratin counts plot (not used)
    
    fig, ax = plt.subplots(figsize=(20,7))

    # nblocks = int((maxstep-minstep)/nprocs)
    nblocks = 20

    data = np.load('PFASST_GRAYSCOTT_stats_hf_NOFAULT_P32.npz')

    iter_count = data['iter_count'][:]

    iterblocks = np.zeros(nblocks)
    iterblocks[:] = iter_count[nprocs-1::nprocs]
    # for i in range(nblocks):
    #     iterblocks[i] = np.sum(iter_count[i*nprocs:(i+1)*nprocs])/nprocs

    miniter = np.amin(iterblocks)

    plt.plot(range(1,nblocks+1),iterblocks,color='k',label='no fault',marker='',linestyle='--',linewidth=lw,markersize=12)

    for file,strategy,label,color,marker in list:

        data = np.load(file)

        iter_count = data['iter_count']

        iterblocks = np.zeros(nblocks)
        iterblocks[:] = iter_count[nprocs-1::nprocs]
        # for i in range(nblocks):
        #     iterblocks[i] = np.sum(iter_count[i*nprocs:(i+1)*nprocs])/nprocs

        plt.plot(range(1,nblocks+1),iterblocks,color=color,label=label,marker=marker,linestyle='-',linewidth=lw,markersize=12)

    plt.xlabel('block')
    plt.ylabel('number of iterations')
    plt.xlim(0.5,nblocks+0.5)
    plt.ylim(miniter-0.5,maxiter+0.5)
    ax.set_xticks(np.arange(1,nblocks+1), minor=False)
    # ax.set_xticklabels(np.arange(minstep,maxstep,xtick_dist), minor=False)
    plt.legend(loc=2,numpoints=1)

    plt.tight_layout()

    fname = 'GRAYSCOTT_iteration_count_hf.png'
    plt.savefig(fname, rasterized=True, transparent=True, bbox_inches='tight')
    # plt.show()