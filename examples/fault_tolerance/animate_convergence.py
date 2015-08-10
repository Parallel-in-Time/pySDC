import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt
from matplotlib import animation

if __name__ == "__main__":

    rc('font', family='sans-serif',size=30)
    rc('legend', fontsize='small')
    rc('xtick', labelsize='small')
    rc('ytick', labelsize='small')

    nprocs = 1

    xtick_dist = 16

    minstep = 288
    maxstep = 384

    maxiter_full = 14
    maxiter = 0
    nsteps = 0

    ref = 'SDC_GRAYSCOTT_stats_hf_NOFAULT_new.npz'
    # ref = 'PFASST_GRAYSCOTT_stats_hf_NOFAULT_P32.npz'
    # ref = 'PFASST_GRAYSCOTT_stats_hf_SPREAD_P32.npz'

    data = np.load(ref)

    iter_count = data['iter_count'][minstep:maxstep]
    residual = data['residual'][:,minstep:maxstep]
    iter_count_blocks = []
    for p in range(int((maxstep-minstep)/nprocs)):
        step = p*nprocs
        iter_count_blocks.append(int(max(iter_count[step:step+nprocs])))

    residual = np.where(residual > 0, np.log10(residual), -99)
    vmin = -9
    vmax = int(np.amax(residual))

    maxiter = max(maxiter,int(max(iter_count)))
    maxiter_full = max(maxiter_full,maxiter)
    nsteps = max(nsteps,len(iter_count))


    fig, ax = plt.subplots(figsize=(20,7))

    ticks = np.arange(vmin,vmax+1,2)
    tickpos = np.linspace(ticks[0]+0.5, ticks[-1]-0.5, len(ticks))
    # cax = plt.colorbar(ticks=tickpos)
    # cax.set_ticklabels(ticks)
    # cax.set_label('log10(residual)')

    ax.set_xlabel('step')
    ax.set_ylabel('iteration')

    ax.set_yticks(np.arange(1,maxiter_full,2)+0.5, minor=False)
    ax.set_xticks(np.arange(0,nsteps,xtick_dist)+0.5, minor=False)
    ax.set_yticklabels(np.arange(1,maxiter_full,2)+1, minor=False)
    ax.set_xticklabels(np.arange(minstep,maxstep,xtick_dist), minor=False)

    cmap = plt.get_cmap('Reds',vmax-vmin+1)

    residual = np.zeros((maxiter_full,maxstep-minstep))
    plot = plt.pcolor(residual,cmap=cmap,vmin=vmin,vmax=vmax)
    text = plt.text(0,0,'',horizontalalignment='center',verticalalignment='center')

    ticks = np.arange(vmin,vmax+1,2)
    tickpos = np.linspace(ticks[0]+0.5, ticks[-1]-0.5, len(ticks))
    cax = plt.colorbar(ticks=tickpos)
    cax.set_ticklabels(ticks)
    cax.set_label('log10(residual)')

    fig.tight_layout()

    def init():
        res = np.zeros((maxiter_full,maxstep-minstep))
        plot.set_array(res.ravel())

        return plot

    def animate(index):

        csum_blocks = np.zeros(len(iter_count_blocks)+1)
        csum_blocks[1:] = np.cumsum(iter_count_blocks)
        block = np.searchsorted(csum_blocks[1:],index)
        step = block*nprocs + minstep
        iter = index - int(csum_blocks[block])

        # print(block, step, iter, iter_count_blocks, sum(iter_count_blocks))

        res = np.zeros((maxiter_full,maxstep-minstep))
        res[0:maxiter,0:step-minstep] = data['residual'][0:maxiter,minstep:step]
        res[0:iter,0:step+nprocs-minstep] = data['residual'][0:iter,minstep:step+nprocs]
        res = np.where(res > 0, np.log10(res), -99)
        plot.set_array(res.ravel())

        return plot

    anim = animation.FuncAnimation(fig,animate,init_func=init,frames=sum(iter_count_blocks)+1,interval=1,blit=False,repeat=False)

    if not "NOFAULT" in ref:
        stats = data['hard_stats']
        for item in stats:
            if item[0] in range(minstep,maxstep):
                plt.text(item[0]+0.5-(maxstep-nsteps),item[1]-1+0.5,'x',horizontalalignment='center',verticalalignment='center')

    # fig.subplots_adjust(left=0.01, bottom=0.01, right=1.2, top=1, wspace=None, hspace=None)

    # Set up formatting for the movie files
    Writer = animation.writers['ffmpeg']
    # choose fps=1 for PFASST, fps=15 for SDC
    writer = Writer(fps=15 , metadata=dict(artist='Me'), bitrate=3200)

    fname = 'anim_conv_'+ref.split('.')[0]+'.mp4'
    anim.save(fname,writer=writer)

    # plt.show()