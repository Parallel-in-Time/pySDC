import matplotlib

matplotlib.use('Agg')

import numpy as np

# import os
import matplotlib.pyplot as plt
from pylab import rcParams

axis_font = {'fontname': 'Arial', 'size': '8', 'family': 'serif'}
fs = 8
ms = 8
lw = 2


def create_plots(cwd=''):
    """
    Function to plot the results of the fault-tolerant Boussinesq system

    Args:
        cwd: current workign directory
    """

    ref = 'PFASST_BOUSSINESQ_stats_hf_NOFAULT_P16.npz'

    # noinspection PyShadowingBuiltins
    list = [
        ('PFASST_BOUSSINESQ_stats_hf_SPREAD_P16.npz', 'SPREAD', '1-sided', 'red', 's'),
        ('PFASST_BOUSSINESQ_stats_hf_INTERP_P16.npz', 'INTERP', '2-sided', 'orange', 'o'),
        ('PFASST_BOUSSINESQ_stats_hf_SPREAD_PREDICT_P16.npz', 'SPREAD_PREDICT', '1-sided+corr', 'blue', '^'),
        ('PFASST_BOUSSINESQ_stats_hf_INTERP_PREDICT_P16.npz', 'INTERP_PREDICT', '2-sided+corr', 'green', 'd'),
        ('PFASST_BOUSSINESQ_stats_hf_NOFAULT_P16.npz', 'NOFAULT', 'no fault', 'black', 'v'),
    ]

    nprocs = 16

    xtick_dist = 8

    minstep = 128
    maxstep = 176
    # minstep = 0
    # maxstep = 320

    nblocks = int(320 / nprocs)

    # maxiter = 14
    nsteps = 0
    maxiter = 0
    vmax = -99
    vmin = -8
    for file, _, _, _, _ in list:
        data = np.load(cwd + 'data/' + file)

        iter_count = data['iter_count'][minstep:maxstep]
        residual = data['residual'][:, minstep:maxstep]

        residual[residual <= 0] = 1e-99
        residual = np.log10(residual)
        vmax = max(vmax, int(np.amax(residual)))

        maxiter = max(maxiter, int(max(iter_count)))
        nsteps = max(nsteps, len(iter_count))

    print(vmin, vmax)
    data = np.load(cwd + 'data/' + ref)
    ref_iter_count = data['iter_count'][nprocs - 1 :: nprocs]

    rcParams['figure.figsize'] = 6.0, 2.5
    fig, ax = plt.subplots()

    plt.plot(range(nblocks), [0] * nblocks, 'k-', linewidth=2)

    ymin = 99
    ymax = 0
    for file, _, label, color, marker in list:
        if file is not ref:
            data = np.load(cwd + 'data/' + file)
            iter_count = data['iter_count'][nprocs - 1 :: nprocs]

            ymin = min(ymin, min(iter_count - ref_iter_count))
            ymax = max(ymax, max(iter_count - ref_iter_count))

            plt.plot(
                range(nblocks),
                iter_count - ref_iter_count,
                color=color,
                label=label,
                marker=marker,
                linestyle='',
                linewidth=lw,
                markersize=ms,
            )

    plt.xlabel('block', **axis_font)
    plt.ylabel('$K_\\mathrm{add}$', **axis_font)
    plt.title('ALL', **axis_font)
    plt.xlim(-1, nblocks)
    plt.ylim(-1 + ymin, ymax + 1)
    plt.legend(loc=2, numpoints=1, fontsize=fs)
    plt.tick_params(axis='both', which='major', labelsize=fs)
    ax.xaxis.labelpad = -0.5
    ax.yaxis.labelpad = -1
    # plt.tight_layout()

    fname = 'data/BOUSSINESQ_Kadd_vs_NOFAULT_hf.png'
    plt.savefig(fname, bbox_inches='tight')
    # os.system('pdfcrop ' + fname + ' ' + fname)

    for file, strategy, _, _, _ in list:
        data = np.load(cwd + 'data/' + file)

        residual = data['residual'][:, minstep:maxstep]
        stats = data['hard_stats']

        residual[residual <= 0] = 1e-99
        residual = np.log10(residual)

        rcParams['figure.figsize'] = 6.0, 2.5
        fig, ax = plt.subplots()

        cmap = plt.get_cmap('Reds', vmax - vmin + 1)
        pcol = plt.pcolor(residual, cmap=cmap, vmin=vmin, vmax=vmax)
        pcol.set_edgecolor('face')

        if file is not ref:
            for item in stats:
                if item[0] in range(minstep, maxstep):
                    plt.text(
                        item[0] + 0.5 - (maxstep - nsteps),
                        item[1] - 1 + 0.5,
                        'x',
                        horizontalalignment='center',
                        verticalalignment='center',
                    )

        plt.axis([0, nsteps, 0, maxiter])

        ticks = np.arange(vmin, vmax + 1)
        tickpos = np.linspace(ticks[0] + 0.5, ticks[-1] - 0.5, len(ticks))
        cax = plt.colorbar(pcol, ticks=tickpos, pad=0.02)
        cax.set_ticklabels(ticks)
        cax.ax.tick_params(labelsize=fs)

        cax.set_label('log10(residual)', **axis_font)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        ax.xaxis.labelpad = -0.5
        ax.yaxis.labelpad = -0.5

        ax.set_xlabel('step', **axis_font)
        ax.set_ylabel('iteration', **axis_font)

        ax.set_yticks(np.arange(1, maxiter, 2) + 0.5, minor=False)
        ax.set_xticks(np.arange(0, nsteps, xtick_dist) + 0.5, minor=False)
        ax.set_yticklabels(np.arange(1, maxiter, 2) + 1, minor=False)
        ax.set_xticklabels(np.arange(minstep, maxstep, xtick_dist), minor=False)

        plt.title(strategy.replace('_', '-'))

        # plt.tight_layout()

        fname = 'data/BOUSSINESQ_steps_vs_iteration_hf_' + strategy + '.png'
        plt.savefig(fname, bbox_inches='tight')
        # os.system('pdfcrop ' + fname + ' ' + fname)

        plt.close('all')


if __name__ == "__main__":
    create_plots()
