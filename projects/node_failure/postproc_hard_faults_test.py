import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
import os


def create_plots(setup):
    """
        Function to create heatmaps for faults at different steps and iterations

        Args:
            setup (str): name of the setup (heat or advection)
        """

    axis_font = {'fontname': 'Arial', 'size': '8', 'family': 'serif'}
    fs = 8

    fields = [(setup + '_results_hf_SPREAD.npz', 'SPREAD'),
              (setup + '_results_hf_SPREAD_PREDICT.npz', 'SPREAD_PREDICT')]
    # (setup+'_results_hf_INTERP.npz','INTERP'),
    # (setup+'_results_hf_INTERP_PREDICT.npz','INTERP_PREDICT'),

    vmin = 99
    vmax = 0
    for file, strategy in fields:
        infile = np.load(file)

        data = infile['iter_count'].T

        data = data - data[0, 0]

        vmin = min(vmin, data.min())
        vmax = max(vmax, data.max())

    for file, strategy in fields:

        infile = np.load(file)

        data = infile['iter_count'].T

        data = data - data[0, 0]

        ft_iter = infile['ft_iter']
        ft_step = infile['ft_step']

        rcParams['figure.figsize'] = 3.0, 2.5
        fig, ax = plt.subplots()

        cmap = plt.get_cmap('Reds', vmax - vmin + 1)
        pcol = plt.pcolor(data, cmap=cmap, vmin=vmin, vmax=vmax)
        pcol.set_edgecolor('face')

        plt.axis([ft_step[0], ft_step[-1] + 1, ft_iter[0] - 1, ft_iter[-1]])

        ticks = np.arange(int(vmin) + 1, int(vmax) + 2, 2)
        tickpos = np.linspace(ticks[0] + 0.5, ticks[-1] - 0.5, len(ticks))
        cax = plt.colorbar(pcol, ticks=tickpos, format='%2i')

        plt.tick_params(axis='both', which='major', labelsize=fs)

        cax.set_ticklabels(ticks)
        cax.set_label('$K_\mathrm{add}$', **axis_font)
        cax.ax.tick_params(labelsize=fs)

        ax.set_xlabel('affected step', labelpad=1, **axis_font)
        ax.set_ylabel('affected iteration ($K_\mathrm{fault}$)', labelpad=1, **axis_font)

        ax.set_xticks(np.arange(len(ft_step)) + 0.5, minor=False)
        ax.set_xticklabels(ft_step, minor=False)
        ax.set_yticks(np.arange(len(ft_iter)) + 0.5, minor=False)
        ax.set_yticklabels(ft_iter, minor=False)

        # Set every second label to invisible
        for label in ax.xaxis.get_ticklabels()[::2]:
            label.set_visible(False)

        ax.tick_params(pad=2)
        plt.tight_layout()

        # fname = setup+'_iteration_counts_hf_'+strategy+'.png'
        fname = 'data/' + setup + '_iteration_counts_hf_' + strategy + '.pdf'

        # plt.savefig(fname, rasterized=True, transparent=True, bbox_inches='tight')
        plt.savefig(fname, bbox_inches='tight')
        os.system('pdfcrop ' + fname + ' ' + fname)


if __name__ == "__main__":
    create_plots(setup='HEAT')
    create_plots(setup='ADVECTION')
