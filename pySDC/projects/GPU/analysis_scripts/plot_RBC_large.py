import pickle
import numpy
import matplotlib.pyplot as plt
from pySDC.helpers.stats_helper import get_sorted, get_list_of_types
from pySDC.helpers.plot_helper import figsize_by_journal, setup_mpl


class PlotRBC:
    def __init__(self, res, procs, base_path):
        self.res = res
        self.procs = procs
        self.base_path = base_path

    def get_path(self, idx, n_space, stats=False):
        _name = '-stats' if stats else ''
        return f'{self.base_path}/data/RayleighBenard_large-res_{self.res}-useGPU_False-procs_{self.procs[0]}_{self.procs[1]}_{self.procs[2]}-0-{n_space}-solution_{idx:06d}{_name}.pickle'

    def plot_verification(self):
        stats = {}
        for idx in range(999):
            try:
                path = self.get_path(idx=idx, n_space=0, stats=True)
                with open(path, 'rb') as file:
                    stats = {**stats, **pickle.load(file)}
            except FileNotFoundError:
                break

        nu = get_sorted(stats, type='Nusselt', recomputed=False)
        fig, ax = plt.subplots(figsize=figsize_by_journal('TUHH_thesis', 0.8, 0.6))
        for key in ['t', 'b', 'V']:
            ax.plot([me[0] for me in nu], [me[1][key] for me in nu], label=fr'$Nu_\mathrm{{{key}}}$')
        ax.legend(frameon=False)
        ax.set_xlabel('$t$')
        fig.savefig('plots/RBC_large_Nusselt.pdf', bbox_inches='tight')
        # print(get_list_of_types(stats))

    def plot_series(self):
        indices = [0, 50, 80, 150]  # [0, 10, 20, 49]

        from mpl_toolkits.axes_grid1 import make_axes_locatable

        plt.rcParams['figure.constrained_layout.use'] = True
        fig, axs = plt.subplots(
            len(indices), 1, figsize=figsize_by_journal('TUHH_thesis', 1, 1.3), sharex=True, sharey=True
        )
        caxs = {}
        for i in range(len(indices)):
            divider = make_axes_locatable(axs[i])
            caxs[axs[i]] = divider.append_axes('right', size='3%', pad=0.03)

        # get grid
        X = {}
        Z = {}
        for r in range(self.procs[2]):
            path = self.get_path(idx=0, n_space=r)
            with open(path, 'rb') as file:
                data = pickle.load(file)
            X[r] = data['X']
            Z[r] = data['Z']

        def plot_single(idx, ax):
            for r in range(self.procs[2]):
                path = self.get_path(idx=idx, n_space=r)
                with open(path, 'rb') as file:
                    data = pickle.load(file)
                im = ax.pcolormesh(X[r], Z[r], data['u'][2], vmin=0, vmax=2, cmap='plasma', rasterized=True), data['t']
            return im

        for i, ax in zip(indices, axs):
            im, t = plot_single(i, ax)
            fig.colorbar(im, caxs[ax], label=f'$T(t={{{t:.1f}}})$')

        axs[-1].set_xlabel('$x$')
        axs[-1].set_ylabel('$z$')
        fig.savefig('plots/RBC_large_series.pdf', bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    setup_mpl()
    plotter = PlotRBC(256, [1, 4, 1], '.')
    # plotter = PlotRBC(4096, [1, 4, 1024], '/p/scratch/ccstma/baumann7/large_runs/')
    # plotter.plot_series()
    plotter.plot_verification()
