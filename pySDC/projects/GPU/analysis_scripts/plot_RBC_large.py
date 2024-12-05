import pickle
import numpy as np
import matplotlib.pyplot as plt
from pySDC.helpers.stats_helper import get_sorted, get_list_of_types
from pySDC.helpers.plot_helper import figsize_by_journal, setup_mpl

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = None


class PlotRBC:
    def __init__(self, res, procs, base_path):
        self.res = res
        self.procs = procs
        self.base_path = base_path

        self._stats = None

    def get_path(self, idx, n_space, n_time=0, stats=False):
        _name = '-stats' if stats else ''
        return f'{self.base_path}/data/RayleighBenard_large-res_{self.res}-useGPU_False-procs_{self.procs[0]}_{self.procs[1]}_{self.procs[2]}-{n_time}-{n_space}-solution_{idx:06d}{_name}.pickle'

    def get_stats(self, max_frames=350):
        self._stats = {}

        frame_range = range(max_frames)
        if tqdm:
            frame_range = tqdm(frame_range)
            frame_range.set_description('Loading files')

        for idx in frame_range:
            for n_time in range(1):
                try:
                    path = self.get_path(idx=idx, n_space=0, n_time=n_time, stats=True)
                    with open(path, 'rb') as file:
                        self._stats = {**self._stats, **pickle.load(file)}
                except FileNotFoundError:
                    continue

    @property
    def stats(self):
        if self._stats is None:
            self.get_stats()
        return self._stats

    def save_fig(self, fig, name):
        path = f'plots/RBC_large_{name}.pdf'
        fig.savefig(path, bbox_inches='tight')
        print(f'Saved {path!r}', flush=True)

    def get_fig(self, *args, **kwargs):
        return plt.subplots(*args, figsize=figsize_by_journal('TUHH_thesis', 0.8, 0.6), **kwargs)

    def plot_work(self):
        fig, ax = self.get_fig()
        for key in ['factorizations', 'rhs']:
            work = get_sorted(self.stats, type=f'work_{key}')
            ax.plot([me[0] for me in work], np.cumsum([4 * me[1] for me in work]), label=fr'\#{key}')
        ax.set_yscale('log')
        ax.set_xlabel('$t$')
        ax.legend(frameon=False)
        self.save_fig(fig, 'work')

    def plot_step_size(self):
        fig, ax = self.get_fig()
        nu = get_sorted(self.stats, type='dt', recomputed=False)
        ax.plot([me[0] for me in nu], [me[1] for me in nu])
        ax.set_yscale('log')
        ax.set_xlabel('$t$')
        ax.set_ylabel(r'$\Delta t$')
        self.save_fig(fig, 'dt')

    def plot_verification(self):
        fig, ax = self.get_fig()

        nu = get_sorted(self.stats, type='Nusselt', recomputed=False)
        for key in ['t', 'b', 'V']:
            ax.plot([me[0] for me in nu], [me[1][key] for me in nu], label=fr'$Nu_\mathrm{{{key}}}$')
        ax.legend(frameon=False)
        ax.set_xlabel('$t$')
        self.save_fig(fig, 'verification')

    def plot_series(self):
        indices = [0, 56, 70, 82, 100, 132]

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

        frame_range = range(self.procs[2])
        if tqdm:
            frame_range = tqdm(frame_range)
            frame_range.set_description('Loading grid')

        for r in frame_range:
            path = self.get_path(idx=0, n_space=r)
            with open(path, 'rb') as file:
                data = pickle.load(file)
            X[r] = data['X']
            Z[r] = data['Z']

        if tqdm:
            frame_range.set_description('Plotting slice')

        def plot_single(idx, ax):
            for r in frame_range:
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
        self.save_fig(fig, 'series')


if __name__ == '__main__':
    setup_mpl()
    plotter = PlotRBC(128, [1, 1, 4], '.')
    # plotter = PlotRBC(4096, [1, 4, 1024], '/p/scratch/ccstma/baumann7/large_runs/')
    plotter.plot_work()
    plotter.plot_step_size()
    plotter.plot_verification()
    plotter.plot_series()
    plt.show()
