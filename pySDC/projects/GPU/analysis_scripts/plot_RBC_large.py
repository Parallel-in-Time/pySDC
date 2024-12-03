import pickle
import numpy
import matplotlib.pyplot as plt
from pySDC.helpers.stats_helper import get_sorted, get_list_of_types
from pySDC.helpers.plot_helper import figsize_by_journal


class PlotRBC:
    def __init__(self, res, procs, base_path):
        self.res = res
        self.procs = procs
        self.base_path = base_path

    def get_path(self, idx, n_space, stats=False):
        _name = '-stats' if stats else ''
        return f'./data/RayleighBenard_large-res_{self.res}-useGPU_False-procs_{self.procs[0]}_{self.procs[1]}_{self.procs[2]}-0-{n_space}-solution_{idx:06d}{_name}.pickle'

    def plot_verification():
        res = 128
        procs = [1, 1, 4]
        path = f'./data/RayleighBenard_large-res_{res}-useGPU_False-procs_{procs[0]}_{procs[1]}_{procs[2]}-0-0-solution_000002-stats.pickle'
        with open(path, 'rb') as file:
            data = pickle.load(file)

        print(get_list_of_types(data))

    def plot_series(self):
        indices = [0, 5, 10, 15]

        fig, axs = plt.subplots(4, 1, figsize=figsize_by_journal('TUHH_thesis', 1, 1), sharex=True, sharey=True)

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
            return ax.pcolormesh(X[r], Z[r], data['u'][3], vmin=0, vmax=2, cmap='plasma', rasterized=True)

        for i, ax in zip(indices, axs):
            plot_single(i, ax)
        fig.savefig('plots/RBC_large_series.pdf', bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    plotter = PlotRBC(128, [1, 1, 4], '.')
    plotter.plot_series()
