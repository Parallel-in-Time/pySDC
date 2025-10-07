import pickle
import numpy as np
import matplotlib.pyplot as plt
from pySDC.helpers.stats_helper import get_sorted, get_list_of_types
from pySDC.helpers.plot_helper import figsize_by_journal, setup_mpl
from mpi4py import MPI

comm = MPI.COMM_WORLD

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = None


class PlotLargeRun:  # pragma: no cover
    name = None

    def __init__(self, res, procs, base_path, max_frames=500, useGPU='False'):
        self.res = res
        self.procs = procs
        self.base_path = base_path
        self.max_frames = max_frames
        self.useGPU = useGPU

        self._stats = None
        self._prob = None

    def get_stats(self):
        self._stats = {}

        frame_range = range(self.max_frames)
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

    def get_fig(self, *args, **kwargs):
        return plt.subplots(*args, figsize=figsize_by_journal('TUHH_thesis', 0.8, 0.6), **kwargs)

    def save_fig(self, fig, name):
        path = f'plots/{self.name}_{name}.pdf'
        fig.savefig(path, bbox_inches='tight', dpi=300)
        print(f'Saved {path!r}', flush=True)

    @property
    def prob(self):
        if self._prob is None:
            self._prob = self.get_problem()
        return self._prob

    def plot_work(self):  # pragma: no cover
        fig, ax = self.get_fig()
        for key, label in zip(['factorizations', 'rhs'], ['LU decompositions', 'rhs evaluations']):
            work = get_sorted(self.stats, type=f'work_{key}')
            ax.plot([me[0] for me in work], np.cumsum([4 * me[1] for me in work]), label=fr'\#{label}')
        ax.set_yscale('log')
        ax.set_xlabel('$t$')
        ax.legend(frameon=False)
        self.save_fig(fig, 'work')

    def plot_residual(self):  # pragma: no cover
        fig, ax = self.get_fig()
        residual = get_sorted(self.stats, type='residual_post_step', recomputed=False)
        increment = get_sorted(self.stats, type='error_embedded_estimate', recomputed=False)
        ax.plot([me[0] for me in residual], [me[1] for me in residual], label=r'residual')
        ax.plot([me[0] for me in increment], [me[1] for me in increment], label=r'$\epsilon$')
        ax.set_yscale('log')
        ax.set_xlabel('$t$')
        ax.legend(frameon=False)
        self.save_fig(fig, 'residual')

    def get_problem(self):
        raise NotImplementedError()


class PlotRBC(PlotLargeRun):  # pragma: no cover
    name = 'RBC_large'

    def get_path(self, idx, n_space, n_time=0, stats=False):
        _name = '-stats' if stats else ''
        return f'{self.base_path}/data/RayleighBenard_large-res_{self.res}-useGPU_{self.useGPU}-procs_{self.procs[0]}_{self.procs[1]}_{self.procs[2]}-{n_time}-{n_space}-solution_{idx:06d}{_name}.pickle'

    def plot_verification(self):  # pragma: no cover
        fig, ax = self.get_fig()

        nu = get_sorted(self.stats, type='Nusselt', recomputed=False)
        for key in ['t', 'b', 'V']:
            ax.plot([me[0] for me in nu], [me[1][key] for me in nu], label=fr'$Nu_\mathrm{{{key}}}$')
        ax.legend(frameon=False)
        ax.set_xlabel('$t$')
        self.save_fig(fig, 'verification')

    def plot_step_size(self):  # pragma: no cover
        fig, ax = self.get_fig()
        dt = get_sorted(self.stats, type='dt', recomputed=False)
        CFL = self.get_CFL_limit()

        ax.plot([me[0] for me in dt], [me[1] for me in dt], label=r'$\Delta t$')
        ax.plot(CFL.keys(), CFL.values(), label='CFL limit')
        ax.set_yscale('log')
        ax.set_xlabel('$t$')
        ax.set_ylabel(r'$\Delta t$')
        ax.legend(frameon=False)
        self.save_fig(fig, 'dt')

    def get_problem(self):
        from pySDC.projects.GPU.configs.RBC_configs import get_config
        from pySDC.implementations.problem_classes.RayleighBenard import RayleighBenard

        config = get_config(
            {
                'config': 'RBC_large',
                'procs': [1, 1, self.procs[2]],
                'res': self.res,
                'mode': None,
                'useGPU': False,
            }
        )
        desc = config.get_description(res=self.res)

        return RayleighBenard(**{**desc['problem_params'], 'comm': comm})

    def _compute_CFL_single_frame(self, frame):
        from pySDC.implementations.problem_classes.RayleighBenard import CFLLimit

        u = self.prob.u_init

        path = self.get_path(idx=frame, n_space=comm.rank)
        with open(path, 'rb') as file:
            data = pickle.load(file)

        for i in range(u.shape[0]):
            u[i] = data['u'][i]

        CFL = CFLLimit.compute_max_step_size(self.prob, u)
        return {'CFL': CFL, 't': data['t']}

    def compute_CFL_limit(self):
        frame_range = range(self.max_frames)

        if tqdm and comm.rank == 0:
            frame_range = tqdm(frame_range)
            frame_range.set_description('Computing CFL')

        CFL = {}
        for frame in frame_range:
            try:
                _cfl = self._compute_CFL_single_frame(frame)
                CFL[_cfl['t']] = _cfl['CFL']
            except FileNotFoundError:
                pass

        if comm.rank == 0:
            with open(self._get_CFL_limit_path(), 'wb') as file:
                pickle.dump(CFL, file)
            print(f'Stored {self._get_CFL_limit_path()!r}')
        return CFL

    def _get_CFL_limit_path(self):
        return f'{self.base_path}/data/RayleighBenard_large-res_{self.res}-useGPU_{self.useGPU}-procs_{self.procs[0]}_{self.procs[1]}_{self.procs[2]}-CFL_limit.pickle'

    def get_CFL_limit(self, recompute=False):
        import os

        path = self._get_CFL_limit_path()

        if os.path.exists(path) and not recompute:
            with open(path, 'rb') as file:
                CFL = pickle.load(file)
        else:
            CFL = self.compute_CFL_limit()
            with open(path, 'wb') as file:
                pickle.dump(CFL, file)
            print(f'Stored {path!r}')
        return CFL

    def plot_work(self):  # pragma: no cover
        fig, ax = self.get_fig()
        for key, label in zip(['factorizations', 'rhs'], ['LU decompositions', 'rhs evaluations']):
            work = get_sorted(self.stats, type=f'work_{key}')
            ax.plot([me[0] for me in work], np.cumsum([4 * me[1] for me in work]), label=fr'\#{label}')
        ax.set_yscale('log')
        ax.set_xlabel('$t$')
        ax.legend(frameon=False)
        self.save_fig(fig, 'work')

    def plot_residual(self):  # pragma: no cover
        fig, ax = self.get_fig()
        residual = get_sorted(self.stats, type='residual_post_step', recomputed=False)
        increment = get_sorted(self.stats, type='error_embedded_estimate', recomputed=False)
        ax.plot([me[0] for me in residual], [me[1] for me in residual], label=r'residual')
        ax.plot([me[0] for me in increment], [me[1] for me in increment], label=r'$\epsilon$')
        ax.set_yscale('log')
        ax.set_xlabel('$t$')
        ax.legend(frameon=False)
        self.save_fig(fig, 'residual')

    def plot_series(self):  # pragma: no cover
        indices = [0, 56, 82, 100, 162, 186]

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

        def plot_single(idx, ax):  # pragma: no cover
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


class PlotGS(PlotLargeRun):  # pragma: no cover
    name = 'GS_large'

    def get_path(self, idx, n_space, n_time=0, stats=False):
        _name = '-stats' if stats else ''
        return f'{self.base_path}/data/GrayScottLarge-res_{self.res}-useGPU_{self.useGPU}-procs_{self.procs[0]}_{self.procs[1]}_{self.procs[2]}-{n_time}-{n_space}-solution_{idx:06d}{_name}.pickle'

    def plot_step_size(self):  # pragma: no cover
        fig, ax = self.get_fig()
        dt = get_sorted(self.stats, type='dt', recomputed=False)

        ax.plot([me[0] for me in dt], [me[1] for me in dt], label=r'$\Delta t$')
        ax.set_xlabel('$t$')
        ax.set_ylabel(r'$\Delta t$')
        ax.legend(frameon=False)
        self.save_fig(fig, 'dt')

    def plot_series(self, test=False):  # pragma: no cover
        if test:
            indices = [0, 1, 2, 3, 4, 5]
            process = 0
            layer = 0
        else:
            indices = [91]  # [0, 10, 20, 30, 40, 91]
            process = 120  # 141#20#96  # 11
            layer = 6

        from mpl_toolkits.axes_grid1 import make_axes_locatable

        # get grid
        path = self.get_path(idx=0, n_space=process)
        with open(path, 'rb') as file:
            data = pickle.load(file)
        X = data['X']

        if tqdm:
            frame_range = tqdm(indices)
            frame_range.set_description('Plotting slice')

        for frame in frame_range:

            plt.rcParams['figure.constrained_layout.use'] = True
            fig, ax = plt.subplots(1, 1, figsize=figsize_by_journal('TUHH_thesis', 0.7, 1.1))
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='3%', pad=0.03)

            path = self.get_path(idx=frame, n_space=process)
            with open(path, 'rb') as file:
                data = pickle.load(file)

            # im = ax.pcolormesh(X[1][0], X[2][0], np.max(data['v'], axis=0), cmap='binary', rasterized=True, vmin=0, vmax=0.5)
            # im = ax.pcolormesh(X[1][layer], X[2][layer], data['v'][layer], cmap='binary', rasterized=True, vmin=0, vmax=0.5)
            im = ax.pcolormesh(
                X[1][layer], X[2][layer], data['v'][layer], cmap='binary', rasterized=True, vmin=0, vmax=0.5
            )
            ax.set_xlim((9, 14))
            ax.set_ylim((-4, 1))

            fig.colorbar(im, cax)  # , label=f'$T(t={{{t:.1f}}})$')

            ax.set_xlabel('$y$')
            ax.set_ylabel('$z$')
            ax.set_aspect(1)
            self.save_fig(fig, f'series_{frame}_t={data["t"]:.0f}')


if __name__ == '__main__':  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', type=str, default='GS', choices=['RBC', 'GS'])
    parser.add_argument('--config', type=str, default='production', choices=['production', 'test'])
    args = parser.parse_args()

    setup_mpl()

    if args.problem == 'RBC':
        if args.config == 'test':
            plotter = PlotRBC(256, [1, 4, 1], '.', 100)
        else:
            plotter = PlotRBC(4096, [1, 4, 1024], '/p/scratch/ccstma/baumann7/large_runs/', 200)
    elif args.problem == 'GS':
        if args.config == 'test':
            plotter = PlotGS(64, [1, 1, 4], '.', 100)
        else:
            plotter = PlotGS(2304, [1, 4, 192], '/p/scratch/ccstma/baumann7/large_runs/', 91, 'True')

    if args.problem == 'RBC':
        if comm.size > 1:
            plotter.compute_CFL_limit()
            exit()

        plotter.plot_residual()
        plotter.plot_step_size()
        plotter.plot_work()
        plotter.plot_verification()
        plotter.plot_series()
    else:
        plotter.plot_series(args.config == 'test')
        plotter.plot_step_size()

    plt.show()
