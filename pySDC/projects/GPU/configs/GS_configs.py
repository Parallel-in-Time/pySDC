from pySDC.projects.GPU.configs.base_config import Config
from mpi4py_fft.distarray import newDistArray


def get_config(args):
    name = args['config']

    if name == 'GS':
        return GrayScott(args)
    elif name == 'GS_dt':
        return GrayScott_dt_adaptivity(args)
    elif name == 'GS_GoL':
        return GrayScott_GoL(args)
    elif name == 'GS_USkate':
        return GrayScott_USkate(args)
    elif name == 'GS_scaling':
        return GrayScottScaling(args)
    elif name == 'GS_scaling3D':
        return GrayScottScaling3D(args)
    elif name == 'GS_large':
        return GrayScottLarge(args)
    else:
        return NotImplementedError(f'Don\'t know config {name}')


def get_A_B_from_f_k(f, k):
    return {'A': f, 'B': f + k}


class GrayScott(Config):
    Tend = 6000
    num_frames = 200
    sweeper_type = 'IMEX'
    res_per_blob = 2**7
    ndim = 3

    def get_LogToFile(self, ranks=None):
        import numpy as np
        from pySDC.implementations.hooks.log_solution import LogToFileAfterXs as LogToFile

        LogToFile.path = f'{self.base_path}/data/'
        LogToFile.file_name = f'{self.get_path(ranks=ranks)}-solution'
        LogToFile.time_increment = self.Tend / self.num_frames

        def process_solution(L):
            P = L.prob

            if P.spectral:
                tmpu = newDistArray(P.fft, False)
                tmpv = newDistArray(P.fft, False)
                tmpu[:] = P.fft.backward(L.uend[0, ...], tmpu)
                tmpv[:] = P.fft.backward(L.uend[1, ...], tmpv)
                uend = P.xp.stack((tmpu, tmpv))
            else:
                uend = L.uend

            data = {
                't': L.time + L.dt,
                'local_slice': P.fft.local_slice(False),
                'shape': P.fft.global_shape(False),
            }

            if P.useGPU:
                data['u'] = uend[0].get().view(np.ndarray)
                data['v'] = uend[1].get().view(np.ndarray)
                if L.time == 0:
                    data['X'] = [me.get().view(np.ndarray) for me in L.prob.X]
            else:
                data['u'] = uend[0]
                data['v'] = uend[1]
                if L.time == 0:
                    data['X'] = L.prob.X
            return data

        def logging_condition(L):
            sweep = L.sweep
            if hasattr(sweep, 'comm'):
                if sweep.comm.rank == sweep.comm.size - 1:
                    return True
                else:
                    return False
            else:
                return True

        LogToFile.process_solution = process_solution
        LogToFile.logging_condition = logging_condition
        return LogToFile

    def plot(self, P, idx, n_procs_list, projection=0, projection_type='flat'):  # pragma: no cover
        import numpy as np
        from matplotlib import ticker as tkr

        fig = P.get_fig(n_comps=1)
        cax = P.cax
        ax = fig.get_axes()[0]

        buffer = {}
        vmin = {'u': np.inf, 'v': np.inf}
        vmax = {'u': -np.inf, 'v': -np.inf}

        for rank in range(n_procs_list[2]):
            # for rank in [n_procs_list[2] // 2]:
            ranks = [0, 0] + [rank]
            LogToFile = self.get_LogToFile(ranks=ranks)

            buffer[f'u-{rank}'] = LogToFile.load(idx)

            vmin['v'] = min([vmin['v'], buffer[f'u-{rank}']['v'].real.min()])
            vmax['v'] = max([vmax['v'], buffer[f'u-{rank}']['v'].real.max()])
            vmin['u'] = min([vmin['u'], buffer[f'u-{rank}']['u'].real.min()])
            vmax['u'] = max([vmax['u'], buffer[f'u-{rank}']['u'].real.max()])

        for rank in range(n_procs_list[2]):
            if len(buffer[f'u-{rank}']['X']) == 2:
                ax.set_xlabel('$x$')
                ax.set_ylabel('$y$')
                im = ax.pcolormesh(
                    buffer[f'u-{rank}']['X'][0],
                    buffer[f'u-{rank}']['X'][1],
                    buffer[f'u-{rank}']['v'].real,
                    vmin=vmin['v'],
                    vmax=vmax['v'],
                    cmap='binary',
                )
            else:
                v3d = buffer[f'u-{rank}']['v'].real

                if projection == 2:
                    slices = [slice(None), slice(None), v3d.shape[2] // 2]
                    x = buffer[f'u-{rank}']['X'][0][*slices]
                    y = buffer[f'u-{rank}']['X'][1][*slices]
                    ax.set_xlabel('$x$')
                    ax.set_ylabel('$y$')
                elif projection == 1:
                    slices = [slice(None), v3d.shape[1] // 2, slice(None)]
                    x = buffer[f'u-{rank}']['X'][0][*slices]
                    y = buffer[f'u-{rank}']['X'][2][*slices]
                    ax.set_xlabel('$x$')
                    ax.set_ylabel('$z$')
                elif projection == 0:
                    slices = [v3d.shape[0] // 2, slice(None), slice(None)]
                    x = buffer[f'u-{rank}']['X'][1][*slices]
                    y = buffer[f'u-{rank}']['X'][2][*slices]
                    ax.set_xlabel('$y$')
                    ax.set_ylabel('$z$')

                if projection_type == 'sum':
                    v = v3d.sum(axis=projection)
                else:
                    v = v3d[*slices]

                im = ax.pcolormesh(
                    x,
                    y,
                    v,
                    vmin=vmin['v'],
                    vmax=vmax['v'],
                    cmap='binary',
                )
            fig.colorbar(im, cax, format=tkr.FormatStrFormatter('%.1f'))
            ax.set_title(f't={buffer[f"u-{rank}"]["t"]:.2f}')
            ax.set_aspect(1.0)
        return fig

    def get_description(self, *args, res=-1, **kwargs):
        from pySDC.implementations.problem_classes.GrayScott_MPIFFT import grayscott_imex_diffusion

        desc = super().get_description(*args, **kwargs)

        desc['step_params']['maxiter'] = 5

        desc['level_params']['dt'] = 1e0
        desc['level_params']['restol'] = -1

        desc['sweeper_params']['quad_type'] = 'RADAU-RIGHT'
        desc['sweeper_params']['num_nodes'] = 3
        desc['sweeper_params']['QI'] = 'MIN-SR-S'
        desc['sweeper_params']['QE'] = 'PIC'

        desc['problem_params']['nvars'] = (2**8 if res == -1 else res,) * self.ndim
        desc['problem_params']['Du'] = 0.00002
        desc['problem_params']['Dv'] = 0.00001
        desc['problem_params']['A'] = 0.04
        desc['problem_params']['B'] = 0.1
        desc['problem_params']['L'] = 2 * desc['problem_params']['nvars'][0] // self.res_per_blob
        desc['problem_params']['num_blobs'] = desc['problem_params']['nvars'][0] // self.res_per_blob

        desc['problem_class'] = grayscott_imex_diffusion

        return desc


class GrayScott_dt_adaptivity(GrayScott):
    """
    Configuration with dt adaptivity added to base configuration
    """

    ndim = 2

    def get_description(self, *args, **kwargs):
        from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity

        desc = super().get_description(*args, **kwargs)
        desc['convergence_controllers'][Adaptivity] = {'e_tol': 1e-5}
        return desc


class GrayScott_GoL(GrayScott):
    '''
    This configuration shows gliders that are similar in complexity to Conway's Game of life.
    '''

    num_frames = 400
    res_per_blob = 2**8

    def get_description(self, *args, **kwargs):
        from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity

        desc = super().get_description(*args, **kwargs)
        desc['problem_params'] = {**desc['problem_params'], **get_A_B_from_f_k(f=0.010, k=0.049)}
        desc['convergence_controllers'][Adaptivity] = {'e_tol': 1e-5}
        self.Tend = 10000
        return desc


class GrayScott_USkate(GrayScott):
    '''
    See arXiv:1501.01990 or http://www.mrob.com/sci/papers/2009smp-figs/index.html
    '''

    num_frames = 200
    res_per_blob = 2**7
    Tend = 200000

    def get_description(self, *args, **kwargs):
        from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity

        desc = super().get_description(*args, **kwargs)
        desc['problem_params'] = {**desc['problem_params'], **get_A_B_from_f_k(f=0.062, k=0.0609)}
        desc['problem_params']['num_blobs'] = -12 * desc['problem_params']['L'] ** 2
        desc['problem_params']['Du'] = 2e-5
        desc['problem_params']['Dv'] = 1e-5
        desc['convergence_controllers'][Adaptivity] = {'e_tol': 1e-3}
        return desc


class GrayScottScaling(GrayScott):
    def get_description(self, *args, **kwargs):
        desc = super().get_description(*args, **kwargs)
        desc['problem_params']['L'] = 2
        desc['problem_params']['num_blobs'] = 4
        desc['sweeper_params']['skip_residual_computation'] = ('IT_CHECK', 'IT_DOWN', 'IT_UP', 'IT_FINE', 'IT_COARSE')
        desc['sweeper_params']['num_nodes'] = 4
        self.Tend = 50 * desc['level_params']['dt']
        return desc

    def get_controller_params(self, *args, **kwargs):
        params = super().get_controller_params(*args, **kwargs)
        params['hook_class'] = []
        return params


class GrayScottScaling3D(GrayScottScaling):
    ndim = 3
    nsteps = 15

    def get_description(self, *args, **kwargs):
        from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity

        desc = super().get_description(*args, **kwargs)
        desc['problem_params']['L'] = 2
        desc['problem_params']['num_blobs'] = 4
        desc['sweeper_params']['skip_residual_computation'] = ('IT_CHECK', 'IT_DOWN', 'IT_UP', 'IT_FINE', 'IT_COARSE')
        desc['sweeper_params']['num_nodes'] = 4
        desc['step_params']['maxiter'] = 4
        desc['level_params']['dt'] = 0.1
        self.Tend = self.nsteps * desc['level_params']['dt']
        return desc

    def get_controller_params(self, *args, **kwargs):
        params = super().get_controller_params(*args, **kwargs)
        params['hook_class'] = []
        return params


class GrayScottLarge(GrayScott):
    Tend = 5000
    num_frames = 100
    res_per_blob = 2**7
    ndim = 3

    def get_description(self, *args, **kwargs):
        from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity

        desc = super().get_description(*args, **kwargs)
        desc['sweeper_params']['skip_residual_computation'] = ('IT_CHECK', 'IT_DOWN', 'IT_UP', 'IT_FINE', 'IT_COARSE')
        desc['sweeper_params']['num_nodes'] = 4
        desc['sweeper_params']['QI'] = 'MIN-SR-S'
        desc['sweeper_params']['QE'] = 'PIC'
        desc['step_params']['maxiter'] = 4
        desc['level_params']['dt'] = 1e-1
        desc['problem_params']['spectral'] = False

        # desc['problem_params']['num_blobs'] *= -1
        # desc['problem_params']['num_blobs'] = 40

        desc['problem_params']['L'] = 2 * desc['problem_params']['nvars'][0] // self.res_per_blob
        desc['problem_params']['num_blobs'] = int(1 * desc['problem_params']['L'])

        desc['convergence_controllers'][Adaptivity] = {'e_tol': 1e-3}
        return desc

    def plot(self, P, idx, n_procs_list, projection=2, projection_type='flat'):  # pragma: no cover
        import numpy as np
        from matplotlib import ticker as tkr
        import matplotlib.pyplot as plt
        import gc

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        thresh = 0.25

        data = self.get_LogToFile().load(0)

        min_pos = [-20, 2, 12]
        # box_size = [4, 8, 8]
        box_size = [
            4,
        ] * 3

        ax.set_xlim(min_pos[0], min_pos[0] + box_size[0])
        ax.set_ylim(min_pos[1], min_pos[1] + box_size[1])
        ax.set_zlim(min_pos[2], min_pos[2] + box_size[2])
        ax.set_aspect('equal')

        grid = None
        for rank in range(n_procs_list[2]):
            gc.collect()
            ranks = [0, 0] + [rank]
            LogToFile = self.get_LogToFile(ranks=ranks)

            data = LogToFile.load(idx)
            u = data['v']
            grid = data['X']
            ax.set_title(f't={data["t"]:.2f}')

            x = grid[0][:, 0, 0]
            y = grid[1][0, :, 0]
            z = grid[2][0, 0, :]
            grids1d = [x, y, z]

            if min_pos[0] > x.max():
                continue
            elif (min_pos[0] + box_size[0]) < x.min():
                break

            slice_starts = [np.searchsorted(grids1d[i], min_pos[i]) for i in range(self.ndim)]
            slice_ends = [np.searchsorted(grids1d[i], min_pos[i] + box_size[i]) for i in range(self.ndim)]
            slices = [slice(slice_starts[i], slice_ends[i]) for i in range(self.ndim)]
            slice_data = [slice(slice_starts[i] + 1, slice_ends[i]) for i in range(self.ndim)]

            if any(abs(slice_starts[i] - slice_ends[i]) <= 1 for i in range(self.ndim)):
                continue

            mask = u > thresh

            if mask.any():
                filled = np.zeros_like(u).astype(bool)
                filled[mask] = True
                ax.voxels(
                    grid[0][*slices],
                    grid[1][*slices],
                    grid[2][*slices],
                    filled[*slice_data],
                    alpha=0.5,
                    facecolors='teal',
                )
                # ax.scatter(grid[0][mask], grid[1][mask], grid[2][mask], alpha=0.1, color='black', marker='.')

            gc.collect()

        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_zlabel('$z$')
        return fig

    def get_initial_condition(self, P, *args, restart_idx=0, **kwargs):
        if restart_idx > 0:
            return super().get_initial_condition(P, *args, restart_idx=restart_idx, **kwargs)
        else:
            _u0 = P.u_exact(t=0)
            return _u0, 0
