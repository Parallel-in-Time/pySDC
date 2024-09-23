from pySDC.projects.GPU.configs.base_config import Config


def get_config(args):
    name = args['config']

    if name == 'GS':
        return GrayScott(args)
    elif name == 'GS_dt':
        return GrayScott_dt_adaptivity(args)
    elif name == 'GS_GoL':
        return GrayScott_GoL(args)
    else:
        return NotImplementedError(f'Don\'t know config {name}')


def get_A_B_from_f_k(f, k):
    return {'A': f, 'B': f + k}


class GrayScott(Config):
    Tend = 6000
    num_frames = 200
    sweeper_type = 'IMEX'
    res_per_blob = 2**7

    def get_LogToFile(self, ranks=None):
        import numpy as np
        from pySDC.implementations.hooks.log_solution import LogToFileAfterXs as LogToFile

        LogToFile.path = './data/'
        LogToFile.file_name = f'{self.get_path(ranks=ranks)}-solution'
        LogToFile.time_increment = self.Tend / self.num_frames

        def process_solution(L):
            P = L.prob

            if P.spectral:
                uend = P.itransform(L.uend)
            else:
                uend = L.uend

            if P.useGPU:
                return {
                    't': L.time + L.dt,
                    'u': uend[0].get().view(np.ndarray),
                    'v': uend[1].get().view(np.ndarray),
                    'X': L.prob.X[0].get().view(np.ndarray),
                    'Y': L.prob.X[1].get().view(np.ndarray),
                }
            else:
                return {
                    't': L.time + L.dt,
                    'u': uend[0],
                    'v': uend[1],
                    'X': L.prob.X[0],
                    'Y': L.prob.X[1],
                }

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

    def plot(self, P, idx, n_procs_list):  # pragma: no cover
        import numpy as np
        from matplotlib import ticker as tkr

        fig = P.get_fig(n_comps=1)
        cax = P.cax
        ax = fig.get_axes()[0]

        buffer = {}
        vmin = {'u': np.inf, 'v': np.inf}
        vmax = {'u': -np.inf, 'v': -np.inf}

        for rank in range(n_procs_list[2]):
            ranks = [0, 0] + [rank]
            LogToFile = self.get_LogToFile(ranks=ranks)

            buffer[f'u-{rank}'] = LogToFile.load(idx)

            vmin['v'] = min([vmin['v'], buffer[f'u-{rank}']['v'].real.min()])
            vmax['v'] = max([vmax['v'], buffer[f'u-{rank}']['v'].real.max()])
            vmin['u'] = min([vmin['u'], buffer[f'u-{rank}']['u'].real.min()])
            vmax['u'] = max([vmax['u'], buffer[f'u-{rank}']['u'].real.max()])

        for rank in range(n_procs_list[2]):
            im = ax.pcolormesh(
                buffer[f'u-{rank}']['X'],
                buffer[f'u-{rank}']['Y'],
                buffer[f'u-{rank}']['v'].real,
                vmin=vmin['v'],
                vmax=vmax['v'],
                cmap='binary',
            )
            fig.colorbar(im, cax, format=tkr.FormatStrFormatter('%.1f'))
            ax.set_title(f't={buffer[f"u-{rank}"]["t"]:.2f}')
            ax.set_xlabel('$x$')
            ax.set_ylabel('$y$')
            ax.set_aspect(1.0)
            ax.set_aspect(1.0)
        return fig

    def get_description(self, *args, res=-1, **kwargs):
        from pySDC.implementations.problem_classes.GrayScott_MPIFFT import grayscott_imex_diffusion

        description = super().get_description(*args, **kwargs)

        description['step_params']['maxiter'] = 5

        description['level_params']['dt'] = 1e0
        description['level_params']['restol'] = -1

        description['sweeper_params']['quad_type'] = 'RADAU-RIGHT'
        description['sweeper_params']['num_nodes'] = 3
        description['sweeper_params']['QI'] = 'MIN-SR-S'
        description['sweeper_params']['QE'] = 'PIC'

        description['problem_params']['nvars'] = (2**8 if res == -1 else res,) * 2
        description['problem_params']['Du'] = 0.00002
        description['problem_params']['Dv'] = 0.00001
        description['problem_params']['A'] = 0.04
        description['problem_params']['B'] = 0.1
        description['problem_params']['L'] = 2 * description['problem_params']['nvars'][0] // self.res_per_blob
        description['problem_params']['num_blobs'] = description['problem_params']['nvars'][0] // self.res_per_blob

        description['problem_class'] = grayscott_imex_diffusion

        return description


class GrayScott_dt_adaptivity(GrayScott):
    def get_description(self, *args, **kwargs):
        from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity

        desc = super().get_description(*args, **kwargs)
        desc['convergence_controllers'][Adaptivity] = {'e_tol': 1e-5}
        return desc


class GrayScott_GoL(GrayScott):
    num_frames = 400
    res_per_blob = 2**8

    def get_description(self, *args, **kwargs):
        from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity

        desc = super().get_description(*args, **kwargs)
        desc['problem_params'] = {**desc['problem_params'], **get_A_B_from_f_k(f=0.010, k=0.049)}
        desc['convergence_controllers'][Adaptivity] = {'e_tol': 1e-5}
        self.Tend = 10000
        return desc
