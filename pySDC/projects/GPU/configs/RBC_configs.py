from pySDC.projects.GPU.configs.base_config import Config


def get_config(args):
    name = args['config']
    if name == 'RBC':
        return RayleighBenardRegular(args)
    elif name == 'RBC_dt':
        return RayleighBenard_dt_adaptivity(args)
    elif name == 'RBC_k':
        return RayleighBenard_k_adaptivity(args)
    elif name == 'RBC_dt_k':
        return RayleighBenard_dt_k_adaptivity(args)
    elif name == 'RBC_RK':
        return RayleighBenardRK(args)
    elif name == 'RBC_dedalus':
        return RayleighBenardDedalusComp(args)
    elif name == 'RBC_Tibo':
        return RayleighBenard_Thibaut(args)
    elif name == 'RBC_scaling':
        return RayleighBenard_scaling(args)
    elif name == 'RBC_large':
        return RayleighBenard_large(args)
    else:
        raise NotImplementedError(f'There is no configuration called {name!r}!')


class RayleighBenardRegular(Config):
    sweeper_type = 'IMEX'
    Tend = 50

    def get_LogToFile(self, ranks=None):
        import numpy as np
        from pySDC.implementations.hooks.log_solution import LogToFileAfterXs as LogToFile

        LogToFile.path = f'{self.base_path}/data/'
        LogToFile.file_name = f'{self.get_path(ranks=ranks)}-solution'
        LogToFile.time_increment = 1e-1

        def process_solution(L):
            P = L.prob

            if P.spectral_space:
                uend = P.itransform(L.uend)

            data = {
                't': L.time + L.dt,
                'local_slice': P.local_slice,
                'shape': P.global_shape,
            }

            if P.useGPU:
                data['u'] = uend.get().view(np.ndarray)
                data['vorticity'] = L.prob.compute_vorticity(L.uend).get().view(np.ndarray)
                if L.time == 0:
                    data['X'] = L.prob.X.get()
                    data['Z'] = L.prob.Z.get()
            else:
                data['u'] = uend.view(np.ndarray)
                data['vorticity'] = L.prob.compute_vorticity(L.uend).view(np.ndarray)
                if L.time == 0:
                    data['X'] = L.prob.X
                    data['Z'] = L.prob.Z
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

    def get_controller_params(self, *args, **kwargs):
        from pySDC.implementations.hooks.log_step_size import LogStepSize

        controller_params = super().get_controller_params(*args, **kwargs)
        controller_params['hook_class'] += [LogStepSize]
        return controller_params

    def get_description(self, *args, MPIsweeper=False, res=-1, **kwargs):
        from pySDC.implementations.problem_classes.RayleighBenard import (
            RayleighBenard,
            CFLLimit,
        )
        from pySDC.implementations.problem_classes.generic_spectral import (
            compute_residual_DAE,
            compute_residual_DAE_MPI,
        )
        from pySDC.implementations.convergence_controller_classes.step_size_limiter import StepSizeSlopeLimiter

        desc = super().get_description(*args, MPIsweeper=MPIsweeper, **kwargs)

        if MPIsweeper:
            desc['sweeper_class'].compute_residual = compute_residual_DAE_MPI
        else:
            desc['sweeper_class'].compute_residual = compute_residual_DAE

        desc['level_params']['dt'] = 0.1
        desc['level_params']['restol'] = 1e-7

        desc['convergence_controllers'][CFLLimit] = {'dt_max': 0.1, 'dt_min': 1e-6, 'cfl': 0.8}
        desc['convergence_controllers'][StepSizeSlopeLimiter] = {'dt_rel_min_slope': 0.1}

        desc['sweeper_params']['quad_type'] = 'RADAU-RIGHT'
        desc['sweeper_params']['num_nodes'] = 2
        desc['sweeper_params']['QI'] = 'MIN-SR-S'
        desc['sweeper_params']['QE'] = 'PIC'

        desc['problem_params']['Rayleigh'] = 2e6
        desc['problem_params']['nx'] = 2**8 if res == -1 else res
        desc['problem_params']['nz'] = desc['problem_params']['nx'] // 4
        desc['problem_params']['dealiasing'] = 3 / 2

        desc['step_params']['maxiter'] = 3

        desc['problem_class'] = RayleighBenard

        return desc

    def get_initial_condition(self, P, *args, restart_idx=0, **kwargs):
        if restart_idx > 0:
            return super().get_initial_condition(P, *args, restart_idx=restart_idx, **kwargs)
        else:
            u0 = P.u_exact(t=0, seed=P.comm.rank, noise_level=1e-3)
            u0_with_pressure = P.solve_system(u0, 1e-9, u0)
            return u0_with_pressure, 0

    def prepare_caches(self, prob):
        """
        Cache the fft objects, which are expensive to create on GPU because graphs have to be initialized.
        """
        prob.eval_f(prob.u_init)

    def plot(self, P, idx, n_procs_list, quantitiy='T', quantitiy2='vorticity'):
        import numpy as np

        cmaps = {'vorticity': 'bwr', 'p': 'bwr'}

        fig = P.get_fig()
        cax = P.cax
        axs = fig.get_axes()

        buffer = {}
        vmin = {quantitiy: np.inf, quantitiy2: np.inf}
        vmax = {quantitiy: -np.inf, quantitiy2: -np.inf}

        X = {}
        Z = {}

        for rank in range(n_procs_list[2]):
            ranks = self.ranks[:-1] + [rank]
            LogToFile = self.get_LogToFile(ranks=ranks)

            buffer[f'u-{rank}'] = LogToFile.load(idx)

            vmin[quantitiy2] = min([vmin[quantitiy2], buffer[f'u-{rank}'][quantitiy2].real.min()])
            vmax[quantitiy2] = max([vmax[quantitiy2], buffer[f'u-{rank}'][quantitiy2].real.max()])
            vmin[quantitiy] = min([vmin[quantitiy], buffer[f'u-{rank}']['u'][P.index(quantitiy)].real.min()])
            vmax[quantitiy] = max([vmax[quantitiy], buffer[f'u-{rank}']['u'][P.index(quantitiy)].real.max()])

            first_frame = LogToFile.load(0)
            X[rank] = first_frame['X']
            Z[rank] = first_frame['Z']

        for rank in range(n_procs_list[2]):
            im2 = axs[1].pcolormesh(
                X[rank],
                Z[rank],
                buffer[f'u-{rank}'][quantitiy2].real,
                vmin=-vmax[quantitiy2] if cmaps.get(quantitiy2, None) in ['bwr'] else vmin[quantitiy2],
                vmax=vmax[quantitiy2],
                cmap=cmaps.get(quantitiy2, None),
            )
            im = axs[0].pcolormesh(
                X[rank],
                Z[rank],
                buffer[f'u-{rank}']['u'][P.index(quantitiy)].real,
                vmin=vmin[quantitiy],
                vmax=-vmin[quantitiy] if cmaps.get(quantitiy, None) in ['bwr', 'seismic'] else vmax[quantitiy],
                cmap=cmaps.get(quantitiy, 'plasma'),
            )
        fig.colorbar(im2, cax[1])
        fig.colorbar(im, cax[0])
        axs[0].set_title(f't={buffer[f"u-{rank}"]["t"]:.2f}')
        axs[1].set_xlabel('x')
        axs[1].set_ylabel('z')
        axs[0].set_aspect(1.0)
        axs[1].set_aspect(1.0)
        return fig


class RayleighBenard_k_adaptivity(RayleighBenardRegular):
    def get_description(self, *args, **kwargs):
        from pySDC.implementations.convergence_controller_classes.adaptivity import AdaptivityPolynomialError
        from pySDC.implementations.problem_classes.RayleighBenard import CFLLimit

        desc = super().get_description(*args, **kwargs)

        desc['convergence_controllers'][CFLLimit] = {'dt_max': 0.1, 'dt_min': 1e-6, 'cfl': 0.8}
        desc['level_params']['restol'] = 1e-7
        desc['sweeper_params']['num_nodes'] = 4
        desc['sweeper_params']['QI'] = 'MIN-SR-S'
        desc['step_params']['maxiter'] = 12

        return desc

    def get_controller_params(self, *args, **kwargs):
        from pySDC.implementations.problem_classes.RayleighBenard import (
            LogAnalysisVariables,
        )

        controller_params = super().get_controller_params(*args, **kwargs)
        controller_params['hook_class'] = [
            me for me in controller_params['hook_class'] if me is not LogAnalysisVariables
        ]
        return controller_params


class RayleighBenard_dt_k_adaptivity(RayleighBenardRegular):
    def get_description(self, *args, **kwargs):
        from pySDC.implementations.convergence_controller_classes.adaptivity import AdaptivityPolynomialError
        from pySDC.implementations.problem_classes.RayleighBenard import CFLLimit

        desc = super().get_description(*args, **kwargs)

        desc['convergence_controllers'][AdaptivityPolynomialError] = {
            'e_tol': 1e-3,
            'abort_at_growing_residual': False,
            'interpolate_between_restarts': False,
            'dt_min': 1e-3,
            'dt_rel_min_slope': 0.1,
        }
        desc['convergence_controllers'].pop(CFLLimit)
        desc['level_params']['restol'] = 1e-7
        desc['sweeper_params']['num_nodes'] = 3
        desc['sweeper_params']['QI'] = 'MIN-SR-S'
        desc['step_params']['maxiter'] = 16
        desc['problem_params']['nx'] *= 2
        desc['problem_params']['nz'] *= 2

        return desc

    def get_controller_params(self, *args, **kwargs):
        from pySDC.implementations.problem_classes.RayleighBenard import (
            LogAnalysisVariables,
        )

        controller_params = super().get_controller_params(*args, **kwargs)
        controller_params['hook_class'] = [
            me for me in controller_params['hook_class'] if me is not LogAnalysisVariables
        ]
        return controller_params


class RayleighBenard_dt_adaptivity(RayleighBenardRegular):
    def get_description(self, *args, **kwargs):
        from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity
        from pySDC.implementations.problem_classes.RayleighBenard import CFLLimit

        desc = super().get_description(*args, **kwargs)

        desc['convergence_controllers'][Adaptivity] = {'e_tol': 1e-4, 'dt_rel_min_slope': 0.1}
        desc['convergence_controllers'].pop(CFLLimit)
        desc['level_params']['restol'] = -1
        desc['sweeper_params']['num_nodes'] = 3
        desc['sweeper_params']['skip_residual_computation'] = ('IT_CHECK', 'IT_DOWN', 'IT_UP', 'IT_FINE', 'IT_COARSE')
        desc['step_params']['maxiter'] = 5
        return desc


class RayleighBenard_Thibaut(RayleighBenardRegular):
    Tend = 1

    def get_description(self, *args, **kwargs):
        from pySDC.implementations.problem_classes.RayleighBenard import CFLLimit

        desc = super().get_description(*args, **kwargs)

        desc['convergence_controllers'].pop(CFLLimit)
        desc['level_params']['restol'] = -1
        desc['level_params']['dt'] = 2e-2 / 4
        desc['sweeper_params']['num_nodes'] = 4
        desc['sweeper_params']['QI'] = 'MIN-SR-S'
        desc['sweeper_params']['node_type'] = 'LEGENDRE'
        desc['sweeper_params']['quad_type'] = 'RADAU-RIGHT'
        desc['sweeper_params']['skip_residual_computation'] = ('IT_CHECK', 'IT_DOWN', 'IT_UP', 'IT_FINE', 'IT_COARSE')
        desc['step_params']['maxiter'] = 4
        return desc

    def get_controller_params(self, *args, **kwargs):
        controller_params = super().get_controller_params(*args, **kwargs)
        controller_params['hook_class'] = []
        return controller_params


class RayleighBenardRK(RayleighBenardRegular):

    def get_description(self, *args, **kwargs):
        from pySDC.implementations.sweeper_classes.Runge_Kutta import ARK3
        from pySDC.implementations.problem_classes.RayleighBenard import CFLLimit
        from pySDC.implementations.convergence_controller_classes.step_size_limiter import StepSizeSlopeLimiter

        desc = super().get_description(*args, **kwargs)

        desc['sweeper_class'] = ARK3

        desc['step_params']['maxiter'] = 1

        desc['convergence_controllers'][CFLLimit] = {'dt_max': 0.1, 'dt_min': 1e-6, 'cfl': 0.5}
        desc['convergence_controllers'][StepSizeSlopeLimiter] = {'dt_rel_min_slope': 0.1}
        return desc

    def get_controller_params(self, *args, **kwargs):
        from pySDC.implementations.problem_classes.RayleighBenard import (
            LogAnalysisVariables,
        )

        controller_params = super().get_controller_params(*args, **kwargs)
        controller_params['hook_class'] = [
            me for me in controller_params['hook_class'] if me is not LogAnalysisVariables
        ]
        return controller_params


class RayleighBenardDedalusComp(RayleighBenardRK):
    Tend = 150

    def get_description(self, *args, **kwargs):
        from pySDC.implementations.sweeper_classes.Runge_Kutta import ARK222

        desc = super().get_description(*args, **kwargs)

        desc['sweeper_class'] = ARK222

        desc['step_params']['maxiter'] = 1
        desc['level_params']['dt'] = 5e-3

        from pySDC.implementations.problem_classes.RayleighBenard import CFLLimit

        desc['convergence_controllers'].pop(CFLLimit)
        return desc


class RayleighBenard_scaling(RayleighBenardRegular):
    Tend = 7

    def get_description(self, *args, **kwargs):
        from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity
        from pySDC.implementations.problem_classes.RayleighBenard import CFLLimit
        from pySDC.implementations.convergence_controller_classes.step_size_limiter import (
            StepSizeRounding,
            StepSizeSlopeLimiter,
        )

        desc = super().get_description(*args, **kwargs)

        desc['convergence_controllers'][Adaptivity] = {'e_tol': 1e-3, 'dt_rel_min_slope': 1.0, 'beta': 0.5}
        desc['convergence_controllers'][StepSizeRounding] = {}
        desc['convergence_controllers'][StepSizeSlopeLimiter] = {'dt_rel_min_slope': 1.0}
        desc['convergence_controllers'].pop(CFLLimit)
        desc['level_params']['restol'] = -1
        desc['level_params']['dt'] = 8e-2
        desc['sweeper_params']['num_nodes'] = 4
        desc['step_params']['maxiter'] = 4
        desc['problem_params']['max_cached_factorizations'] = 4
        return desc

    def get_controller_params(self, *args, **kwargs):
        from pySDC.implementations.hooks.log_work import LogWork

        params = super().get_controller_params(*args, **kwargs)
        params['hook_class'] = [LogWork]
        return params


class RayleighBenard_large(RayleighBenardRegular):
    Ra = 3.2e8
    relaxation_steps = 5

    def get_description(self, *args, **kwargs):
        from pySDC.implementations.convergence_controller_classes.adaptivity import AdaptivityPolynomialError
        from pySDC.implementations.problem_classes.RayleighBenard import CFLLimit
        from pySDC.implementations.convergence_controller_classes.step_size_limiter import StepSizeRounding

        desc = super().get_description(*args, **kwargs)

        desc['convergence_controllers'][AdaptivityPolynomialError] = {
            'e_tol': 1e-5,
            'abort_at_growing_residual': False,
            'interpolate_between_restarts': False,
            'dt_min': 1e-5,
            'dt_rel_min_slope': 2,
            'beta': 0.5,
        }
        desc['convergence_controllers'][StepSizeRounding] = {}
        desc['convergence_controllers'].pop(CFLLimit)
        desc['level_params']['restol'] = 5e-6
        desc['level_params']['e_tol'] = 5e-6
        desc['level_params']['dt'] = 5e-3
        desc['sweeper_params']['num_nodes'] = 4
        desc['sweeper_params']['QI'] = 'MIN-SR-S'
        desc['sweeper_params']['QE'] = 'PIC'
        desc['step_params']['maxiter'] = 16

        desc['problem_params']['Rayleigh'] = self.Ra
        desc['problem_params']['max_cached_factorizations'] = 4

        return desc

    def get_controller_params(self, *args, **kwargs):
        from pySDC.implementations.problem_classes.RayleighBenard import (
            LogAnalysisVariables,
        )

        controller_params = super().get_controller_params(*args, **kwargs)
        controller_params['hook_class'] += [LogAnalysisVariables]
        return controller_params

    def get_initial_condition(self, P, *args, restart_idx=0, **kwargs):
        if restart_idx > 0:
            return super().get_initial_condition(P, *args, restart_idx=restart_idx, **kwargs)
        else:
            u0 = P.u_exact(t=0, seed=P.comm.rank, noise_level=1e-3)
            for _ in range(self.relaxation_steps):
                u0 = P.solve_system(u0, 1e-1, u0)
            return u0, 0
