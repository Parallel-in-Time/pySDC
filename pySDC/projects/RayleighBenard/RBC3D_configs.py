from pySDC.projects.GPU.configs.base_config import Config


def get_config(args):
    name = args['config']
    if name == 'RBC3D':
        return RayleighBenard3DRegular(args)
    elif name in globals().keys():
        return globals()[name](args)
    else:
        raise NotImplementedError(f'There is no configuration called {name!r}!')


class RayleighBenard3DRegular(Config):
    sweeper_type = 'IMEX'
    Tend = 50
    gamma = 1
    res_ratio = 1
    dealiasing = 3.0 / 2.0

    def get_file_name(self):
        res = self.args['res']
        return f'{self.base_path}/data/{type(self).__name__}-res{res}.pySDC'

    def get_LogToFile(self, *args, **kwargs):
        if self.comms[1].rank > 0:
            return None
        import numpy as np
        from pySDC.implementations.hooks.log_solution import LogToFile

        LogToFile.filename = self.get_file_name()
        LogToFile.time_increment = 5e-1
        # LogToFile.allow_overwriting = True

        return LogToFile

    def get_controller_params(self, *args, **kwargs):
        from pySDC.implementations.hooks.log_step_size import LogStepSize

        controller_params = super().get_controller_params(*args, **kwargs)
        controller_params['hook_class'] += [LogStepSize]
        return controller_params

    def get_description(self, *args, MPIsweeper=False, res=-1, **kwargs):
        from pySDC.implementations.problem_classes.RayleighBenard3D import (
            RayleighBenard3D,
        )
        from pySDC.implementations.problem_classes.generic_spectral import (
            compute_residual_DAE,
            compute_residual_DAE_MPI,
        )
        from pySDC.implementations.convergence_controller_classes.step_size_limiter import StepSizeSlopeLimiter
        from pySDC.implementations.convergence_controller_classes.crash import StopAtNan

        desc = super().get_description(*args, MPIsweeper=MPIsweeper, **kwargs)

        if MPIsweeper:
            desc['sweeper_class'].compute_residual = compute_residual_DAE_MPI
        else:
            desc['sweeper_class'].compute_residual = compute_residual_DAE

        desc['level_params']['dt'] = 0.01
        desc['level_params']['restol'] = 1e-7

        desc['convergence_controllers'][StepSizeSlopeLimiter] = {'dt_rel_min_slope': 0.1}
        desc['convergence_controllers'][StopAtNan] = {}

        desc['sweeper_params']['quad_type'] = 'RADAU-RIGHT'
        desc['sweeper_params']['num_nodes'] = 2
        desc['sweeper_params']['QI'] = 'MIN-SR-S'
        desc['sweeper_params']['QE'] = 'PIC'

        res = 64 if res == -1 else res
        desc['problem_params']['Rayleigh'] = 1e8
        desc['problem_params']['nx'] = self.res_ratio * res
        desc['problem_params']['ny'] = self.res_ratio * res
        desc['problem_params']['nz'] = res
        desc['problem_params']['Lx'] = self.gamma
        desc['problem_params']['Ly'] = self.gamma
        desc['problem_params']['Lz'] = 1
        desc['problem_params']['heterogeneous'] = True
        desc['problem_params']['dealiasing'] = self.dealiasing

        desc['step_params']['maxiter'] = 3

        desc['problem_class'] = RayleighBenard3D

        return desc

    def get_initial_condition(self, P, *args, restart_idx=0, **kwargs):

        if restart_idx == 0:
            u0 = P.u_exact(t=0, seed=P.comm.rank, noise_level=1e-3)
            u0_with_pressure = P.solve_system(u0, 1e-9, u0)
            P.cached_factorizations.pop(1e-9)
            return u0_with_pressure, 0
        else:
            from pySDC.helpers.fieldsIO import FieldsIO

            P.setUpFieldsIO()
            outfile = FieldsIO.fromFile(self.get_file_name())

            t0, solution = outfile.readField(restart_idx)

            u0 = P.u_init

            if P.spectral_space:
                u0[...] = P.transform(solution)
            else:
                u0[...] = solution

            return u0, t0

    def prepare_caches(self, prob):
        """
        Cache the fft objects, which are expensive to create on GPU because graphs have to be initialized.
        """
        prob.eval_f(prob.u_init)


class RBC3Dverification(RayleighBenard3DRegular):
    converged = 0
    dt = 1e-2
    ic_config = {
        'config': None,
        'res': -1,
        'dt': -1,
    }
    res = None
    Ra = None
    Tend = 100
    res_ratio = 4
    gamma = 4

    def get_file_name(self):
        res = self.args['res']
        dt = self.args['dt']
        return f'{self.base_path}/data/{type(self).__name__}-res{res}-dt{dt:.0e}.pySDC'

    def get_description(self, *args, res=-1, dt=-1, **kwargs):
        desc = super().get_description(*args, **kwargs)
        desc['level_params']['nsweeps'] = 4
        desc['level_params']['restol'] = -1
        desc['step_params']['maxiter'] = 1
        desc['sweeper_params']['QI'] = 'MIN-SR-S'
        desc['sweeper_params']['skip_residual_computation'] = ('IT_CHECK', 'IT_DOWN', 'IT_UP', 'IT_FINE', 'IT_COARSE')
        desc['sweeper_params']['num_nodes'] = 4
        Ra = int(type(self).__name__[-3]) * 10 ** int(type(self).__name__[-1])
        desc['problem_params']['Rayleigh'] = Ra
        desc['problem_params']['Prandtl'] = 0.7

        _res = self.res if res == -1 else res
        desc['problem_params']['nx'] = _res * self.res_ratio
        desc['problem_params']['ny'] = _res * self.res_ratio
        desc['problem_params']['nz'] = _res

        _dt = self.dt if dt == -1 else dt
        desc['level_params']['dt'] = _dt

        desc['problem_params']['Lx'] = float(self.gamma)
        desc['problem_params']['Ly'] = float(self.gamma)
        desc['problem_params']['Lz'] = 1.0
        return desc

    def get_initial_condition(self, P, *args, restart_idx=0, **kwargs):
        if self.ic_config['config'] is None or restart_idx != 0:
            return super().get_initial_condition(P, *args, restart_idx=restart_idx, **kwargs)

        # read initial conditions
        from pySDC.helpers.fieldsIO import FieldsIO

        ic_config = self.ic_config['config'](
            args={**self.args, 'res': self.ic_config['res'], 'dt': self.ic_config['dt']}
        )
        desc = ic_config.get_description(res=self.ic_config['res'], dt=self.ic_config['dt'])
        ic_nx = desc['problem_params']['nx']
        ic_ny = desc['problem_params']['ny']
        ic_nz = desc['problem_params']['nz']

        _P = type(P)(nx=ic_nx, ny=ic_ny, nz=ic_nz, comm=P.comm, useGPU=P.useGPU)
        _P.setUpFieldsIO()
        filename = ic_config.get_file_name()
        ic_file = FieldsIO.fromFile(filename)
        t0, ics = ic_file.readField(-1)
        P.logger.info(f'Loaded initial conditions from {filename!r} at t={t0}.')

        # interpolate the initial conditions using padded transforms
        padding = (P.nx / ic_nx, P.ny / ic_ny, P.nz / ic_nz)
        P.logger.info(f'Interpolating initial conditions from {ic_nx}x{ic_ny}x{ic_nz} to {P.nx}x{P.ny}x{P.nz}')

        ics = _P.xp.array(ics)
        _ics_hat = _P.transform(ics)
        ics_interpolated = _P.itransform(_ics_hat, padding=padding)

        self.get_LogToFile()

        P.setUpFieldsIO()
        if P.spectral_space:
            u0_hat = P.u_init_forward
            u0_hat[...] = P.transform(ics_interpolated)
            return u0_hat, 0
        else:
            return ics_interpolated, 0


class RBC3DM2K3(RBC3Dverification):

    def get_description(self, *args, **kwargs):
        desc = super().get_description(*args, **kwargs)
        desc['level_params']['nsweeps'] = 3
        desc['sweeper_params']['num_nodes'] = 2
        return desc


class RBC3DM3K4(RBC3Dverification):

    def get_description(self, *args, **kwargs):
        desc = super().get_description(*args, **kwargs)
        desc['level_params']['nsweeps'] = 4
        desc['sweeper_params']['num_nodes'] = 3
        return desc


class RBC3DM4K4(RBC3Dverification):

    def get_description(self, *args, **kwargs):
        desc = super().get_description(*args, **kwargs)
        desc['level_params']['nsweeps'] = 4
        desc['sweeper_params']['num_nodes'] = 4
        return desc


class RBC3DverificationRK(RBC3Dverification):

    def get_description(self, *args, res=-1, dt=-1, **kwargs):
        from pySDC.implementations.sweeper_classes.Runge_Kutta import ARK3

        desc = super().get_description(*args, res=res, dt=dt, **kwargs)
        desc['level_params']['nsweeps'] = 1
        desc['level_params']['restol'] = -1
        desc['step_params']['maxiter'] = 1
        desc['sweeper_params']['skip_residual_computation'] = ('IT_CHECK', 'IT_DOWN', 'IT_UP', 'IT_FINE', 'IT_COARSE')
        desc['sweeper_params'].pop('QI')
        desc['sweeper_params'].pop('num_nodes')
        desc['sweeper_class'] = ARK3
        return desc


class RBC3DverificationEuler(RBC3DverificationRK):

    def get_description(self, *args, res=-1, dt=-1, **kwargs):
        from pySDC.implementations.sweeper_classes.Runge_Kutta import IMEXEulerStifflyAccurate

        desc = super().get_description(*args, res=res, dt=dt, **kwargs)
        desc['sweeper_class'] = IMEXEulerStifflyAccurate
        return desc


class RBC3DG4R4Ra1e5(RBC3Dverification):
    Tend = 200
    dt = 6e-2
    res = 32
    converged = 50


class RBC3DG4R4SDC23Ra1e5(RBC3DM2K3):
    Tend = 200
    dt = 6e-2
    res = 32
    converged = 50


class RBC3DG4R4SDC34Ra1e5(RBC3DM3K4):
    Tend = 200
    dt = 6e-2
    res = 32
    converged = 50


class RBC3DG4R4SDC44Ra1e5(RBC3DM4K4):
    Tend = 200
    dt = 6e-2
    res = 32
    converged = 50


class RBC3DG4R4RKRa1e5(RBC3DverificationRK):
    Tend = 200
    dt = 8e-2
    res = 32
    converged = 50


class RBC3DG4R4EulerRa1e5(RBC3DverificationEuler):
    Tend = 200
    dt = 8e-2
    res = 32
    converged = 50
