from pySDC.core.convergence_controller import ConvergenceController
import pickle
import numpy as np


def get_config(args):
    name = args['config']
    if name[:2] == 'GS':
        from pySDC.projects.GPU.configs.GS_configs import get_config as _get_config
    elif name[:5] == 'RBC3D':
        from pySDC.projects.GPU.configs.RBC3D_configs import get_config as _get_config
    elif name[:3] == 'RBC':
        from pySDC.projects.GPU.configs.RBC_configs import get_config as _get_config
    else:
        raise NotImplementedError(f'There is no configuration called {name!r}!')

    return _get_config(args)


def get_comms(n_procs_list, comm_world=None, _comm=None, _tot_rank=0, _rank=None, useGPU=False):
    from mpi4py import MPI

    comm_world = MPI.COMM_WORLD if comm_world is None else comm_world
    _comm = comm_world if _comm is None else _comm
    _rank = comm_world.rank if _rank is None else _rank

    if len(n_procs_list) > 0:
        color = _tot_rank + _rank // n_procs_list[0]
        new_comm = comm_world.Split(color)

        assert new_comm.size == n_procs_list[0]

        if useGPU:
            import cupy_backends

            try:
                import cupy
                from pySDC.helpers.NCCL_communicator import NCCLComm

                new_comm = NCCLComm(new_comm)
            except (
                ImportError,
                cupy_backends.cuda.api.runtime.CUDARuntimeError,
                cupy_backends.cuda.libs.nccl.NcclError,
            ):
                print('Warning: Communicator is MPI instead of NCCL in spite of using GPUs!')

        return [new_comm] + get_comms(
            n_procs_list[1:],
            comm_world,
            _comm=new_comm,
            _tot_rank=_tot_rank + _comm.size * new_comm.rank,
            _rank=_comm.rank // new_comm.size,
            useGPU=useGPU,
        )
    else:
        return []


class Config(object):
    sweeper_type = None
    Tend = None
    base_path = './'
    logging_time_increment = 0.5

    def __init__(self, args, comm_world=None):
        from mpi4py import MPI

        self.args = args
        self.comm_world = MPI.COMM_WORLD if comm_world is None else comm_world
        self.n_procs_list = args["procs"]
        if args['mode'] == 'run':
            self.comms = get_comms(
                n_procs_list=self.n_procs_list[::-1], useGPU=args['useGPU'], comm_world=self.comm_world
            )[::-1]
        else:
            self.comms = [MPI.COMM_SELF, MPI.COMM_SELF, MPI.COMM_SELF]
        self.ranks = [me.rank for me in self.comms]

    def get_file_name(self):
        res = self.args['res']
        return f'{self.base_path}/data/{type(self).__name__}-res{res}.pySDC'

    def get_LogToFile(self, *args, **kwargs):
        if self.comms[1].rank > 0:
            return None
        from pySDC.implementations.hooks.log_solution import LogToFile

        LogToFile.filename = self.get_file_name()
        LogToFile.time_increment = self.logging_time_increment
        LogToFile.allow_overwriting = True

        return LogToFile

    def get_description(self, *args, MPIsweeper=False, useGPU=False, **kwargs):
        description = {}
        description['problem_class'] = None
        description['problem_params'] = {'useGPU': useGPU, 'comm': self.comms[2]}
        description['sweeper_class'] = self.get_sweeper(useMPI=MPIsweeper)
        description['sweeper_params'] = {'initial_guess': 'copy'}
        description['level_params'] = {}
        description['step_params'] = {}
        description['convergence_controllers'] = {}

        if self.get_LogToFile():
            path = self.get_file_name()[:-6]
            description['convergence_controllers'][LogStats] = {'path': path}

        if MPIsweeper:
            description['sweeper_params']['comm'] = self.comms[1]
        return description

    def get_controller_params(self, *args, logger_level=15, **kwargs):
        from pySDC.implementations.hooks.log_work import LogWork
        from pySDC.implementations.hooks.log_step_size import LogStepSize
        from pySDC.implementations.hooks.log_restarts import LogRestarts

        controller_params = {}
        controller_params['logger_level'] = logger_level if self.comm_world.rank == 0 else 40
        controller_params['hook_class'] = [LogWork, LogStepSize, LogRestarts]
        logToFile = self.get_LogToFile()
        if logToFile:
            controller_params['hook_class'] += [logToFile]
        controller_params['mssdc_jac'] = False
        return controller_params

    def get_sweeper(self, useMPI):
        if useMPI and self.sweeper_type == 'IMEX':
            from pySDC.implementations.sweeper_classes.imex_1st_order_MPI import imex_1st_order_MPI as sweeper
        elif not useMPI and self.sweeper_type == 'IMEX':
            from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order as sweeper
        elif useMPI and self.sweeper_type == 'generic_implicit':
            from pySDC.implementations.sweeper_classes.generic_implicit_MPI import generic_implicit_MPI as sweeper
        elif not useMPI and self.sweeper_type == 'generic_implicit':
            from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit as sweeper
        else:
            raise NotImplementedError(f'Don\'t know the sweeper for {self.sweeper_type=}')

        return sweeper

    def prepare_caches(self, prob):
        pass

    def get_path(self, *args, ranks=None, **kwargs):
        ranks = self.ranks if ranks is None else ranks
        return f'{type(self).__name__}{self.args_to_str()}-{ranks[0]}-{ranks[2]}'

    def args_to_str(self, args=None):
        args = self.args if args is None else args
        name = ''

        name = f'{name}-res_{args["res"]}'
        name = f'{name}-useGPU_{args["useGPU"]}'
        name = f'{name}-procs_{args["procs"][0]}_{args["procs"][1]}_{args["procs"][2]}'
        return name

    def plot(self, P, idx, num_procs_list):
        raise NotImplementedError

    def get_initial_condition(self, P, *args, restart_idx=0, **kwargs):
        if restart_idx == 0:
            return P.u_exact(t=0), 0
        else:

            from pySDC.helpers.fieldsIO import FieldsIO

            P.setUpFieldsIO()
            outfile = FieldsIO.fromFile(self.get_file_name())

            t0, solution = outfile.readField(restart_idx)
            solution = solution[: P.spectral.ncomponents, ...]

            u0 = P.u_init

            if P.spectral_space:
                u0[...] = P.transform(solution)
            else:
                u0[...] = solution

            return u0, t0

            LogToFile = self.get_LogToFile()
            file = LogToFile.load(restart_idx)
            LogToFile.counter = restart_idx
            u0 = P.u_init
            if hasattr(P, 'spectral_space'):
                if P.spectral_space:
                    u0[...] = P.transform(file['u'])
                else:
                    u0[...] = file['u']
            else:
                u0[...] = file['u']
            return u0, file['t']


class LogStats(ConvergenceController):

    def get_stats_path(self, index=0):
        return f'{self.params.path}_{index:06d}-stats.pickle'

    def merge_all_stats(self, controller):
        hook = self.params.hook

        stats = {}
        for i in range(hook.counter - 1):
            try:
                with open(self.get_stats_path(index=i), 'rb') as file:
                    _stats = pickle.load(file)
                    stats = {**stats, **_stats}
            except (FileNotFoundError, EOFError):
                print(f'Warning: No stats found at path {self.get_stats_path(index=i)}')

        stats = {**stats, **controller.return_stats()}
        return stats

    def reset_stats(self, controller):
        for hook in controller.hooks:
            hook.reset_stats()
        self.logger.debug('Reset stats')

    def setup(self, controller, params, *args, **kwargs):
        params['control_order'] = 999
        if 'hook' not in params.keys():
            from pySDC.implementations.hooks.log_solution import LogToFile

            params['hook'] = LogToFile

        self.counter = params['hook'].counter
        return super().setup(controller, params, *args, **kwargs)

    def post_step_processing(self, controller, S, **kwargs):
        hook = self.params.hook

        P = S.levels[0].prob

        while self.counter < hook.counter:
            path = self.get_stats_path(index=hook.counter - 2)
            stats = controller.return_stats()
            store = True
            if hasattr(S.levels[0].sweep, 'comm') and S.levels[0].sweep.comm.rank > 0:
                store = False
            elif P.comm.rank > 0:
                store = False
            if store:
                with open(path, 'wb') as file:
                    pickle.dump(stats, file)
                    self.log(f'Stored stats in {path!r}', S)
                # print(stats)
                self.reset_stats(controller)
            self.counter = hook.counter

    def post_run_processing(self, controller, S, **kwargs):
        self.post_step_processing(controller, S, **kwargs)

        stats = self.merge_all_stats(controller)

        def return_stats():
            return stats

        controller.return_stats = return_stats
