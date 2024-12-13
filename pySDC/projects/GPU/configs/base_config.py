from pySDC.core.convergence_controller import ConvergenceController
import pickle
import numpy as np


def get_config(args):
    name = args['config']
    if name[:2] == 'GS':
        from pySDC.projects.GPU.configs.GS_configs import get_config as _get_config
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

    def __init__(self, args, comm_world=None):
        from mpi4py import MPI

        self.args = args
        self.comm_world = MPI.COMM_WORLD if comm_world is None else comm_world
        self.n_procs_list = args["procs"]
        if args['mode'] == 'run':
            self.comms = get_comms(n_procs_list=self.n_procs_list, useGPU=args['useGPU'], comm_world=self.comm_world)
        else:
            self.comms = [MPI.COMM_SELF, MPI.COMM_SELF, MPI.COMM_SELF]
        self.ranks = [me.rank for me in self.comms]

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
            description['convergence_controllers'][LogStats] = {}

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
        if restart_idx > 0:
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
        else:
            return P.u_exact(t=0), 0

    def get_LogToFile(self):
        return None


class LogStats(ConvergenceController):

    @staticmethod
    def get_stats_path(hook, counter_offset=-1, index=None):
        index = hook.counter + counter_offset if index is None else index
        return f'{hook.path}/{hook.file_name}_{hook.format_index(index)}-stats.pickle'

    def merge_all_stats(self, controller):
        hook = self.params.hook

        stats = {}
        for i in range(hook.counter):
            with open(self.get_stats_path(hook, index=i), 'rb') as file:
                _stats = pickle.load(file)
                stats = {**stats, **_stats}

        stats = {**stats, **controller.return_stats()}
        return stats

    def reset_stats(self, controller):
        for hook in controller.hooks:
            hook.reset_stats()
        self.logger.debug('Reset stats')

    def setup(self, controller, params, *args, **kwargs):
        params['control_order'] = 999
        if 'hook' not in params.keys():
            from pySDC.implementations.hooks.log_solution import LogToFileAfterXs

            params['hook'] = LogToFileAfterXs

        self.counter = params['hook'].counter

        return super().setup(controller, params, *args, **kwargs)

    def post_step_processing(self, controller, S, **kwargs):
        hook = self.params.hook

        for _hook in controller.hooks:
            _hook.post_step(S, 0)

        P = S.levels[0].prob
        skip_logging = False
        if hasattr(P, 'comm'):
            if P.comm.rank > 0:
                skip_logging = True
                return None

        while self.counter < hook.counter:
            path = self.get_stats_path(hook, index=self.counter)
            stats = controller.return_stats()
            if hook.logging_condition(S.levels[0]) and not skip_logging:
                with open(path, 'wb') as file:
                    pickle.dump(stats, file)
                    self.log(f'Stored stats in {path!r}', S)
                self.reset_stats(controller)
            self.counter = hook.counter

    def post_run_processing(self, controller, *args, **kwargs):
        stats = self.merge_all_stats(controller)

        def return_stats():
            return stats

        controller.return_stats = return_stats
