import pytest


@pytest.mark.mpi4py
def test_get_comms(launch=True):
    if launch:
        import subprocess
        import os

        # Set python path once
        my_env = os.environ.copy()
        my_env['PYTHONPATH'] = '../../..:.'
        my_env['COVERAGE_PROCESS_START'] = 'pyproject.toml'

        cmd = f"mpirun -np 24 python {__file__} --test=get_comms".split()

        p = subprocess.Popen(cmd, env=my_env, cwd=".")

        p.wait()
        assert p.returncode == 0, f'ERROR: did not get return code 0, got {p.returncode}'
    else:
        from pySDC.projects.GPU.configs.base_config import get_comms
        import numpy as np

        n_procs_list = [2, 3, 4]
        comms = get_comms(n_procs_list=n_procs_list)
        assert np.allclose([me.size for me in comms], n_procs_list)


def create_directories():
    import os

    dir_path = __file__.split('/')
    path = ''.join([me + '/' for me in dir_path[:-1]]) + 'data'
    os.makedirs(path, exist_ok=True)


@pytest.mark.order(1)
def test_run_experiment(restart_idx=0):
    from pySDC.projects.GPU.configs.base_config import Config
    from pySDC.projects.GPU.run_experiment import run_experiment, parse_args
    from pySDC.helpers.stats_helper import get_sorted
    import pickle
    import numpy as np

    create_directories()

    class VdPConfig(Config):
        sweeper_type = 'generic_implicit'
        Tend = 1

        def get_description(self, *args, **kwargs):
            from pySDC.implementations.problem_classes.Van_der_Pol_implicit import vanderpol

            desc = super().get_description(*args, **kwargs)
            desc['problem_class'] = vanderpol
            desc['problem_params'].pop('useGPU')
            desc['problem_params'].pop('comm')
            desc['sweeper_params']['num_nodes'] = 2
            desc['sweeper_params']['quad_type'] = 'RADAU-RIGHT'
            desc['sweeper_params']['QI'] = 'LU'
            desc['level_params']['dt'] = 0.1
            desc['step_params']['maxiter'] = 3
            return desc

        def get_LogToFile(self, ranks=None):
            from pySDC.implementations.hooks.log_solution import LogToFileAfterXs as LogToFile

            LogToFile.path = './data/'
            LogToFile.file_name = f'{self.get_path(ranks=ranks)}-solution'
            LogToFile.time_increment = 2e-1

            def logging_condition(L):
                sweep = L.sweep
                if hasattr(sweep, 'comm'):
                    if sweep.comm.rank == sweep.comm.size - 1:
                        return True
                    else:
                        return False
                else:
                    return True

            LogToFile.logging_condition = logging_condition
            return LogToFile

    args = {'procs': [1, 1, 1], 'useGPU': False, 'res': -1, 'logger_level': 15, 'restart_idx': restart_idx}
    config = VdPConfig(args)
    run_experiment(args, config)

    with open(f'data/{config.get_path()}-stats-whole-run.pickle', 'rb') as file:
        stats = pickle.load(file)

    k_Newton = get_sorted(stats, type='work_newton')
    assert len(k_Newton) == 10
    assert sum([me[1] for me in k_Newton]) == 91


@pytest.mark.order(2)
def test_restart():
    test_run_experiment(3)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=str)

    args = parser.parse_args()

    if args.test == 'get_comms':
        test_get_comms(False)
    elif args.test == 'run_experiment':
        test_run_experiment()
    elif args.test == 'restart':
        test_restart()
    else:
        raise NotImplementedError
