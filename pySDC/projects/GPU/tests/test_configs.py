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


def test_run(tmpdir):
    from pySDC.projects.GPU.configs.base_config import get_config
    from pySDC.projects.GPU.run_experiment import run_experiment
    from pySDC.helpers.stats_helper import get_sorted
    from pySDC.helpers.fieldsIO import FieldsIO
    import pickle
    import numpy as np

    args = {
        'config': 'RBC',
        'procs': [1, 1, 1],
        'res': 16,
        'mode': 'run',
        'useGPU': False,
        'o': tmpdir,
        'logger_level': 15,
        'restart_idx': 0,
    }
    config = get_config(args)
    type(config).base_path = args['o']

    def get_LogToFile(self, *args, **kwargs):
        if self.comms[1].rank > 0:
            return None
        from pySDC.implementations.hooks.log_solution import LogToFile

        LogToFile.filename = self.get_file_name()
        LogToFile.time_increment = 0
        LogToFile.allow_overwriting = True

        return LogToFile

    type(config).get_LogToFile = get_LogToFile
    stats_path = f'{config.base_path}/data/{config.get_path()}-stats-whole-run.pickle'
    file_path = config.get_file_name()

    # first run for a short time
    dt = config.get_description()['level_params']['dt']
    config.Tend = 2 * dt
    run_experiment(args, config)

    # check data
    data = FieldsIO.fromFile(file_path)
    with open(stats_path, 'rb') as file:
        stats = pickle.load(file)

    dts = get_sorted(stats, type='dt')
    assert len(dts) == len(data.times) - 1
    assert np.allclose(data.times, 0.1 * np.arange(3)), 'Did not record solutions at expected times'

    # restart run
    args['restart_idx'] = -1
    config.Tend = 4 * dt
    run_experiment(args, config)

    # check data
    data = FieldsIO.fromFile(file_path)
    with open(stats_path, 'rb') as file:
        stats = pickle.load(file)
    dts = get_sorted(stats, type='dt')

    assert len(dts) == len(data.times) - 1
    assert np.allclose(data.times, 0.1 * np.arange(5)), 'Did not record solutions at expected times after restart'


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=str)

    args = parser.parse_args()

    if args.test == 'get_comms':
        test_get_comms(False)
    else:
        raise NotImplementedError
