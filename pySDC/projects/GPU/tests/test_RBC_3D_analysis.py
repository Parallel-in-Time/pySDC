import pytest


def get_args(path):
    args = {}

    args['mode'] = 'run'
    args['dt'] = 0.1
    args['res'] = 4
    args['config'] = 'RBC3DG4R4SDC23Ra1e5'
    args['o'] = path
    args['useGPU'] = False
    args['restart_idx'] = 0
    args['procs'] = [1, 1, 1]
    args['logger_level'] = 15

    return args


def get_config(args):
    from pySDC.projects.GPU.configs.base_config import get_config

    config = get_config(args)
    config.Tend = 1
    config.converged = 0
    return config


def generate_simulation_file(path):
    from pySDC.projects.GPU.run_experiment import run_experiment

    args = get_args(path)
    config = get_config(args)

    run_experiment(args, config)
    return f'{path}/{config.get_file_name()}'


def generate_processed_file(path):
    from pySDC.projects.GPU.analysis_scripts.process_RBC3D_data import process_RBC3D_data

    args = get_args(path)
    config = get_config(args)
    process_RBC3D_data(path, args=args, config=config, plot=False)
    return process_RBC3D_data(path, args=args, config=config, plot=False)


@pytest.fixture
def tmp_sim_data(tmp_path):
    return generate_simulation_file(tmp_path)


@pytest.fixture
def tmp_processed_data(tmp_sim_data, tmp_path):
    return generate_processed_file(tmp_path)


def test_ic_interpolation(tmp_sim_data, tmp_path):
    from pySDC.projects.GPU.run_experiment import run_experiment

    args = get_args(tmp_path)

    ic_res = args['res'] * 1
    args['res'] *= 2
    config = get_config(args)

    config.ic_config = {'config': type(config), 'res': ic_res, 'dt': args['dt']}
    u = run_experiment(args, config)
    assert u.shape[-1] == args['res']


def test_processing(tmp_processed_data):
    import pickle

    with open(tmp_processed_data, 'rb') as file:
        data = pickle.load(file)

    for me in ['t', 'Nu', 'spectrum', 'z', 'k', 'profile_T', 'rms_profile_T']:
        assert me in data.keys()


def test_get_pySDC_data(tmp_processed_data, tmp_path):
    from pySDC.projects.GPU.analysis_scripts.plot_Nu import get_pySDC_data

    args = get_args(tmp_path)
    data = get_pySDC_data(res=args['res'], dt=args['dt'], config_name=args['config'], base_path=tmp_path)

    for me in ['t', 'Nu', 'spectrum', 'z', 'k', 'profile_T', 'rms_profile_T']:
        assert me in data.keys()


def test_Nu_interpolation():
    from pySDC.projects.GPU.analysis_scripts.plot_Nu import interpolate_NuV_to_reference_times
    import numpy as np

    t = sorted(np.random.rand(128))
    t_ref = np.linspace(0, max(t), 128)

    def _get_Nu(_t):
        return np.array(_t) ** 8

    ref_data = {'t': t_ref, 'Nu': {'V': _get_Nu(t_ref)}}
    data = {'t': t, 'Nu': {'V': _get_Nu(t)}}

    # interpolate to sufficient order
    tI, NuI = interpolate_NuV_to_reference_times(data, ref_data, order=8)
    assert np.allclose(NuI, ref_data['Nu']['V'])
    assert not np.allclose(data['Nu']['V'], ref_data['Nu']['V'])
    assert np.allclose(tI, ref_data['t'])

    # interpolate to insufficient order
    tI, NuI = interpolate_NuV_to_reference_times(data, ref_data, order=4)
    assert not np.allclose(NuI, ref_data['Nu']['V'])
    assert np.allclose(tI, ref_data['t'])
