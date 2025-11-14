from pySDC.projects.GPU.configs.base_config import get_config
from pySDC.projects.GPU.run_experiment import parse_args
from pySDC.helpers.fieldsIO import FieldsIO
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpi4py import MPI
import numpy as np
import pickle
import os


def process_RBC3D_data(base_path='./data/RBC_time_averaged', plot=True, args=None, config=None):
    # prepare problem instance
    args = args if args else parse_args()
    comm = MPI.COMM_WORLD
    config = config if config else get_config(args)
    desc = config.get_description(**args)
    P = desc['problem_class'](
        **{
            **desc['problem_params'],
            'spectral_space': False,
            'comm': comm,
            'Dirichlet_recombination': False,
            'left_preconditioner': False,
        }
    )
    P.setUpFieldsIO()
    zInt = P.axes[-1].get_integration_weights()
    xp = P.xp

    # prepare paths
    os.makedirs(base_path, exist_ok=True)
    fname = config.get_file_name()
    fname_trim = fname[fname.index('RBC3D') : fname.index('.pySDC')]
    path = f'{base_path}/{fname_trim}.pickle'

    # open simulation data
    data = FieldsIO.fromFile(fname)

    # prepare arrays to store data in
    Nu = {
        'V': [],
        'b': [],
        't': [],
        'thermal': [],
        'kinetic': [],
    }
    t = []
    profiles = {key: [] for key in ['T', 'u', 'v', 'w']}
    rms_profiles = {key: [] for key in profiles.keys()}
    spectrum = []
    spectrum_all = []

    # try to load time averaged values
    u_mean_profile = P.u_exact()
    if os.path.isfile(path):
        with open(path, 'rb') as file:
            avg_data = pickle.load(file)
            if comm.rank == 0:
                print(f'Read data from file {path!r}')
        for key in profiles.keys():
            if f'profile_{key}' in avg_data.keys():
                u_mean_profile[P.index(key)] = avg_data[f'profile_{key}'][P.local_slice(False)[-1]]
    elif comm.rank == 0:
        print('No mean profiles available yet. Please rerun script after completion to get correct RMS profiles')

    # prepare progress bar
    indeces = range(args['restart_idx'], data.nFields)
    if P.comm.rank == 0:
        indeces = tqdm(indeces)

    # loop through all data points and compute stuff
    for i in indeces:
        _t, u = data.readField(i)

        # Nusselt numbers
        _Nu = P.compute_Nusselt_numbers(u)
        if any(me > 1e3 for me in _Nu.values()):
            continue

        for key in Nu.keys():
            Nu[key].append(_Nu[key])

        t.append(_t)

        # profiles
        _profiles = P.get_vertical_profiles(u, list(profiles.keys()))
        _rms_profiles = P.get_vertical_profiles((u - u_mean_profile) ** 2, list(profiles.keys()))
        for key in profiles.keys():
            profiles[key].append(_profiles[key])
            rms_profiles[key].append(_rms_profiles[key])

        # spectrum
        k, s = P.get_frequency_spectrum(u)
        s_mean = zInt @ P.axes[-1].transform(s[0], axes=(0,))
        spectrum.append(s_mean)
        spectrum_all.append(s)

    # make a plot of the results
    t = xp.array(t)
    z = P.axes[-1].get_1dgrid()

    if config.converged == 0:
        print('Warning: no convergence has been set for this configuration!')

    fig, axs = plt.subplots(1, 4, figsize=(18, 4))
    for key in Nu.keys():
        axs[0].plot(t, Nu[key], label=f'$Nu_{{{key}}}$')
        if config.converged > 0:
            axs[0].axvline(config.converged, color='black')
    axs[0].set_ylabel('$Nu$')
    axs[0].set_xlabel('$t$')
    axs[0].legend(frameon=False)

    # compute differences in Nusselt numbers
    avg_Nu = {}
    std_Nu = {}
    for key in Nu.keys():
        _Nu = [Nu[key][i] for i in range(len(Nu[key])) if t[i] > config.converged]
        avg_Nu[key] = xp.mean(_Nu)
        std_Nu[key] = xp.std(_Nu)

    rel_error = {
        key: abs(avg_Nu[key] - avg_Nu['V']) / avg_Nu['V']
        for key in [
            't',
            'b',
            'thermal',
            'kinetic',
        ]
    }
    if comm.rank == 0:
        print(
            f'With Ra={P.Rayleigh:.0e} got Nu={avg_Nu["V"]:.2f}+-{std_Nu["V"]:.2f} with errors: Top {rel_error["t"]:.2e}, bottom: {rel_error["b"]:.2e}, thermal: {rel_error["thermal"]:.2e}, kinetic: {rel_error["kinetic"]:.2e}'
        )

    # compute average profiles
    avg_profiles = {}
    for key, values in profiles.items():
        values_from_convergence = [values[i] for i in range(len(values)) if t[i] >= config.converged]

        avg_profiles[key] = xp.mean(values_from_convergence, axis=0)

    avg_rms_profiles = {}
    for key, values in rms_profiles.items():
        values_from_convergence = [values[i] for i in range(len(values)) if t[i] >= config.converged]
        avg_rms_profiles[key] = xp.sqrt(xp.mean(values_from_convergence, axis=0))

    # average T
    avg_T = avg_profiles['T']
    axs[1].axvline(0.5, color='black')
    axs[1].plot(avg_T, z)
    axs[1].set_xlabel('$T$')
    axs[1].set_ylabel('$z$')

    # rms profiles
    avg_T = avg_rms_profiles['T']
    max_idx = xp.argmax(avg_T)
    res_in_boundary_layer = max_idx if max_idx < len(z) / 2 else len(z) - max_idx
    boundary_layer = z[max_idx] if max_idx > len(z) / 2 else P.axes[-1].L - z[max_idx]
    if comm.rank == 0:
        print(
            f'Thermal boundary layer of thickness {boundary_layer:.2f} is resolved with {res_in_boundary_layer} points'
        )
    axs[2].axhline(z[max_idx], color='black')
    axs[2].plot(avg_T, z)
    axs[2].scatter(avg_T, z)
    axs[2].set_xlabel(r'$T_\text{rms}$')
    axs[2].set_ylabel('$z$')

    # spectrum
    _s = xp.array(spectrum)
    avg_spectrum = xp.mean(_s[t >= config.converged], axis=0)
    axs[3].loglog(k[avg_spectrum > 1e-15], avg_spectrum[avg_spectrum > 1e-15])
    axs[3].set_xlabel('$k$')
    axs[3].set_ylabel(r'$\|\hat{u}_x\|$')

    if P.comm.rank == 0:
        write_data = {
            't': t,
            'Nu': Nu,
            'avg_Nu': avg_Nu,
            'std_Nu': std_Nu,
            'z': P.axes[-1].get_1dgrid(),
            'k': k,
            'spectrum': spectrum_all,
            'avg_spectrum': avg_spectrum,
            'boundary_layer_thickness': boundary_layer,
            'res_in_boundary_layer': res_in_boundary_layer,
        }
        for key, values in avg_profiles.items():
            write_data[f'profile_{key}'] = values
        for key, values in avg_rms_profiles.items():
            write_data[f'rms_profile_{key}'] = values

        with open(path, 'wb') as file:
            pickle.dump(write_data, file)
            print(f'Wrote data to file {path!r}')

        if plot:
            fig.tight_layout()
            fig.savefig(f'{base_path}/{fname_trim}.pdf')
            plt.show()
    return path


def get_pySDC_data(res=-1, dt=-1, config_name='RBC3DG4', base_path='data/RBC_time_averaged'):
    path = f'{base_path}/{config_name}-res{res}-dt{dt:.0e}.pickle'
    with open(path, 'rb') as file:
        data = pickle.load(file)

    return data


if __name__ == '__main__':
    process_RBC3D_data()
