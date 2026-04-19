"""
This script plots macroscopic verification of simulation data via comparison with data from https://doi.org/10.5281/zenodo.14205874
In order to run this script, you need to download the dataset and copy it to `pySDC/projects/RayleighBenard/data/Nek5000`.
And, you need to have generated the pySDC simulation data, of course.
"""

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

ints = {'1e5': 1e5, '1e6': 1e6, '1e7': 1e7}


def get_Nek5000_Data(Ra, base_path='data/Nek5000'):  # pragma: no cover
    assert type(Ra) == str

    # append relative path to base path
    path = __file__
    base_path = f'{path[::-1][path[::-1].index('/'):][::-1]}../{base_path}'
    Pr = 0.7

    if Ra == '1e5':
        dir_name = '1_1e5'
        start_time = 3500
        nelZ = 64
        nPoly = 5
    elif Ra == '1e6':
        dir_name = '2_1e6'
        start_time = 3500
        nelZ = 64
        nPoly = 7
    elif Ra == '1e7':
        dir_name = '3_1e7'
        start_time = 3100
        nelZ = 64
        nPoly = 9
    elif Ra == '1e8':
        dir_name = '4_1e8'
        start_time = 3000
        nelZ = 96
        nPoly = 7
    elif Ra == '1e9':
        dir_name = '5_1e9'
        start_time = 4000
        nelZ = 96
        nPoly = 9
    elif Ra == '1e10':
        dir_name = '6_1e10'
        start_time = 1700
        nelZ = 200
        nPoly = 7
    elif Ra == '1e11':
        dir_name = '7_1e11'
        start_time = 260
        nelZ = 256
        nPoly = 7
    else:
        raise

    path = f'{base_path}/{dir_name}'
    data = {}

    visc = np.sqrt(Pr / ints[Ra])

    # get averaged data
    avg = np.load(f'{path}/average.npy')
    avg_Nu = np.mean(avg[avg[:, 0] > start_time, 3])
    data['Nu'] = avg_Nu
    data['std_Nu'] = np.std(avg[avg[:, 0] > start_time, 3])

    Re = np.sqrt(avg[:, 1]) / visc
    data['Re'] = np.mean(Re[avg[:, 0] > start_time])

    # get profile data
    profiles = np.load(f'{path}/profile.npy')
    nzPts = nelZ * nPoly + 1
    nSnap = int(profiles.shape[0] / nzPts)
    tVal = profiles[:, 0].reshape((nSnap, nzPts))[:, 0]
    tInterval = tVal[-1] - tVal[0]

    data['z'] = profiles[:nzPts, 1]
    data['profile_T'] = profiles[:, 3].reshape((nSnap, nzPts))

    tRMS = profiles[:, 4].reshape((nSnap, nzPts))
    data['rms_profile_T'] = np.sqrt(integrate.simpson(tRMS**2, tVal, axis=0) / tInterval)

    return data


def get_pySDC_data(Ra):
    from pySDC.projects.RayleighBenard.analysis_scripts.process_RBC3D_data import get_pySDC_data as _get_data

    dts = {'1e5': 0.06, '1e6': 0.01, '1e7': 0.005}
    res = {'1e5': 32, '1e6': 64, '1e7': 128}
    return _get_data(config_name=f'RBC3DG4R4SDC23Ra{Ra}', dt=dts[Ra], res=res[Ra])


def plot_Nu_scaling(ax):  # pragma: no cover

    # reference values
    for Ra in ['1e5', '1e6', '1e7']:
        dat = get_Nek5000_Data(Ra)
        ax.errorbar(ints[Ra], dat['Nu'], yerr=dat['std_Nu'], fmt='o', color='black')

    # pySDC values
    for Ra in ['1e5', '1e6', '1e7']:
        dat = get_pySDC_data(Ra)
        ax.errorbar(ints[Ra], dat['avg_Nu']['V'], yerr=dat['std_Nu']['V'], fmt='.', color='tab:blue')

    ax.errorbar(None, None, fmt='o', color='black', label='Nek5000')
    ax.errorbar(None, None, fmt='.', color='tab:blue', label='pySDC')
    ax.legend(frameon=False, loc='lower right')

    ax.set_xscale('log')
    ax.set_xlabel('$Ra$')
    ax.set_ylabel('$Nu$')


def plot_T_profile(ax):  # pragma: no cover
    colors = {'1e5': 'tab:blue', '1e6': 'tab:orange', '1e7': 'tab:green'}
    markevery = {'1e7': 3}

    # reference values
    for Ra in ['1e5', '1e6', '1e7']:
        dat = get_Nek5000_Data(Ra)
        ax.plot(dat['profile_T'].mean(axis=0), dat['z'], color=colors[Ra], label=f'Nek5000 Ra={Ra}')

    # pySDC values
    for Ra in ['1e5', '1e6', '1e7']:
        dat = get_pySDC_data(Ra)
        s = slice(None, None, markevery.get(Ra, 1))
        ax.scatter(dat['profile_T'][s], dat['z'][s], color=colors[Ra], label=f'pySDC Ra={Ra}')

    ax.set_ylabel('$z$')
    ax.set_xlabel('$T$')
    ax.set_xlim((0.47, 1.03))
    ax.set_ylim((-0.01, 0.33))
    ax.legend(frameon=False)


def plot_T_rms_profile(ax):  # pragma: no cover
    colors = {'1e5': 'tab:blue', '1e6': 'tab:orange', '1e7': 'tab:green'}

    # reference values
    for Ra in ['1e5', '1e6', '1e7']:
        dat = get_Nek5000_Data(Ra)
        ax.plot(dat['rms_profile_T'], dat['z'], color=colors[Ra], label=f'Nek5000 Ra={Ra}')

    # pySDC values
    for Ra in ['1e5', '1e6', '1e7']:
        dat = get_pySDC_data(Ra)
        ax.scatter(dat['rms_profile_T'], dat['z'], color=colors[Ra], label=f'pySDC Ra={Ra}')

    ax.set_ylabel('$z$')
    ax.set_xlabel('$T$')
    ax.legend()


def plot_verification():  # pragma: no cover
    from pySDC.projects.RayleighBenard.analysis_scripts.plotting_utils import savefig, figsize

    fig, axs = plt.subplots(1, 2, figsize=figsize(scale=1, ratio=0.5))
    plot_Nu_scaling(axs[0])
    plot_T_profile(axs[1])
    fig.tight_layout()
    fig.savefig('plots/verification.pdf')


if __name__ == '__main__':
    plot_verification()
    plt.show()
