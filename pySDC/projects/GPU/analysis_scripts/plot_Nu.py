import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from pySDC.helpers.plot_helper import figsize_by_journal, setup_mpl

setup_mpl()


def get_pySDC_data(res=-1, dt=-1, config_name='RBC3DG4', base_path='data/RBC_time_averaged'):
    path = f'{base_path}/{config_name}-res{res}-dt{dt:.0e}.pickle'
    with open(path, 'rb') as file:
        data = pickle.load(file)

    return data


def interpolate_NuV_to_reference_times(data, reference_data, order=12):
    from qmat.lagrange import getSparseInterpolationMatrix

    t_in = np.array(data['t'])
    t_out = np.array([me for me in reference_data['t'] if me <= max(t_in)])

    interpolation_matrix = getSparseInterpolationMatrix(t_in, t_out, order=order)
    return interpolation_matrix @ t_in, interpolation_matrix @ data['Nu']['V']


def plot_Nu(res, dts, config_name, ref, ax, title):  # pragma: no cover
    ax.plot(ref['t'], ref['Nu']['V'], color='black', ls='--')
    ax.set_title(title)
    Nu_ref = np.array(ref['Nu']['V'])

    for dt in dts:
        data = get_pySDC_data(res=res, dt=dt, config_name=config_name)
        t_i, Nu_i = interpolate_NuV_to_reference_times(data, ref)
        ax.plot(t_i, Nu_i, label=rf'$\Delta t={{{dt}}}$')

        error = np.abs(Nu_ref[: Nu_i.shape[0]] - Nu_i) / np.abs(Nu_ref[: Nu_i.shape[0]])

        # compute mean Nu
        mask = np.logical_and(t_i >= 100, t_i <= 200)
        Nu_mean = np.mean(Nu_i[mask])
        Nu_std = np.std(Nu_i[mask])

        last_line = ax.get_lines()[-1]
        if any(error > 1e-2):
            deviates = min(t_i[error > 1e-2])
            ax.axvline(deviates, color=last_line.get_color(), ls=':')
            print(f'{title} dt={dt} Nu={Nu_mean:.3f}+={Nu_std:.3f}, deviates more than 1% from t={deviates:.2f}')
        else:
            print(f'{title} dt={dt} Nu={Nu_mean:.3f}+={Nu_std:.3f}')
        ax.legend(frameon=True, loc='upper left')


def plot_Nu_over_time_Ra1e5():  # pragma: no cover
    Nu_fig, Nu_axs = plt.subplots(4, 1, sharex=True, sharey=True, figsize=figsize_by_journal('Nature_CS', 1, 1.4))

    res = 32

    ref_data = get_pySDC_data(res=res, dt=0.01, config_name='RBC3DG4R4Ra1e5')

    plot_Nu(32, [0.06, 0.04, 0.02], 'RBC3DG4R4SDC34Ra1e5', ref_data, Nu_axs[0], 'SDC34')
    plot_Nu(32, [0.06, 0.05, 0.02, 0.01], 'RBC3DG4R4SDC23Ra1e5', ref_data, Nu_axs[1], 'SDC23')
    plot_Nu(32, [0.05, 0.04, 0.02, 0.01, 0.005], 'RBC3DG4R4RKRa1e5', ref_data, Nu_axs[2], 'RK443')
    plot_Nu(32, [0.02, 0.01, 0.005], 'RBC3DG4R4EulerRa1e5', ref_data, Nu_axs[3], 'RK111')

    Nu_axs[-1].set_xlabel('$t$')
    Nu_axs[-1].set_ylabel('$Nu$')

    Nu_fig.tight_layout()
    Nu_fig.savefig('./plots/Nu_over_time_Ra1e5.pdf', bbox_inches='tight')


if __name__ == '__main__':

    plot_Nu_over_time_Ra1e5()

    plt.show()
