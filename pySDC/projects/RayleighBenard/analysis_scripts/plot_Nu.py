import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from pySDC.projects.RayleighBenard.analysis_scripts.process_RBC3D_data import get_pySDC_data
from pySDC.projects.RayleighBenard.analysis_scripts.plotting_utils import figsize, savefig


def interpolate_NuV_to_reference_times(data, reference_data, order=12):
    from qmat.lagrange import getSparseInterpolationMatrix

    t_in = np.array(data['t'])
    t_out = np.array([me for me in reference_data['t'] if me <= max(t_in)])

    order = min([order, len(t_in), len(t_out)])
    interpolation_matrix = getSparseInterpolationMatrix(t_in, t_out, order=order)
    return interpolation_matrix @ t_in, interpolation_matrix @ data['Nu']['V']


def plot_Nu(res, dts, config_name, ref, ax, title, converged_from=0, **plotting_args):  # pragma: no cover
    ax.plot(ref['t'], ref['Nu']['V'], color='black', ls='--')
    ax.set_title(title)
    Nu_ref = np.array(ref['Nu']['V'])

    for dt in dts:
        data = get_pySDC_data(res=res, dt=dt, config_name=config_name)
        t_i, Nu_i = interpolate_NuV_to_reference_times(data, ref)

        ax.plot(data['t'], data['Nu']['V'], **{'label': rf'$\Delta t={{{dt}}}$', **plotting_args})

        error = np.abs(Nu_ref[: Nu_i.shape[0]] - Nu_i) / np.abs(Nu_ref[: Nu_i.shape[0]])

        # compute mean Nu
        mask = np.logical_and(t_i >= converged_from, t_i <= np.inf)
        Nu_mean = np.mean(Nu_i[mask])
        Nu_std = np.std(Nu_i[mask])

        last_line = ax.get_lines()[-1]
        if any(error > 1e-2):
            deviates = min(t_i[error > 1e-2])
            ax.axvline(deviates, color=last_line.get_color(), ls=':')
            print(f'{title} dt={dt:.4f} Nu={Nu_mean:.3f}+={Nu_std:.3f}, deviates more than 1% from t={deviates:.2f}')
        else:
            print(f'{title} dt={dt:.4f} Nu={Nu_mean:.3f}+={Nu_std:.3f}')
        ax.legend(frameon=True, loc='upper left')


def plot_Nu_over_time_Ra1e5():  # pragma: no cover
    Nu_fig, Nu_axs = plt.subplots(4, 1, sharex=True, sharey=True, figsize=figsize(scale=1, ratio=1.4))

    res = 32
    converged_from = 100

    ref_data = get_pySDC_data(res=res, dt=0.01, config_name='RBC3DG4R4SDC44Ra1e5')

    plot_Nu(res, [0.06, 0.04, 0.02], 'RBC3DG4R4SDC44Ra1e5', ref_data, Nu_axs[0], 'SDC44', converged_from)
    plot_Nu(res, [0.06, 0.05, 0.02, 0.01], 'RBC3DG4R4SDC23Ra1e5', ref_data, Nu_axs[1], 'SDC23', converged_from)
    plot_Nu(res, [0.05, 0.04, 0.02, 0.01, 0.005], 'RBC3DG4R4RKRa1e5', ref_data, Nu_axs[2], 'RK443', converged_from)
    plot_Nu(res, [0.02, 0.01, 0.005], 'RBC3DG4R4EulerRa1e5', ref_data, Nu_axs[3], 'RK111', converged_from)

    Nu_axs[-1].set_xlabel('$t$')
    Nu_axs[-1].set_ylabel('$Nu$')

    Nu_fig.tight_layout()
    Nu_fig.savefig('./plots/Nu_over_time_Ra1e5.pdf', bbox_inches='tight')


def plot_Nu_over_time_Ra1e6():  # pragma: no cover
    Nu_fig, Nu_axs = plt.subplots(4, 1, sharex=True, sharey=True, figsize=figsize(scale=1, ratio=1.4))

    res = 64
    converged_from = 25

    ref_data = get_pySDC_data(res=res, dt=0.002, config_name='RBC3DG4R4SDC34Ra1e6')

    plot_Nu(res, [0.02, 0.01], 'RBC3DG4R4SDC44Ra1e6', ref_data, Nu_axs[0], 'SDC44', converged_from)
    plot_Nu(res, [0.01, 0.005, 0.002], 'RBC3DG4R4SDC23Ra1e6', ref_data, Nu_axs[1], 'SDC23', converged_from)
    plot_Nu(res, [0.01, 0.005, 0.002], 'RBC3DG4R4RKRa1e6', ref_data, Nu_axs[2], 'RK443', converged_from)
    plot_Nu(res, [0.005, 0.002], 'RBC3DG4R4EulerRa1e6', ref_data, Nu_axs[3], 'RK111', converged_from)

    Nu_axs[-1].set_xlabel('$t$')
    Nu_axs[-1].set_ylabel('$Nu$')

    Nu_fig.tight_layout()
    Nu_fig.savefig('./plots/Nu_over_time_Ra1e6.pdf', bbox_inches='tight')


def plot_Nusselt_Ra1e5_same_dt():  # pragma: no cover
    fig, ax = plt.subplots(figsize=figsize(scale=0.9, ratio=0.45))

    res = 32
    converged_from = 100

    ref_data = get_pySDC_data(res=res, dt=0.01, config_name='RBC3DG4R4SDC44Ra1e5')

    plot_Nu(res, [0.02], 'RBC3DG4R4SDC44Ra1e5', ref_data, ax, 'SDC44', converged_from)
    plot_Nu(res, [0.02], 'RBC3DG4R4SDC23Ra1e5', ref_data, ax, 'SDC23', converged_from)
    plot_Nu(res, [0.02], 'RBC3DG4R4RKRa1e5', ref_data, ax, 'RK443', converged_from)
    plot_Nu(res, [0.02], 'RBC3DG4R4EulerRa1e5', ref_data, ax, 'RK111', converged_from)

    ax.set_xlabel('$t$')
    ax.set_ylabel('$Nu$')
    ax.set_title(r'$Ra=10^5 \quad \Delta t=0.02$')

    lines = ax.get_lines()
    for line, label in zip(lines[1::3], ['SDC44', 'SDC23', 'RK443', 'RK111'], strict=True):
        line.set_label(label)
    ax.legend(frameon=False)

    fig.tight_layout()
    savefig(fig, 'NuRa1e5SameDt', pad_inches=0.1)


if __name__ == '__main__':

    # plot_Nu_over_time_Ra1e5()
    # plot_Nu_over_time_Ra1e6()
    plot_Nusselt_Ra1e5_same_dt()

    plt.show()
