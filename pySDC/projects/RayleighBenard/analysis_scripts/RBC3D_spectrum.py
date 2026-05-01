from pySDC.projects.RayleighBenard.analysis_scripts.process_RBC3D_data import get_pySDC_data
from pySDC.projects.RayleighBenard.analysis_scripts.plotting_utils import (
    get_plotting_style,
    savefig,
    figsize,
)
import matplotlib.pyplot as plt


def plot_spectrum(res, dt, config_name, ax, **plotting_params):  # pragma: no cover
    data = get_pySDC_data(res=res, dt=dt, config_name=config_name)

    spectrum = data['avg_spectrum']
    k = data['k']
    ax.loglog(
        k[spectrum > 1e-16],
        spectrum[spectrum > 1e-16],
        **{**get_plotting_style(config_name), **plotting_params},
        markevery=5,
    )
    ax.set_xlabel('$k$')
    ax.set_ylabel(r'$\|\hat{u}_x\|$')


def plot_spectra_Ra1e5():  # pragma: no cover
    fig, ax = plt.subplots(figsize=figsize(scale=1, ratio=0.6))

    configs = [f'RBC3DG4R4{name}Ra1e5' for name in ['SDC23', 'SDC44', 'Euler', 'RK']]
    dts = [0.02, 0.02, 0.02, 0.02]
    res = 32

    for config, dt in zip(configs, dts, strict=True):
        plot_spectrum(res, dt, config, ax)

    ax.legend(frameon=False)
    savefig(fig, 'RBC3D_spectrum_Ra1e5')


def plot_spectra_Ra1e6():  # pragma: no cover
    fig, ax = plt.subplots(figsize=figsize(scale=1, ratio=0.6))

    configs = [f'RBC3DG4R4{name}Ra1e6' for name in ['SDC44', 'SDC23', 'RK']]
    dts = [0.01, 0.01, 0.01]
    res = 64

    for config, dt in zip(configs, dts, strict=True):
        plot_spectrum(res, dt, config, ax)

    ax.legend(frameon=False)
    savefig(fig, 'RBC3D_spectrum_Ra1e6')


def plot_all_spectra(ax=None):  # pragma: no cover
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize(scale=1, ratio=0.6))
    else:
        fig = None

    Ras = ['1e5', '1e6', '1e7']
    dts = [0.06, 0.01, 0.005]
    res = [32, 64, 128]
    colors = [
        'tab:blue',
        'tab:orange',
        'tab:green',
    ]
    markers = ['x', 'o', '.']

    for Ra, dt, _res, color, marker in zip(Ras, dts, res, colors, markers, strict=True):
        config = f'RBC3DG4R4SDC23Ra{Ra}'
        plot_spectrum(_res, dt, config, ax, label=f'$Ra$={Ra}', color=color, marker=marker)

    ax.legend(frameon=False)
    if fig is not None:
        savefig(fig, 'RBC3D_all_spectra')


def plot_microscopic_verification():  # pragma: no cover
    from pySDC.projects.RayleighBenard.analysis_scripts.RBC3D_order import compare_order

    fig, axs = plt.subplots(1, 2, figsize=figsize(scale=1, ratio=0.45))
    plot_all_spectra(axs[0])
    compare_order(1e5, axs[1])
    for ax in axs:
        ax.set_box_aspect(1)
    fig.tight_layout()
    savefig(fig, 'RBC3D_microscopic_verification')


if __name__ == '__main__':
    # plot_spectra_Ra1e5()
    # plot_spectra_Ra1e6()
    # plot_all_spectra()
    plot_microscopic_verification()

    plt.show()
