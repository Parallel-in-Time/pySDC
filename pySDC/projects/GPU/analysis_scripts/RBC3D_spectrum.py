from pySDC.projects.GPU.analysis_scripts.process_RBC3D_data import get_pySDC_data
from pySDC.projects.GPU.analysis_scripts.RBC3D_plotting_utils import figsize_by_journal, get_plotting_style, savefig
import matplotlib.pyplot as plt


def plot_spectrum(res, dt, config_name, ax):  # pragma: no cover
    data = get_pySDC_data(res=res, dt=dt, config_name=config_name)

    spectrum = data['avg_spectrum']
    k = data['k']
    ax.loglog(k[spectrum > 1e-16], spectrum[spectrum > 1e-16], **get_plotting_style(config_name), markevery=5)
    ax.set_xlabel('$k$')
    ax.set_ylabel(r'$\|\hat{u}_x\|$')


def plot_spectra_Ra1e5():  # pragma: no cover
    fig, ax = plt.subplots(figsize=figsize_by_journal('Nature_CS', 1, 0.6))

    configs = [f'RBC3DG4R4{name}Ra1e5' for name in ['SDC34', 'SDC23', '', 'Euler', 'RK']]
    dts = [0.06, 0.06, 0.06, 0.02, 0.04]
    res = 32

    for config, dt in zip(configs, dts, strict=True):
        plot_spectrum(res, dt, config, ax)

    ax.legend(frameon=False)
    savefig(fig, 'RBC3D_spectrum_Ra1e5')


if __name__ == '__main__':
    plot_spectra_Ra1e5()

    plt.show()
