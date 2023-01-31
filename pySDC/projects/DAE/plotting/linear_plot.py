import pySDC.helpers.plot_helper as plt_helper
import numpy as np
import pickle

# pragma: no cover
def linear_plot():
    '''Loads solution data from an .npy file and plots specified parameters with respect to each other on a linear axis'''

    data = pickle.load(open("data/dae_conv_data.p", "rb"))
    plt_helper.setup_mpl()
    fig, ax = plt_helper.newfig(textwidth=500, scale=0.89)  # Create a figure containing a single axes.

    # ax.plot(data['dt'], data['ue'], label=r'$U_e$', lw=0.6, color='r')
    # ax.plot(data['dt'], data['solution'][:, 7], label=r'$U_8$', marker='x', markersize=2, lw=0.6, color='b')
    # ax.plot(data['dt'], data['solution'][0], label=r'$x$', lw=0.6, marker='x', markersize=3)
    # ax.plot(data['dt'], data['solution'][1], label=r'$y$', lw=0.6, marker='x', markersize=3)
    ax.plot(data['dt'], data['solution'][2], label=r'$dx$', lw=0.6, marker='x', markersize=3)
    ax.plot(data['dt'], data['solution'][3], label=r'$dy$', lw=0.6, marker='x', markersize=3)
    # ax.plot(data['dt'], data['solution'][4], label=r'$lambda$', lw=0.6, marker='x', markersize=3)

    # title='Convergence plot two stage implicit Runge-Kutta with Gauss nodes'
    # ax.set(xlabel=r'time (s)', ylabel=r'voltage (V)')
    ax.set(xlabel=r'$x$')
    ax.grid(visible=True)
    fig.tight_layout()
    ax.legend(loc='upper left')

    fname = 'data/lin_plot_1'
    plt_helper.savefig(fname)
    # plt.savefig('../results/problematic_good.png')


if __name__ == "__main__":
    linear_plot()
