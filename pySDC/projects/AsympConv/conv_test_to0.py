import numpy as np
import scipy.linalg as LA

import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
from matplotlib import rc

from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right


def compute_and_plot_specrad(Nnodes, lam):
    """
    Compute and plot the spectral radius of the smoother for different step-sizes

    Args:
        Nnodes: number of collocation nodes
        lam: test parameter representing the spatial problem
    """

    coll = CollGaussRadau_Right(Nnodes, 0, 1)
    Qmat = coll.Qmat[1:, 1:]

    # do LU decomposition of QT (St. Martin's trick)
    QT = coll.Qmat[1:, 1:].T
    [_, _, U] = LA.lu(QT, overwrite_a=True)
    QDmat = U.T

    Nmat = np.zeros((Nnodes, Nnodes))
    Nmat[:, -1] = 1

    Nsteps_list = [64, 256]
    color_list = ['red', 'blue']
    marker_list = ['s', 'o']

    setup_list = zip(Nsteps_list, color_list, marker_list)

    xlist = [0.1 ** i for i in range(11)]

    rc('font', **{"sans-serif": ["Arial"], "size": 24})
    plt.subplots(figsize=(15, 10))

    for Nsteps, color, marker in setup_list:

        Emat = np.zeros((Nsteps, Nsteps))
        np.fill_diagonal(Emat[1:, :], 1)

        Prho_list = []
        predict_list = []
        for x in xlist:

            mat = np.linalg.inv(np.eye(Nnodes * Nsteps) - x * lam * np.kron(np.eye(Nsteps), QDmat)).dot(
                x * lam * np.kron(np.eye(Nsteps), (Qmat - QDmat)) + np.kron(Emat, Nmat))

            Prho_list.append(max(abs(np.linalg.eigvals(mat))))
            # predict_list.append((1 + x) ** (1.0 - 1.0 / (Nnodes * Nsteps)) * x ** (1.0 / (Nnodes * Nsteps)))
            predict_list.append(x ** (1.0 / (Nsteps)))

            if len(predict_list) > 1:
                print(x, predict_list[-1], Prho_list[-1], Prho_list[-2] / Prho_list[-1],
                      predict_list[-2] / predict_list[-1])

        plt.loglog(xlist, Prho_list, linestyle='-', linewidth=3, color=color, marker=marker, markersize=10,
                   label='spectral radius, L=' + str(Nsteps))
        plt.loglog(xlist, [item for item in predict_list], linestyle='--', linewidth=2, color=color, marker=marker,
                   markersize=10, label='estimate, L=' + str(Nsteps))

    ax = plt.gca()
    ax.invert_xaxis()

    plt.xlabel('time-step size')
    plt.ylabel('spectral radius')
    plt.legend(loc=3, numpoints=1)
    plt.grid()
    plt.ylim([1E-02, 1E01])

    if type(lam) is complex:
        fname = 'data/smoother_specrad_to0_L64+256_M' + str(Nnodes) + 'LU_imag.png'
    else:
        fname = 'data/smoother_specrad_to0_L64+256_M' + str(Nnodes) + 'LU_real.png'
    plt.savefig(fname, rasterized=True, transparent=True, bbox_inches='tight')


if __name__ == "__main__":
    compute_and_plot_specrad(Nnodes=3, lam=-1)
    compute_and_plot_specrad(Nnodes=3, lam=1j)
