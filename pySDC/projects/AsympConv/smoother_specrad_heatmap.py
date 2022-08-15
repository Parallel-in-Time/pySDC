import matplotlib
import numpy as np
import scipy.linalg as LA

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def compute_and_plot_specrad():
    """
    Compute and plot spectral radius of smoother iteration matrix for a whole range of eigenvalues
    """
    # setup_list = [('LU', 'to0'), ('LU', 'toinf'), ('IE', 'to0'), ('IE', 'toinf')]
    # setup_list = [('LU', 'to0'), ('LU', 'toinf')]
    # setup_list = [('IE', 'to0'), ('IE', 'toinf')]
    # setup_list = [('LU', 'toinf'), ('IE', 'toinf')]
    setup_list = [('IE', 'full'), ('LU', 'full')]
    # setup_list = [('EX', 'to0'), ('PIC', 'to0')]

    # set up plotting parameters
    params = {
        'legend.fontsize': 24,
        'figure.figsize': (12, 8),
        'axes.labelsize': 24,
        'axes.titlesize': 24,
        'xtick.labelsize': 24,
        'ytick.labelsize': 24,
        'lines.linewidth': 3,
    }
    plt.rcParams.update(params)

    Nnodes = 3
    Nsteps = 4

    coll = CollGaussRadau_Right(Nnodes, 0, 1)
    Qmat = coll.Qmat[1:, 1:]

    Nmat = np.zeros((Nnodes, Nnodes))
    Nmat[:, -1] = 1

    Emat = np.zeros((Nsteps, Nsteps))
    np.fill_diagonal(Emat[1:, :], 1)

    for qd_type, conv_type in setup_list:

        if qd_type == 'LU':

            QT = coll.Qmat[1:, 1:].T
            [_, _, U] = LA.lu(QT, overwrite_a=True)
            QDmat = U.T

        elif qd_type == 'IE':

            QI = np.zeros(np.shape(coll.Qmat))
            for m in range(coll.num_nodes + 1):
                QI[m, 1 : m + 1] = coll.delta_m[0:m]
            QDmat = QI[1:, 1:]

        elif qd_type == 'EE':

            QE = np.zeros(np.shape(coll.Qmat))
            for m in range(coll.num_nodes + 1):
                QE[m, 0:m] = coll.delta_m[0:m]
            QDmat = QE[1:, 1:]

        elif qd_type == 'PIC':

            QDmat = np.zeros(np.shape(coll.Qmat[1:, 1:]))

        elif qd_type == 'EX':

            QT = coll.Qmat[1:, 1:].T
            [_, _, U] = LA.lu(QT, overwrite_a=True)
            QDmat = np.tril(U.T, k=-1)
            print(QDmat)

        else:
            raise NotImplementedError('qd_type %s is not implemented' % qd_type)

        # lim_specrad = max(abs(np.linalg.eigvals(np.eye(Nnodes) - np.linalg.inv(QDmat).dot(Qmat))))
        # print('qd_type: %s -- lim_specrad: %6.4e -- conv_type: %s' % (qd_type, lim_specrad, conv_type))

        if conv_type == 'to0':

            ilim_left = -4
            ilim_right = 2
            rlim_left = 2
            rlim_right = -4

        elif conv_type == 'toinf':

            ilim_left = 0
            ilim_right = 11
            rlim_left = 6
            rlim_right = 0

        elif conv_type == 'full':

            ilim_left = -10
            ilim_right = 11
            rlim_left = 10
            rlim_right = -11

        else:
            raise NotImplementedError('conv_type %s is not implemented' % conv_type)

        ilam_list = 1j * np.logspace(ilim_left, ilim_right, 201)
        rlam_list = -1 * np.logspace(rlim_left, rlim_right, 201)

        assert (rlim_right - rlim_left + 1) % 5 == 0
        assert (ilim_right - ilim_left - 1) % 5 == 0
        assert (len(rlam_list) - 1) % 5 == 0
        assert (len(ilam_list) - 1) % 5 == 0

        Prho = np.zeros((len(rlam_list), len(ilam_list)))

        for idr, rlam in enumerate(rlam_list):
            for idi, ilam in enumerate(ilam_list):
                dxlam = rlam + ilam

                mat = np.linalg.inv(np.eye(Nnodes * Nsteps) - dxlam * np.kron(np.eye(Nsteps), QDmat)).dot(
                    dxlam * np.kron(np.eye(Nsteps), (Qmat - QDmat)) + np.kron(Emat, Nmat)
                )
                mat = np.linalg.matrix_power(mat, Nnodes)

                Prho[idr, idi] = max(abs(np.linalg.eigvals(mat)))

        print(np.amax(Prho))

        fig, ax = plt.subplots(figsize=(15, 10))

        ax.set_xticks([i + 0.5 for i in range(0, len(rlam_list), int(len(rlam_list) / 5))])
        ax.set_xticklabels(
            [r'-$10^{%d}$' % i for i in range(rlim_left, rlim_right, int((rlim_right - rlim_left + 1) / 5))]
        )
        ax.set_yticks([i + 0.5 for i in range(0, len(ilam_list), int(len(ilam_list) / 5))])
        ax.set_yticklabels(
            [r'$10^{%d}i$' % i for i in range(ilim_left, ilim_right, int((ilim_right - ilim_left - 1) / 5))]
        )

        cmap = plt.get_cmap('Reds')
        pcol = plt.pcolor(Prho.T, cmap=cmap, norm=LogNorm(vmin=1e-09, vmax=1e-00))

        plt.colorbar(pcol)

        plt.xlabel(r'$Re(\Delta t\lambda)$')
        plt.ylabel(r'$Im(\Delta t\lambda)$')

        fname = (
            'data/heatmap_smoother_' + conv_type + '_Nsteps' + str(Nsteps) + '_M' + str(Nnodes) + '_' + qd_type + '.png'
        )
        plt.savefig(fname, transparent=True, bbox_inches='tight')


if __name__ == "__main__":
    compute_and_plot_specrad()
