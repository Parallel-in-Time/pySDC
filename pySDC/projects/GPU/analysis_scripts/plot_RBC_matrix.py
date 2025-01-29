from pySDC.implementations.problem_classes.RayleighBenard import RayleighBenard
import numpy as np
import matplotlib.pyplot as plt
from pySDC.helpers.plot_helper import figsize_by_journal, setup_mpl


def plot_preconditioners():  # pragma: no cover
    P = RayleighBenard(nx=3, nz=5, Dirichlet_recombination=True, left_preconditioner=True)
    dt = 1.0

    A = P.M + dt * P.L
    A_b = P.put_BCs_in_matrix(P.M + dt * P.L)
    A_r = P.spectral.put_BCs_in_matrix(A) @ P.Pr
    A_l = P.Pl @ P.spectral.put_BCs_in_matrix(A) @ P.Pr

    fig, axs = plt.subplots(1, 4, figsize=figsize_by_journal('TUHH_thesis', 1, 0.4), sharex=True, sharey=True)

    for M, ax in zip([A, A_b, A_r, A_l], axs):
        ax.imshow((M / abs(M)).real + (M / abs(M)).imag, rasterized=False, cmap='Spectral')

    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig('plots/RBC_matrix.pdf', bbox_inches='tight', dpi=300)
    plt.show()


def plot_ultraspherical():  # pragma: no cover
    from pySDC.helpers.spectral_helper import ChebychevHelper, UltrasphericalHelper

    N = 16
    cheby = ChebychevHelper(N=N)
    ultra = UltrasphericalHelper(N=N)

    D_cheby = cheby.get_differentiation_matrix()
    I_cheby = cheby.get_Id()

    fig, axs = plt.subplots(2, 3, figsize=figsize_by_journal('TUHH_thesis', 0.9, 0.65), sharex=True, sharey=True)

    axs[0, 0].imshow(D_cheby / abs(D_cheby))
    axs[1, 0].imshow(I_cheby / abs(I_cheby))

    for i in range(2):
        D_ultra = ultra.get_differentiation_matrix(p=i + 1)
        I_ultra = ultra.get_basis_change_matrix(0, i + 1)
        axs[0, i + 1].imshow(D_ultra / abs(D_ultra))
        axs[1, i + 1].imshow(I_ultra / abs(I_ultra))
        axs[1, i + 1].set_xlabel(rf'$T \rightarrow C^{{({{{i+1}}})}}$')
    for ax in axs.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    axs[0, 0].set_ylabel('Differentiation')
    axs[1, 0].set_ylabel('Left preconditioner')
    axs[1, 0].set_xlabel(r'$T \rightarrow T$')

    axs[0, 0].set_title('first derivative')
    axs[0, 1].set_title('first derivative')
    axs[0, 2].set_title('second derivative')
    fig.savefig('plots/ultraspherical_matrix.pdf', bbox_inches='tight', dpi=300)
    plt.show()


def plot_DCT():  # pragma: no cover
    fig, axs = plt.subplots(1, 3, figsize=figsize_by_journal('TUHH_thesis', 1, 0.28), sharey=True)

    N = 8
    color = 'black'

    x = np.linspace(0, 3, N)
    y = x**3 - 4 * x**2
    axs[0].plot(y, marker='o', color=color)

    y_m = np.append(y, y[::-1])
    axs[1].scatter(np.arange(2 * N)[::2], y_m[::2], marker='<', color=color)
    axs[1].scatter(np.arange(2 * N)[1::2], y_m[1::2], marker='>', color=color)
    axs[1].plot(np.arange(2 * N), y_m, color=color)

    v = y_m[::2]
    axs[2].plot(np.arange(N), v, color=color, marker='x')

    axs[0].set_title('original')
    axs[1].set_title('mirrored')
    axs[2].set_title('periodically reordered')

    for ax in axs:
        # ax.set_xlabel(r'$n$')
        ax.set_yticks([])
    fig.savefig('plots/DCT_via_FFT.pdf', bbox_inches='tight', dpi=300)


if __name__ == '__main__':
    setup_mpl()
    plot_DCT()
    # plot_ultraspherical()
    plt.show()
