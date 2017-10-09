import matplotlib
matplotlib.use('Agg')

import numpy as np
from matplotlib import pyplot as plt
from pylab import rcParams

from pySDC.implementations.problem_classes.FastWaveSlowWave_0D import swfw_scalar
from pySDC.implementations.datatype_classes.complex_mesh import mesh, rhs_imex_mesh
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right

from pySDC.core.Step import step


# noinspection PyShadowingNames
def compute_specrad():
    """
    Routine to compute spectral radius and norm  of the error propagation matrix E

    Returns:
        numpy.nparray: list of number of nodes
        numpy.nparray: list of fast lambdas
        numpy.nparray: list of spectral radii
        numpy.nparray: list of norms

    """
    problem_params = dict()
    # SET VALUE FOR lambda_slow AND VALUES FOR lambda_fast ###
    problem_params['lambda_s'] = np.array([1.0 * 1j], dtype='complex')
    problem_params['lambda_f'] = np.array([50.0 * 1j, 100.0 * 1j], dtype='complex')
    problem_params['u0'] = 1.0

    # initialize sweeper parameters
    sweeper_params = dict()
    # SET TYPE OF QUADRATURE NODES ###
    sweeper_params['collocation_class'] = CollGaussRadau_Right

    # initialize level parameters
    level_params = dict()
    level_params['dt'] = 1.0
    t0 = 0.0

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = swfw_scalar  # pass problem class
    description['problem_params'] = problem_params  # pass problem parameters
    description['dtype_u'] = mesh  # pass data type for u
    description['dtype_f'] = rhs_imex_mesh  # pass data type for f
    description['sweeper_class'] = imex_1st_order  # pass sweeper (see part B)
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = dict()  # pass step parameters

    nodes_v = np.arange(2, 10)
    specrad = np.zeros((3, np.size(nodes_v)))
    norm = np.zeros((3, np.size(nodes_v)))

    for i in range(0, np.size(nodes_v)):

        sweeper_params['num_nodes'] = nodes_v[i]
        description['sweeper_params'] = sweeper_params  # pass sweeper parameters

        # now the description contains more or less everything we need to create a step
        S = step(description=description)

        L = S.levels[0]
        P = L.prob

        u0 = S.levels[0].prob.u_exact(t0)
        S.init_step(u0)
        QE = L.sweep.QE[1:, 1:]
        QI = L.sweep.QI[1:, 1:]
        Q = L.sweep.coll.Qmat[1:, 1:]
        nnodes = L.sweep.coll.num_nodes
        dt = L.params.dt

        assert nnodes == nodes_v[i], 'Something went wrong during instantiation, nnodes is not correct, got %s' % nnodes

        for j in range(0, 2):
            LHS = np.eye(nnodes) - dt * (P.params.lambda_f[j] * QI + P.params.lambda_s[0] * QE)
            RHS = dt * ((P.params.lambda_f[j] + P.params.lambda_s[0]) * Q -
                        (P.params.lambda_f[j] * QI + P.params.lambda_s[0] * QE))
            evals, evecs = np.linalg.eig(np.linalg.inv(LHS).dot(RHS))
            specrad[j + 1, i] = np.linalg.norm(evals, np.inf)
            norm[j + 1, i] = np.linalg.norm(np.linalg.inv(LHS).dot(RHS), np.inf)

        if L.sweep.coll.left_is_node:
            # For Lobatto nodes, first column and row are all zeros, since q_1 = q_0; hence remove them
            QI = QI[1:, 1:]
            Q = Q[1:, 1:]
            # Eigenvalue of error propagation matrix in stiff limit: E = I - inv(QI)*Q
            evals, evecs = np.linalg.eig(np.eye(nnodes - 1) - np.linalg.inv(QI).dot(Q))
            norm[0, i] = np.linalg.norm(np.eye(nnodes - 1) - np.linalg.inv(QI).dot(Q), np.inf)
        else:
            evals, evecs = np.linalg.eig(np.eye(nnodes) - np.linalg.inv(QI).dot(Q))
            norm[0, i] = np.linalg.norm(np.eye(nnodes) - np.linalg.inv(QI).dot(Q), np.inf)
        specrad[0, i] = np.linalg.norm(evals, np.inf)

        print("Spectral radius of infinitely fast wave case > 1.0 for M=%2i" % nodes_v[np.argmax(specrad[0, :] > 1.0)])
        print("Spectral radius of > 1.0 for M=%2i" % nodes_v[np.argmax(specrad[1, :] > 1.0)])

    return nodes_v, problem_params['lambda_f'], specrad, norm


# noinspection PyShadowingNames
def plot_specrad(nodes_v, lambda_f, specrad, norm):
    """
    Plotting function for spectral radii and norms

    Args:
        nodes_v (numpy.nparray): list of number of nodes
        lambda_f (numpy.nparray): list of fast lambdas
        specrad (numpy.nparray): list of spectral radii
        norm (numpy.nparray): list of norms
    """
    fs = 8
    rcParams['figure.figsize'] = 2.5, 2.5
    fig = plt.figure()
    plt.plot(nodes_v, specrad[0, :], 'rd-', markersize=fs - 2, label=r'$\lambda_{\rm fast} = \infty$')
    plt.plot(nodes_v, specrad[1, :], 'bo-', markersize=fs - 2,
             label=r'$\lambda_{\rm fast} = %2.0f $' % lambda_f[0].imag)
    plt.plot(nodes_v, specrad[2, :], 'gs-', markersize=fs - 2,
             label=r'$\lambda_{\rm fast} = %2.0f $' % lambda_f[1].imag)
    plt.xlabel(r'Number of nodes $M$', fontsize=fs)
    plt.ylabel(r'Spectral radius  $\sigma\left( \mathbf{E} \right)$', fontsize=fs, labelpad=2)
    plt.legend(loc='lower right', fontsize=fs, prop={'size': fs})
    plt.xlim([np.min(nodes_v), np.max(nodes_v)])
    plt.ylim([0, 1.0])
    plt.yticks(fontsize=fs)
    plt.xticks(fontsize=fs)
    filename = 'data/stifflimit-specrad.png'
    fig.savefig(filename, bbox_inches='tight')

    fig = plt.figure()
    plt.plot(nodes_v, norm[0, :], 'rd-', markersize=fs - 2, label=r'$\lambda_{\rm fast} = \infty$')
    plt.plot(nodes_v, norm[1, :], 'bo-', markersize=fs - 2,
             label=r'$\lambda_{\rm fast} = %2.0f $' % lambda_f[0].imag)
    plt.plot(nodes_v, norm[2, :], 'gs-', markersize=fs - 2,
             label=r'$\lambda_{\rm fast} = %2.0f $' % lambda_f[1].imag)
    plt.xlabel(r'Number of nodes $M$', fontsize=fs)
    plt.ylabel(r'Norm  $\left|| \mathbf{E} \right||_{\infty}$', fontsize=fs, labelpad=2)
    plt.legend(loc='lower right', fontsize=fs, prop={'size': fs})
    plt.xlim([np.min(nodes_v), np.max(nodes_v)])
    plt.ylim([0, 2.4])
    plt.yticks(fontsize=fs)
    plt.xticks(fontsize=fs)
    filename = 'data/stifflimit-norm.png'
    fig.savefig(filename, bbox_inches='tight')


if __name__ == "__main__":
    nodes_v, lambda_f, specrad, norm = compute_specrad()
    plot_specrad(nodes_v, lambda_f, specrad, norm)
