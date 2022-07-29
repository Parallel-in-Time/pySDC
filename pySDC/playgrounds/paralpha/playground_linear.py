import numpy as np
import matplotlib.pyplot as plt

from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.problem_classes.AdvectionEquation_ND_FD_periodic import advectionNd_periodic


def run():

    nsteps = 8
    L = 4
    M = 3
    N = 16

    alpha = 0.001
    dt = 0.2
    t0 = 0.0
    nblocks = nsteps // L

    # initialize problem (ADVECTION)
    ndim = 1
    problem_params = dict()
    problem_params['ndim'] = ndim  # will be iterated over
    problem_params['order'] = 6  # order of accuracy for FD discretization in space
    problem_params['type'] = 'center'  # order of accuracy for FD discretization in space
    problem_params['c'] = 0.1  # diffusion coefficient
    problem_params['freq'] = tuple(2 for _ in range(ndim))  # frequencies
    problem_params['nvars'] = tuple(N for _ in range(ndim))  # number of dofs
    problem_params['direct_solver'] = False  # do GMRES instead of LU
    problem_params['liniter'] = 10  # number of GMRES iterations

    prob = advectionNd_periodic(problem_params)

    IL = np.eye(L)
    LM = np.eye(M)
    IN = np.eye(N)

    coll = CollGaussRadau_Right(M, 0, 1)

    Q = coll.Qmat[1:, 1:]
    A = prob.A.todense()

    E = np.zeros((L, L))
    np.fill_diagonal(E[1:, :], 1)
    Ealpha = np.zeros((L, L))
    np.fill_diagonal(Ealpha[1:, :], 1)
    if L > 1:
        Ealpha[0, -1] = alpha

    H = np.zeros((M, M))
    H[:, -1] = 1

    C = np.kron(np.kron(IL, LM), IN) - dt * np.kron(np.kron(IL, Q), A) - np.kron(np.kron(E, H), IN)
    Calpha = np.kron(np.kron(IL, LM), IN) - dt * np.kron(np.kron(IL, Q), A) - np.kron(np.kron(Ealpha, H), IN)
    Calpha_inv = np.linalg.inv(Calpha)

    uinit = prob.u_exact(t=t0)
    u0_M = np.kron(np.ones(M), uinit)
    u0 = np.kron(np.concatenate([[1], [0] * (L - 1)]), u0_M)[:, None]

    u = np.kron(np.ones(L * M), uinit)[:, None]
    u = u0.copy()

    maxiter = 10

    for nb in range(nblocks):

        k = 0
        restol = 1e-10
        res = u0 - C @ u
        while k < maxiter and np.linalg.norm(res, np.inf) > restol:
            k += 1
            u += Calpha_inv @ res
            res = u0 - C @ u
            uex = prob.u_exact(t=t0 + (nb + 1) * L * dt)[:, None]
            err = np.linalg.norm(uex - u[-N:], np.inf)
            print(k, np.linalg.norm(res, np.inf), err)

        uinit = u[-N:, 0]
        u0_M = np.kron(np.ones(M), uinit)
        u0 = np.kron(np.concatenate([[1], [0] * (L - 1)]), u0_M)[:, None]
        u = u0.copy()

    # plt.plot(uex-u[-N:])
    # plt.plot(uinit)
    # plt.plot(u[-N:])
    # plt.show()


if __name__ == '__main__':
    run()
