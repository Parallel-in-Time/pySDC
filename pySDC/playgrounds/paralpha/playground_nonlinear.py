import numpy as np
import matplotlib.pyplot as plt

from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.problem_classes.AllenCahn_1D_FD import allencahn_front_fullyimplicit


def run():

    nsteps = 4
    L = 1
    M = 3
    N = 127

    alpha = 0.001
    dt = 0.01
    t0 = 0.0
    nblocks = nsteps // L

    # initialize problem (ALLEN-CAHN)
    problem_params = dict()
    problem_params['nvars'] = N
    problem_params['dw'] = -0.00
    problem_params['eps'] = 0.04
    problem_params['newton_maxiter'] = 200
    problem_params['newton_tol'] = 1E-08
    problem_params['lin_tol'] = 1E-08
    problem_params['lin_maxiter'] = 100
    problem_params['radius'] = 0.5
    problem_params['interval'] = (-0.0, 1.0)

    prob = allencahn_front_fullyimplicit(problem_params)

    IL = np.eye(L)
    IM = np.eye(M)
    IN = np.eye(N + 2)

    coll = CollGaussRadau_Right(M, 0, 1)

    Q = coll.Qmat[1:, 1:]
    A = prob.A.todense()
    A[0, :] = 0
    A[-1, :] = 0
    # A[0, 0] = 1
    # A[-1, -1] = 1

    E = np.zeros((L, L))
    np.fill_diagonal(E[1:, :], 1)
    Ealpha = np.zeros((L, L))
    np.fill_diagonal(Ealpha[1:, :], 1)
    if L > 1:
        Ealpha[0, -1] = alpha

    H = np.zeros((M, M))
    H[:, -1] = 1

    uinit = np.zeros(N + 2)
    # uinit[1:-1] = prob.u_exact(t=t0).values
    # uinit[0] = 0.5 * (1 + np.tanh((prob.params.interval[0]) / (np.sqrt(2) * prob.params.eps)))
    # uinit[-1] = 0.5 * (1 + np.tanh((prob.params.interval[1]) / (np.sqrt(2) * prob.params.eps)))
    uinit[1:-1] = np.sin(2*np.pi*prob.xvalues)
    uinit[0] = np.sin(2*np.pi*prob.params.interval[0])
    uinit[-1] = np.sin(2*np.pi*prob.params.interval[1])

    uex = np.zeros(L * M * (N + 2))
    for l in range(L):
        for m in range(M):

            t = t0 + dt * l + coll.nodes[m] * dt
            istart = m * (N + 2) + M * l * (N + 2)
            iend = istart + N + 2
            uex[istart + 1:iend - 1] = np.sin(2 * np.pi * prob.xvalues) * np.exp(-(2*np.pi)**2*t)
            uex[istart] = np.sin(2 * np.pi * prob.params.interval[0]) * np.exp(-(2*np.pi)**2*t)
            uex[iend-1] = np.sin(2 * np.pi * prob.params.interval[1]) * np.exp(-(2*np.pi)**2*t)

    u0_M = np.kron(np.ones(M), uinit)
    u0 = np.kron(np.concatenate([[1], [0] * (L - 1)]), u0_M)[:, None]

    u = np.kron(np.ones(L), u0_M)[:, None]
    u = uex.copy()[:, None]

    # plt.plot(uinit)
    # plt.plot(uex[-(N + 2):])
    # plt.show()
    # exit()

    maxiter = 20

    for nb in range(nblocks):

        outer_k = 0
        outer_restol = 1E-10
        outer_res = u0 - ((np.kron(np.kron(IL, IM), IN) - np.kron(np.kron(E, H), IN)) @ u - dt * np.kron(np.kron(IL, Q), A) @ u - 0.0 * dt * np.kron(np.kron(IL, Q), IN) @
                          (- 2.0 / prob.params.eps ** 2 * u * (1.0 - u) * (1.0 - 2 * u) - 6.0 * prob.params.dw * u * (1.0 - u)))
        print(np.linalg.norm(outer_res, np.inf))
        plt.plot(outer_res)
        plt.show()
        exit()
        inner_iter = 0

        while outer_k < maxiter and np.linalg.norm(outer_res, np.inf) > outer_restol:
            outer_k += 1

            # inner_k = 0
            # inner_restol = 1E-11
            # v = u.copy()
            # inner_res = u0 + np.kron(np.kron(E, H), IN) @ u - (np.kron(np.kron(IL, IM), IN) @ v - dt * np.kron(np.kron(IL, Q), A) @ v - dt * np.kron(np.kron(IL, Q), IN) @
            #               (- 2.0 / prob.params.eps ** 2 * v * (1.0 - v) * (1.0 - 2 * v) - 6.0 * prob.params.dw * v * (1.0 - v)))
            #
            # while inner_k < 1 and np.linalg.norm(inner_res, np.inf) > inner_restol:
            #     inner_k += 1
            #     for l in range(L):
            #         A_grad = np.zeros((N + 2, N + 2))
            #         for m in range(M):
            #             istart = m * (N + 2) + M * l * (N + 2)
            #             iend = istart + N + 2
            #             # print(istart, iend)
            #             tmp = v[istart:iend, 0]
            #             A_grad += A - 2.0 / prob.params.eps ** 2 * np.diag(
            #                 (1.0 - tmp) * (1.0 - 2.0 * tmp) - tmp * ((1.0 - 2.0 * tmp) + 2.0 * (1.0 - tmp))) \
            #                       - 6.0 * prob.params.dw * np.diag((1.0 - tmp) - tmp)
            #         A_grad /= M
            #         A_grad[0, :] = 0
            #         A_grad[-1, :] = 0
            #
            #         P_grad_loc = -(np.kron(IM, IN) - dt * np.kron(Q, A_grad))
            #         v[M * l * (N + 2): M * (l + 1) * (N + 2)] -= np.linalg.inv(P_grad_loc).dot(inner_res[M * l * (N + 2): M * (l + 1) * (N + 2)])
            #
            #     inner_res = u0 + np.kron(np.kron(E, H), IN) @ u - (np.kron(np.kron(IL, IM), IN) @ v - dt * np.kron(np.kron(IL, Q), A) @ v - dt * np.kron(np.kron(IL, Q), IN) @
            #               (- 2.0 / prob.params.eps ** 2 * v * (1.0 - v) * (1.0 - 2 * v) - 6.0 * prob.params.dw * v * (1.0 - v)))
            #     print('Newton-Jacobi:', inner_k,  np.linalg.norm(inner_res, np.inf))
            #
            # u = v.copy()
            # outer_res = u0 - ((np.kron(np.kron(IL, IM), IN) - np.kron(np.kron(E, H), IN)) @ u - dt * np.kron(np.kron(IL, Q), A) @ u - dt * np.kron(np.kron(IL, Q), IN) @
            #       (- 2.0 / prob.params.eps ** 2 * u * (1.0 - u) * (1.0 - 2 * u) - 6.0 * prob.params.dw * u * (1.0 - u)))

            A_grad = np.zeros((N + 2, N + 2))
            for l in range(L):
                for m in range(M):
                    istart = m * (N + 2) + M * l * (N + 2)
                    iend = istart + N + 2
                    # print(istart, iend)
                    tmp = u[istart:iend, 0]
                    A_grad += A - 2.0 / prob.params.eps ** 2 * np.diag((1.0 - tmp) * (1.0 - 2.0 * tmp) - tmp * ((1.0 - 2.0 * tmp) + 2.0 * (1.0 - tmp))) \
                     - 6.0 * prob.params.dw * np.diag((1.0 - tmp) - tmp)
            A_grad /= L * M
            A_grad[0, :] = 0
            A_grad[-1, :] = 0
            C_grad = -(np.kron(np.kron(IL, IM), IN) - dt * np.kron(np.kron(IL, Q), A_grad) - np.kron(np.kron(E, H), IN))
            Calpha_grad = -(np.kron(np.kron(IL, IM), IN) - dt * np.kron(np.kron(IL, Q), A_grad) - np.kron(np.kron(Ealpha, H), IN))
            Calpha_grad_inv = np.linalg.inv(Calpha_grad)

            inner_k = 0
            inner_restol = 1E-11
            e = np.zeros(L * M * (N + 2))[:, None]

            inner_res = outer_res - C_grad @ e
            while inner_k < 1 and np.linalg.norm(inner_res, np.inf) > inner_restol:
                inner_k += 1
                e += Calpha_grad_inv @ inner_res

                inner_res = outer_res - C_grad @ e
                print('Newton-Paralpha:', inner_k, np.linalg.norm(inner_res, np.inf))

            inner_iter += inner_k
            u -= e
            print(e[0], e[-1], u[0], u[-1])
            outer_res = u0 - ((np.kron(np.kron(IL, IM), IN) - np.kron(np.kron(E, H), IN)) @ u - dt * np.kron(np.kron(IL, Q), A) @ u - dt * np.kron(np.kron(IL, Q), IN) @
                        (- 2.0 / prob.params.eps ** 2 * u * (1.0 - u) * (1.0 - 2 * u) - 6.0 * prob.params.dw * u * (1.0 - u)))


        ures = u[-(N + 2):, 0]
        uinit = u[-(N + 2):, 0]
        u0_M = np.kron(np.ones(M), uinit)
        u0 = np.kron(np.concatenate([[1], [0] * (L - 1)]), u0_M)[:, None]
        u = u0.copy()
        u = np.kron(np.ones(L), u0_M)[:, None]

        uex = np.zeros(N + 2)
        t = t0 + (nb + 1) * L * dt
        uex[1:-1] = prob.u_exact(t=t).values
        v = 3.0 * np.sqrt(2) * prob.params.eps * prob.params.dw
        uex[0] = 0.5 * (1 + np.tanh((prob.params.interval[0] - v * t) / (np.sqrt(2) * prob.params.eps)))
        uex[-1] = 0.5 * (1 + np.tanh((prob.params.interval[1] - v * t) / (np.sqrt(2) * prob.params.eps)))

        err = np.linalg.norm(uex[:, None] - ures[:, None], np.inf)
        print(outer_k, inner_iter, np.linalg.norm(outer_res, np.inf), err)

    uinit[1:-1] = prob.u_exact(t=t0).values
    uinit[0] = 0.5 * (1 + np.tanh((prob.params.interval[0]) / (np.sqrt(2) * prob.params.eps)))
    uinit[-1] = 0.5 * (1 + np.tanh((prob.params.interval[1]) / (np.sqrt(2) * prob.params.eps)))
    # plt.plot(uex[1:-1])
    plt.plot(uex[:, None]-uinit[:, None])
    # plt.plot(uinit)
    # plt.plot(ures[1:-1])
    plt.show()


if __name__ == '__main__':
    run()
