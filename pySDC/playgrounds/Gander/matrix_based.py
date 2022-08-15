from pySDC.core.Sweeper import sweeper
import numpy as np
import matplotlib.pyplot as plt


def iteration_vs_estimate():

    M = 5
    K = 10
    swee = sweeper({'collocation_class': CollGaussRadau_Right, 'num_nodes': M})
    Q = swee.coll.Qmat[1:, 1:]
    Qd = swee.get_Qdelta_implicit(swee.coll, 'IE')[1:, 1:]
    # Qd = swee.get_Qdelta_implicit(swee.coll, 'LU')[1:, 1:]
    # Qd = swee.get_Qdelta_explicit(swee.coll, 'EE')[1:, 1:]
    print(np.linalg.norm(Q - Qd, np.inf))
    exit()
    I = np.eye(M)

    lam = -0.7
    print(1 / np.linalg.norm(Qd, np.inf), np.linalg.norm(np.linalg.inv(Qd), np.inf))
    C = I - lam * Q
    P = I - lam * Qd
    Pinv = np.linalg.inv(P)

    R = Pinv.dot(lam * (Q - Qd))
    rho = max(abs(np.linalg.eigvals(R)))
    infnorm = np.linalg.norm(R, np.inf)
    twonorm = np.linalg.norm(R, 2)

    # uex = np.exp(lam)
    u0 = np.ones(M, dtype=np.complex128)
    uex = np.linalg.inv(C).dot(u0)[-1]
    u = u0.copy()
    # u = np.random.rand(M)
    res = u0 - C.dot(u)
    err = []
    resnorm = []
    for k in range(K):
        u += Pinv.dot(res)
        res = u0 - C.dot(u)
        err.append(abs(u[-1] - uex))
        resnorm.append(np.linalg.norm(res, np.inf))
        print(k, resnorm[-1], err[-1])

    plt.figure()
    plt.semilogy(range(K), err, 'o-', color='red', label='error')
    plt.semilogy(range(K), resnorm, 'd-', color='orange', label='residual')
    plt.semilogy(range(K), [err[0] * rho**k for k in range(K)], '--', color='green', label='spectral')
    plt.semilogy(range(K), [err[0] * infnorm**k for k in range(K)], '--', color='blue', label='infnorm')
    plt.semilogy(range(K), [err[0] * twonorm**k for k in range(K)], '--', color='cyan', label='twonorm')
    plt.semilogy(range(K), [err[0] * ((-1 / lam) ** (1 / M)) ** k for k in range(K)], 'x--', color='black', label='est')
    plt.semilogy(
        range(K),
        [err[0] * (abs(lam) * np.linalg.norm(Q - Qd)) ** k for k in range(K)],
        'x-.',
        color='black',
        label='est',
    )
    # plt.semilogy(range(K), [err[0] * (1/abs(lam) + np.linalg.norm((I-np.linalg.inv(Qd).dot(Q)) ** k)) for k in range(K)], 'x-.', color='black', label='est')
    plt.grid()
    plt.legend()
    plt.show()


def estimates_over_lambda():
    M = 5
    K = 10
    swee = sweeper({'collocation_class': CollGaussRadau_Right, 'num_nodes': M})
    Q = swee.coll.Qmat[1:, 1:]
    # Qd = swee.get_Qdelta_implicit(swee.coll, 'IE')[1:, 1:]
    Qd = swee.get_Qdelta_implicit(swee.coll, 'LU')[1:, 1:]

    Qdinv = np.linalg.inv(Qd)
    I = np.eye(M)
    # lam = 1/np.linalg.eigvals(Q)[0]
    # print(np.linalg.inv(I - lam * Q))
    # print(np.linalg.norm(1/lam * Qdinv, np.inf))
    # exit()

    # print(np.linalg.norm((I-Qdinv.dot(Q)) ** K))
    # lam_list = np.linspace(start=20j, stop=0j, num=1000, endpoint=False)
    lam_list = np.linspace(start=-1000, stop=0, num=100, endpoint=True)

    rho = []
    est = []
    infnorm = []
    twonorm = []
    for lam in lam_list:
        # C = I - lam * Q
        P = I - lam * Qd
        Pinv = np.linalg.inv(P)

        R = Pinv.dot(lam * (Q - Qd))
        # w, V = np.linalg.eig(R)
        # Vinv = np.linalg.inv(V)
        # assert np.linalg.norm(V.dot(np.diag(w)).dot(Vinv) - R) < 1E-14, np.linalg.norm(V.dot(np.diag(w)).dot(Vinv) - R)
        rho.append(max(abs(np.linalg.eigvals(R))))
        # est.append(0.62*(-1/lam) ** (1/(M-1)))  # M = 3
        # est.append(0.57*(-1/lam) ** (1/(M-1)))  # M = 5
        # est.append(0.71*(-1/lam) ** (1/(M-1.5)))  # M = 7
        # est.append(0.92*(-1/lam) ** (1/(M-2.5)))  # M = 9
        # est.append((-1/lam) ** (1/(M)))
        est.append(1000 * (1 / abs(lam)) ** (1 / (1)))
        # est.append(abs(lam))
        # est.append(np.linalg.norm((I-Qdinv.dot(Q)) ** M) + 1/abs(lam))
        # est.append(np.linalg.norm(np.linalg.inv(I - lam * Qd), np.inf))
        # est.append(np.linalg.norm(np.linalg.inv(I - np.linalg.inv(I - lam * np.diag(Qd)).dot(lam*np.tril(Qd, -1))), np.inf))

        infnorm.append(np.linalg.norm(np.linalg.matrix_power(R, M), np.inf))
        # infnorm.append(np.linalg.norm(Vinv.dot(R).dot(V), np.inf))
        twonorm.append(np.linalg.norm(np.linalg.matrix_power(R, M), 2))
        # twonorm.append(np.linalg.norm(Vinv.dot(R).dot(V), 2))
    plt.figure()
    plt.semilogy(lam_list, rho, '--', color='green', label='spectral')
    plt.semilogy(lam_list, est, '--', color='red', label='est')
    # plt.semilogy(lam_list, infnorm, 'x--', color='blue', label='infnorm')
    # plt.semilogy(lam_list, twonorm, 'd--', color='cyan', label='twonorm')
    # plt.semilogy(np.imag(lam_list), rho, '--', color='green', label='spectral')
    # plt.semilogy(np.imag(lam_list), est, '--', color='red', label='est')
    # plt.semilogy(np.imag(lam_list), infnorm, '--', color='blue', label='infnorm')
    # plt.semilogy(np.imag(lam_list), twonorm, '--', color='cyan', label='twonorm')
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # iteration_vs_estimate()
    estimates_over_lambda()
