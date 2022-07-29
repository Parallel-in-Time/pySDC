import numpy as np
import scipy as sp
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.helpers.transfer_helper import interpolation_matrix_1d


def get_transfer_matrix_Q(f_nodes, c_nodes):
    """
    Helper routine to quickly define transfer matrices between sets of nodes (fully Lagrangian)
    Args:
        f_nodes: fine nodes
        c_nodes: coarse nodes

    Returns:
        matrix containing the interpolation weights
    """
    nnodes_f = len(f_nodes)
    nnodes_c = len(c_nodes)

    tmat = np.zeros((nnodes_f, nnodes_c))

    for i in range(nnodes_f):
        xi = f_nodes[i]
        for j in range(nnodes_c):
            den = 1.0
            num = 1.0
            for k in range(nnodes_c):
                if k == j:
                    continue
                else:
                    den *= c_nodes[j] - c_nodes[k]
                    num *= xi - c_nodes[k]
            tmat[i, j] = num / den

    return tmat


def SDC():

    M = 9
    Mc = int((M + 1) / 2)
    # Mc = 1
    dt = 1.0
    lamb = -1.0
    tol = 1e-10

    coll = CollGaussRadau_Right(M, 0, 1)
    collc = CollGaussRadau_Right(Mc, 0, 1)
    collc2 = CollGaussRadau_Right(1, 0, 1)

    Q = coll.Qmat[1:, 1:]
    Qc = collc.Qmat[1:, 1:]
    Qc2 = collc2.Qmat[1:, 1:]

    _, _, U = sp.linalg.lu(Q.T, overwrite_a=False)
    Qd = U.T
    _, _, U = sp.linalg.lu(Qc.T, overwrite_a=False)
    Qdc = U.T
    _, _, U = sp.linalg.lu(Qc2.T, overwrite_a=False)
    Qdc2 = U.T

    # Qd = np.zeros((M, M))
    # Qdc = np.zeros((Mc,Mc))

    I = get_transfer_matrix_Q(coll.nodes, collc.nodes)
    R = get_transfer_matrix_Q(collc.nodes, coll.nodes)

    Id = np.eye(M)
    #
    # print(I)
    # print(R)

    C = Id - dt * lamb * Q
    Cc = np.eye(Mc) - dt * lamb * Qc
    Cc2 = np.eye(1) - dt * lamb * Qc2
    P = Id - dt * lamb * Qd
    Pc = np.eye(Mc) - dt * lamb * Qdc
    Pc2 = np.eye(1) - dt * lamb * Qdc2
    Pinv = np.linalg.inv(P)
    Pcinv = np.linalg.inv(Pc)

    u0 = 1.0
    u = np.zeros(M, dtype=np.complex128)
    u[0] = u0

    res = C.dot(u) - np.ones(M) * u0
    k = 0
    while np.linalg.norm(res, np.inf) > tol and k < 100:
        u += Pinv.dot(np.ones(M) * u0 - C.dot(u))
        res = C.dot(u) - np.ones(M) * u0
        k += 1
        print(k, np.linalg.norm(res, np.inf))
    print()

    I2 = get_transfer_matrix_Q(collc.nodes, collc2.nodes)
    R2 = get_transfer_matrix_Q(collc2.nodes, collc.nodes)

    K = k
    E = np.zeros((K, K))
    np.fill_diagonal(E[1:, :], 1)

    S = np.kron(np.eye(K), P) - np.kron(E, P - C)

    Rfull = np.kron(np.eye(K), R)
    Ifull = np.kron(np.eye(K), I)
    R2full = np.kron(np.eye(K), R2)
    I2full = np.kron(np.eye(K), I2)
    # Sc = Rfull.dot(S).dot(Ifull)
    Sc = np.kron(np.eye(K), Pc) - np.kron(E, Pc - Cc)
    Sc2 = np.kron(np.eye(K), Pc2) - np.kron(E, Pc2 - Cc2)
    Scinv = np.linalg.inv(Sc)
    Sinv = np.linalg.inv(S)

    Sc2inv = np.linalg.inv(Sc2)

    Sdiaginv = np.linalg.inv(np.kron(np.eye(K), P))
    Scdiaginv = np.linalg.inv(np.kron(np.eye(K), Pc))
    Sc2diaginv = np.linalg.inv(np.kron(np.eye(K), Pc2))
    u0vec = np.kron(np.ones(K), np.ones(M) * u0)
    u = np.zeros(M * K, dtype=np.complex128)
    l = 0
    res = C.dot(u[-M:]) - np.ones(M) * u0
    while np.linalg.norm(res, np.inf) > tol and l < K:
        # u += Sainv.dot(u0vec - S.dot(u))
        u += Sdiaginv.dot(u0vec - S.dot(u))
        uH = Rfull.dot(u)
        uHold = uH.copy()
        rhsH = Rfull.dot(u0vec) + Sc.dot(uH) - Rfull.dot(S.dot(u))
        uH = Scinv.dot(rhsH)
        # uH += Scdiaginv.dot(rhsH - Sc.dot(uH))
        # uH2 = R2full.dot(uH)
        # uH2old = uH2.copy()
        # rhsH2 = R2full.dot(rhsH) + Sc2.dot(uH2) - R2full.dot(Sc.dot(uH))
        # uH2 = Sc2inv.dot(rhsH2)
        # uH += I2full.dot(uH2 - uH2old)
        # uH += Scdiaginv.dot(rhsH - Sc.dot(uH))
        u += Ifull.dot(uH - uHold)
        u += Sdiaginv.dot(u0vec - S.dot(u))
        res = C.dot(u[-M:]) - np.ones(M) * u0
        l += 1
        print(l, np.linalg.norm(res, np.inf))
    print()

    Ea = E.copy()
    Ea[0, -1] = 1e00
    Sa = np.kron(np.eye(K), P) - np.kron(Ea, P - C)
    Sainv = np.linalg.inv(Sa)

    u0vec = np.kron(np.ones(K), np.ones(M) * u0)
    u = np.zeros(M * K, dtype=np.complex128)
    l = 0
    res = C.dot(u[-M:]) - np.ones(M) * u0
    while np.linalg.norm(res, np.inf) > tol and l < K:
        u += Sainv.dot(u0vec - S.dot(u))
        res = C.dot(u[-M:]) - np.ones(M) * u0
        l += 1
        print(l, np.linalg.norm(res, np.inf))
    print()

    Da, Va = np.linalg.eig(Ea)
    Da = np.diag(Da)
    Vainv = np.linalg.inv(Va)
    # print(Ea - Va.dot(Da).dot(Vainv))
    # exit()

    Dafull = np.kron(np.eye(K), P) - np.kron(Da, P - C)
    # Dafull = Ifull.dot(np.kron(np.eye(K), Pc) - np.kron(Da, Pc - Cc)).dot(Rfull)
    Dafull = np.kron(np.eye(K) - Da, np.eye(M) - P)
    DaPfull = np.kron(np.eye(K), P)
    Dafullinv = np.linalg.inv(Dafull)
    DaPfullinv = np.linalg.inv(DaPfull)
    Vafull = np.kron(Va, np.eye(M))
    Vafullinv = np.kron(Vainv, np.eye(M))

    u0vec = np.kron(np.ones(K), np.ones(M) * u0)
    u = np.zeros(M * K, dtype=np.complex128)
    l = 0
    res = C.dot(u[-M:]) - np.ones(M) * u0
    while np.linalg.norm(res, np.inf) > tol and l < K:
        rhs = Vafullinv.dot(u0vec - Sa.dot(u))
        # x = np.zeros(u.shape, dtype=np.complex128)
        # x += DaPfullinv.dot(rhs - Dafull.dot(x))
        # x += DaPfullinv.dot(rhs - Dafull.dot(x))
        # x += DaPfullinv.dot(rhs - Dafull.dot(x))
        # x += DaPfullinv.dot(rhs - Dafull.dot(x))
        # u += x
        u += Dafullinv.dot(rhs)
        u = Vafull.dot(u)
        res = C.dot(u[-M:]) - np.ones(M) * u0
        l += 1
        print(l, np.linalg.norm(res, np.inf))
    print()

    # T = np.eye(M) - Pinv.dot(C)
    # Tc = I.dot(np.eye(Mc) - Pcinv.dot(Cc)).dot(R)
    #
    # MF = np.eye(K * M) - np.kron(E, T)
    # MG = np.eye(K * M) - np.kron(E, Tc)
    # MGinv = np.linalg.inv(MG)
    #
    # tol = np.linalg.norm(res, np.inf)
    # u = np.zeros(M * K)
    # # u[0] = u0
    # u0vec = np.kron(np.ones(K), Pinv.dot(np.ones(M) * u0))
    # res = C.dot(u[-M:]) - np.ones(M) * u0
    # l = 0
    #
    # while np.linalg.norm(res, np.inf) > tol and l < K:
    #
    #     u = MGinv.dot(u0vec) + (np.eye(M * K) - MGinv.dot(MF)).dot(u)
    #     res = C.dot(u[-M:]) - np.ones(M) * u0
    #     l += 1
    #     print(l, np.linalg.norm(res, np.inf))
    # print()
    #
    # u = np.zeros(M * K)
    # utmpf = np.zeros(M * K)
    # utmpc = np.zeros(M * K)
    # # u[0] = u0
    # u0vec = np.kron(np.ones(K), Pinv.dot(np.ones(M) * u0))
    # res = C.dot(u[-M:]) - np.ones(M) * u0
    # l = 0
    #
    # for k in range(1, K):
    #     utmpc[k * M: (k + 1) * M] = Tc.dot(u[(k-1) * M: k * M])
    #
    # while np.linalg.norm(res, np.inf) > tol and l < K:
    #
    #     for k in range(1, K):
    #         utmpf[k * M: (k + 1) * M] = T.dot(u[(k - 1) * M: k * M])
    #
    #     for k in range(1, K):
    #         u[k * M: (k+1) * M] = Tc.dot(u[(k-1) * M: k * M]) + utmpf[k * M: (k + 1) * M] - utmpc[k * M: (k + 1) * M] + u0vec[(k-1) * M: k * M]
    #         utmpc[k * M: (k + 1) * M] = Tc.dot(u[(k - 1) * M: k * M])
    #
    #     res = C.dot(u[-M:]) - np.ones(M) * u0
    #     l += 1
    #     print(l, np.linalg.norm(res, np.inf))
    # print()
    #
    # u = np.zeros(M * K)
    # # u[0] = u0
    # u0vec = np.kron(np.ones(K), Pinv.dot(np.ones(M) * u0))
    # res = C.dot(u[-M:]) - np.ones(M) * u0
    # uold = u.copy()
    # l = 0
    #
    # while np.linalg.norm(res, np.inf) > tol and l < K:
    #
    #     for k in range(1, K):
    #         u[k * M: (k+1) * M] = T.dot(uold[(k-1) * M: k * M]) + Tc.dot(u[(k-1) * M: k * M] - uold[(k-1) * M: k * M]) + u0vec[(k-1) * M: k * M]
    #
    #     res = C.dot(u[-M:]) - np.ones(M) * u0
    #     l += 1
    #     uold = u.copy()
    #     print(l, np.linalg.norm(res, np.inf))
    # print()


def Jacobi():
    N = 127
    dx = 1.0 / (N + 1)
    nu = 1.0
    K = 20
    stencil = [-1, 2, -1]
    A = sp.sparse.diags(stencil, [-1, 0, 1], shape=(N, N), format='csc')
    A *= nu / (dx**2)

    D = sp.sparse.diags(2.0 * A.diagonal(), 0, shape=(N, N), format='csc')
    Dinv = sp.sparse.diags(0.5 * 1.0 / A.diagonal(), 0, shape=(N, N), format='csc')

    f = np.ones(N)
    f = np.zeros(N)
    u = np.sin([int(3.0 * N / 4.0) * np.pi * (i + 1) * dx for i in range(N)])
    res = f - A.dot(u)

    for k in range(1, K + 1):
        u += Dinv.dot(res)
        res = f - A.dot(u)
        print(k, np.linalg.norm(res, np.inf))
    print()
    # print(u)

    tol = np.linalg.norm(res, np.inf)

    Nc = int((N + 1) / 2 - 1)
    dxc = 1.0 / (Nc + 1)

    Ac = sp.sparse.diags(stencil, [-1, 0, 1], shape=(Nc, Nc), format='csc')
    Ac *= nu / (dxc**2)

    Dc = sp.sparse.diags(2.0 * Ac.diagonal(), 0, shape=(Nc, Nc), format='csc')
    Dcinv = sp.sparse.diags(0.5 * 1.0 / Ac.diagonal(), 0, shape=(Nc, Nc), format='csc')

    fine_grid = np.array([(i + 1) * dx for i in range(N)])
    coarse_grid = np.array([(i + 1) * dxc for i in range(Nc)])
    I = sp.sparse.csc_matrix(interpolation_matrix_1d(fine_grid, coarse_grid, k=6, periodic=False, equidist_nested=True))
    R = sp.sparse.csc_matrix(I.T)

    T = sp.sparse.csc_matrix(sp.sparse.eye(N) - Dinv.dot(A))
    Tc = sp.sparse.csc_matrix(I.dot(sp.sparse.eye(Nc) - Dcinv.dot(Ac)).dot(R))

    fvec = np.kron(np.ones(K), Dinv.dot(f))
    u = np.zeros(N * K)
    u = np.kron(np.ones(K), np.sin([int(3.0 * N / 4.0) * np.pi * (i + 1) * dx for i in range(N)]))
    # u[0: N] = np.sin([int(3.0 * N / 4.0) * np.pi * (i + 1) * dx for i in range(N)])
    res = f - A.dot(u[0:N])
    uold = u.copy()
    l = 0
    while np.linalg.norm(res, np.inf) > tol and l < K:

        for k in range(1, K):
            u[k * N : (k + 1) * N] = (
                T.dot(uold[(k - 1) * N : k * N])
                + Tc.dot(u[(k - 1) * N : k * N] - uold[(k - 1) * N : k * N])
                + fvec[(k - 1) * N : k * N]
            )

        res = f - A.dot(u[-N:])
        l += 1
        uold = u.copy()
        print(l, np.linalg.norm(res, np.inf))
    print()
    # print(u[-N:])

    E = np.zeros((K, K))
    np.fill_diagonal(E[1:, :], 1)
    E = sp.sparse.csc_matrix(E)

    Rfull = sp.sparse.kron(sp.sparse.eye(K), R)
    Ifull = sp.sparse.kron(sp.sparse.eye(K), I)
    S = sp.sparse.kron(sp.sparse.eye(K), D) - sp.sparse.kron(E, D - A)
    Sc = Rfull.dot(S).dot(Ifull)
    # Sc = sp.sparse.kron(sp.sparse.eye(K), Dcinv) - sp.sparse.kron(E, Dc - Ac)
    Scinv = np.linalg.inv(Sc.todense())
    # Sinv = np.linalg.inv(S.todense())

    Sdiaginv = sp.sparse.kron(sp.sparse.eye(K), Dinv)
    Scdiaginv = sp.sparse.kron(sp.sparse.eye(K), Dcinv)
    u = np.zeros(N * K)
    # u[0: N] = np.sin([int(3.0 * N / 4.0) * np.pi * (i + 1) * dx for i in range(N)])
    u = np.kron(np.ones(K), np.sin([int(3.0 * N / 4.0) * np.pi * (i + 1) * dx for i in range(N)]))
    l = 0
    fvec = np.kron(np.ones(K), f)
    res = f - A.dot(u[0:N])
    while np.linalg.norm(res, np.inf) > tol and l < K:
        # u += Sainv.dot(u0vec - S.dot(u))
        u += Sdiaginv.dot(fvec - S.dot(u))
        uH = Rfull.dot(u)
        uHold = uH.copy()
        rhsH = Rfull.dot(fvec) + Sc.dot(uH) - Rfull.dot(S.dot(u))
        uH += np.ravel(Scinv.dot(rhsH))
        # uH += Scdiaginv.dot(rhsH - Sc.dot(uH))
        # uH2 = R2full.dot(uH)
        # uH2old = uH2.copy()
        # rhsH2 = R2full.dot(rhsH) + Sc2.dot(uH2) - R2full.dot(Sc.dot(uH))
        # uH2 = Sc2inv.dot(rhsH2)
        # uH += I2full.dot(uH2 - uH2old)
        # uH += Scdiaginv.dot(rhsH - Sc.dot(uH))
        u += Ifull.dot(uH - uHold)
        u += Sdiaginv.dot(fvec - S.dot(u))
        res = f - A.dot(u[-N:])
        l += 1
        print(l, np.linalg.norm(res, np.inf))
        # print(u[-N:])
    print()

    # print(np.linalg.inv(A.todense()).dot(f))


if __name__ == "__main__":
    # SDC()
    Jacobi()
