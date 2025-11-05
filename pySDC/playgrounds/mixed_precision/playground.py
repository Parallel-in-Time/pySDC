import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg




from qmat import QDELTA_GENERATORS

from pySDC.core.collocation import CollBase
from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_unforced
from pySDC.implementations.problem_classes.AllenCahn_2D_FD import allencahn_fullyimplicit


class counter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            # print('iter %3i\trk = %s' % (self.niter, str(rk)))
            print('   LIN: %3i' % self.niter)

def linear():

    # instantiate problem
    prob = heatNd_unforced(
        nvars=1023,  # number of degrees of freedom
        nu=0.1,  # diffusion coefficient
        freq=4,  # frequency for the test value
        bc='dirichlet-zero',  # boundary conditions
    )

    coll = CollBase(num_nodes=3, tleft=0, tright=1, node_type='LEGENDRE', quad_type='RADAU-RIGHT')

    generator = QDELTA_GENERATORS['LU'](
        # for algebraic types (LU, ...)
        Q=coll.generator.Q,
        # for MIN in tables, MIN-SR-S ...
        nNodes=coll.num_nodes,
        nodeType=coll.node_type,
        quadType=coll.quad_type,
        # for time-stepping types, MIN-SR-NS
        nodes=coll.nodes,
        tLeft=coll.tleft,
    )

    dt = 0.001

    # shrink collocation matrix: first line and column deals with initial value, not needed here
    Q = coll.Qmat[1:, 1:]
    QDmat = generator.genCoeffs(k=None)

    # build system matrix M of collocation problem
    M = sp.eye(prob.nvars[0] * coll.num_nodes) - dt * sp.kron(Q, prob.A)

    P = sp.eye(prob.nvars[0] * coll.num_nodes) - dt * sp.kron(QDmat, prob.A)

    # get initial value at t0 = 0
    u0 = prob.u_exact(t=0)
    # fill in u0-vector as right-hand side for the collocation problem
    u0_coll = np.kron(np.ones(coll.num_nodes), u0)
    # get exact solution at Tend = dt
    uend = prob.u_exact(t=dt)

    # solve collocation problem directly
    u_coll = sp.linalg.spsolve(M, u0_coll)
    # compute error
    err = np.linalg.norm(u_coll[-prob.nvars[0] :] - uend, np.inf)
    print('Error exact inverse:', err)

    kmax = 10
    tol = 1E-10
    uk = np.zeros(prob.nvars[0] * coll.num_nodes, dtype='float64')
    res = np.zeros(prob.nvars[0] * coll.num_nodes, dtype='float32')
    res[:] = u0_coll - M.dot(uk)
    for k in range(kmax):

        # does not work well with float32
        # res[:] = u0_coll + dt * sp.kron(Q-QDmat, prob.A).dot(uk)
        # uk = sp.linalg.spsolve(P,res)

        # works well with float32
        uk += sp.linalg.spsolve(P, res)
        res[:] = u0_coll - M.dot(uk)

        # but how can we do this for nonlinear problems? it seems there is no iterative refinement for nonlinear problems??
        # May need to go to Outer-Newton, inner SDC. See also https://zenodo.org/records/6835437

        resnorm = np.linalg.norm(res, np.inf)
        err = np.linalg.norm(uk[-prob.nvars[0]:] - uend, np.inf)
        print(k, resnorm, err)
        if resnorm < tol:
            break
    # print(k, resnorm)

def nonlinear():
    # instantiate problem
    prob = allencahn_fullyimplicit(nvars=(64,64))

    coll = CollBase(num_nodes=4, tleft=0, tright=1, node_type='LEGENDRE', quad_type='RADAU-RIGHT')

    generator = QDELTA_GENERATORS['MIN-SR-S'](
        # for algebraic types (LU, ...)
        Q=coll.generator.Q,
        # for MIN in tables, MIN-SR-S ...
        nNodes=coll.num_nodes,
        nodeType=coll.node_type,
        quadType=coll.quad_type,
        # for time-stepping types, MIN-SR-NS
        nodes=coll.nodes,
        tLeft=coll.tleft,
    )
    QDmat = generator.genCoeffs(k=None)

    dt = 0.001 / 2

    # shrink collocation matrix: first line and column deals with initial value, not needed here
    Q = coll.Qmat[1:, 1:]

    # c = coll.nodes
    # V = np.fliplr(np.vander(c))
    # C = np.diag(c)
    # R = np.diag([1 / i for i in range(1,coll.num_nodes+1)])
    # print(C @ V @ R @ np.linalg.inv(V) - Q)
    # exit()


    u0 = prob.u_exact(t=0).flatten()
    # fill in u0-vector as right-hand side for the collocation problem
    u0_coll = np.kron(np.ones(coll.num_nodes), u0)

    nvars = prob.nvars[0] * prob.nvars[1]

    un = np.zeros(nvars * coll.num_nodes, dtype='float64')
    fn = np.zeros(nvars * coll.num_nodes, dtype='float64')
    g = np.zeros(nvars * coll.num_nodes, dtype='float64')
    # vk = np.zeros(nvars * coll.num_nodes, dtype='float64')

    ksum = 0
    n = 0
    n_max = 100
    tol_newton = 1E-10

    count = counter()

    while n < n_max:
        # form the function g with g(u) = 0
        for m in range(coll.num_nodes):
            fn[m * nvars: (m + 1) * nvars] = prob.eval_f(un[m * nvars: (m + 1) * nvars],
                                                         t=dt * coll.nodes[m]).flatten()
        g[:] = u0_coll - (un - dt * sp.kron(Q, sp.eye(nvars)).dot(fn))

        # if g is close to 0, then we are done
        res_newton = np.linalg.norm(g, np.inf)
        print('Newton:', n, res_newton)

        if res_newton < tol_newton:
            break

        # assemble dg
        dg = -sp.eye(nvars * coll.num_nodes) + dt * sp.kron(Q, sp.eye(nvars)).dot(sp.kron(sp.eye(coll.num_nodes), prob.A) + 1.0 / prob.eps**2 * sp.diags((1.0 - (prob.nu + 1) * un ** prob.nu), offsets=0))

        # Newton
        # vk =  sp.linalg.spsolve(dg, g)
        # vk = sp.linalg.gmres(dg, g, x0=np.zeros_like(vk), maxiter=10, atol=1E-10, callback=count)[0]
        # iter_count += count.niter
        # un -= vk
        # continue

        # Collocation
        # Emulate Newton
        # dgP = -sp.eye(nvars * coll.num_nodes) + dt * sp.kron(Q, sp.eye(nvars)).dot(sp.kron(sp.eye(coll.num_nodes), prob.A) + 1.0 / prob.eps**2 * sp.diags((1.0 - (prob.nu + 1) * un ** prob.nu), offsets=0))
        # Linear SDC (e.g. with diagonal preconditioner)
        # dgP = -sp.eye(nvars * coll.num_nodes) + dt * sp.kron(QDmat, sp.eye(nvars)).dot(sp.kron(sp.eye(coll.num_nodes), prob.A) + 1.0 / prob.eps**2 * sp.diags((1.0 - (prob.nu + 1) * un ** prob.nu), offsets=0))
        # Linear SDC with frozen Jacobian
        dgP = -sp.eye(nvars * coll.num_nodes) + dt * sp.kron(QDmat, prob.A + 1.0 / prob.eps**2 * sp.diags((1.0 - (prob.nu + 1) * un[0:nvars] ** prob.nu), offsets=0))
        # dgP = dgP.astype('float64')

        # Diagonalization
        # D, V = np.linalg.eig(Q)
        # Vinv = np.linalg.inv(V)
        # dg_diag = -sp.eye(nvars * coll.num_nodes) + dt * sp.kron(sp.diags(D), prob.A + 1.0 / prob.eps**2 * sp.diags((1.0 - (prob.nu + 1) * un[0:nvars] ** prob.nu), offsets=0))
        # dg_diag = dg_diag.astype('complex64')

        vk = np.zeros(nvars * coll.num_nodes, dtype='float64')
        res = np.zeros(nvars * coll.num_nodes, dtype='float64')

        res[:] = g - dg.dot(vk)
        k = 0
        tol_sdc = 1E-11
        kmax = 1
        while k < kmax:
            # Newton/Collocation: works well with float32 for both P and res, but not for vk
            # vk += sp.linalg.spsolve(dgP, res)
            vk += sp.linalg.gmres(dgP, res, x0=np.zeros_like(res), maxiter=1, atol=1E-10, callback=count)[0]

            # Newton/Diagonalization
            # vk += np.real(sp.kron(V,sp.eye(nvars)).dot(sp.linalg.spsolve(dg_diag, sp.kron(Vinv,sp.eye(nvars)).dot(res))))
            # vk += np.real(sp.kron(V,sp.eye(nvars)).dot(sp.linalg.gmres(dg_diag, sp.kron(Vinv,sp.eye(nvars)).dot(res), x0=np.zeros_like(res), maxiter=1, atol=1E-10, callback=count)[0]))

            res[:] = g - dg.dot(vk)
            resnorm = np.linalg.norm(res, np.inf)
            print('   SDC:', n, k, ksum, resnorm)

            if resnorm < tol_sdc:
                break
            k += 1
            ksum += 1

        # Update
        un -= vk

        # increase Newton iteration count
        n += 1

def nonlinear_sdc():
    # instantiate problem
    prob = allencahn_fullyimplicit(nvars=(64,64))

    coll = CollBase(num_nodes=4, tleft=0, tright=1, node_type='LEGENDRE', quad_type='RADAU-RIGHT')

    generator = QDELTA_GENERATORS['MIN-SR-S'](
        # for algebraic types (LU, ...)
        Q=coll.generator.Q,
        # for MIN in tables, MIN-SR-S ...
        nNodes=coll.num_nodes,
        nodeType=coll.node_type,
        quadType=coll.quad_type,
        # for time-stepping types, MIN-SR-NS
        nodes=coll.nodes,
        tLeft=coll.tleft,
    )
    QDmat = generator.genCoeffs(k=None)

    dt = 0.001 / 2

    # shrink collocation matrix: first line and column deals with initial value, not needed here
    Q = coll.Qmat[1:, 1:]

    u0 = prob.u_exact(t=0).flatten()
    # fill in u0-vector as right-hand side for the collocation problem
    u0_coll = np.kron(np.ones(coll.num_nodes), u0)

    nvars = prob.nvars[0] * prob.nvars[1]

    uk = np.zeros(nvars * coll.num_nodes, dtype='float64')
    fk = np.zeros(nvars * coll.num_nodes, dtype='float64')
    rhs = np.zeros(nvars * coll.num_nodes, dtype='float64')

    ksum = 0
    k = 0
    k_max = 100
    tol_sdc = 1E-10

    count = counter()

    while k < k_max:
        # eval rhs
        for m in range(coll.num_nodes):
            fk[m * nvars: (m + 1) * nvars] = prob.eval_f(uk[m * nvars: (m + 1) * nvars],
                                                         t=dt * coll.nodes[m]).flatten()

        resnorm = np.linalg.norm(u0_coll - (uk - dt * sp.kron(Q, sp.eye(nvars)).dot(fk)), np.inf)
        print('SDC:', k, ksum, resnorm)
        if resnorm < tol_sdc:
            break

        g = np.zeros(nvars * coll.num_nodes, dtype='float64')
        vn = np.zeros(nvars * coll.num_nodes, dtype='float64')
        fn = np.zeros(nvars * coll.num_nodes, dtype='float64')
        n = 0
        n_max = 100
        tol_newton = 1E-11
        while n < n_max:
            for m in range(coll.num_nodes):
                fn[m * nvars: (m + 1) * nvars] = prob.eval_f(vn[m * nvars: (m + 1) * nvars],
                                                             t=dt * coll.nodes[m]).flatten()
            g[:] = u0_coll + dt * sp.kron(Q-QDmat, sp.eye(nvars)).dot(fk) - (vn - dt * sp.kron(QDmat, sp.eye(nvars)).dot(fn))

            # if g is close to 0, then we are done
            res_newton = np.linalg.norm(g, np.inf)
            print('  Newton:', n, res_newton)
            n += 1
            if res_newton < tol_newton:
                break

            # assemble dg
            dg = -sp.eye(nvars * coll.num_nodes) + dt * sp.kron(QDmat, sp.eye(nvars)).dot(sp.kron(sp.eye(coll.num_nodes), prob.A) + 1.0 / prob.eps**2 * sp.diags((1.0 - (prob.nu + 1) * vn ** prob.nu), offsets=0))

            # Newton
            # vk =  sp.linalg.spsolve(dg, g)
            vn -= sp.linalg.gmres(dg, g, x0=np.zeros_like(vn), maxiter=100, atol=1E-12, callback=count)[0]

        uk = vn.copy()

        k += 1


if __name__ == "__main__":
    # linear()
    # nonlinear()
    nonlinear_sdc()
