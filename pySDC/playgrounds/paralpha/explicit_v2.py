import numpy as np
import scipy.sparse as sp
import scipy.linalg as sl

from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.problem_classes.AdvectionEquation_ND_FD_periodic import advectionNd_periodic

def get_config():
    problem_params = dict()
    problem_params['nvars'] = (1024,)
    problem_params['c'] = 1
    problem_params['freq'] = (2,)
    problem_params['type'] = 'upwind'
    problem_params['ndim'] = 1
    problem_params['order'] = 1

    problem = advectionNd_periodic(problem_params)

    time_params = dict()
    time_params['dt'] = 5E-04
    time_params['Tend'] = 1E-02

    return problem, time_params

def run_explicit_euler(problem, time_params):

    u0 = problem.u_exact(t=0)

    dt = time_params['dt']
    L = int(time_params['Tend'] / time_params['dt'])
    t = 0
    # explicit euler
    us = problem.dtype_u(u0)
    for l in range(L):
        us = us + dt * problem.eval_f(us, t=t)
        t += dt

    err_euler = abs(us - problem.u_exact(t=time_params['Tend']))
    print('seq err = ', err_euler)

    return err_euler


def get_alpha(m0, rhs1, uL, Lip, L):

    eps = np.finfo(complex).eps
    norm_rhs1 = max([abs(r) for r in rhs1])
    p = L * 6 * eps * norm_rhs1 / (m0 + L * Lip * m0 - L * 3 * eps * norm_rhs1)
    alpha = -p / 2 + np.sqrt(p ** 2 + 2 * p) / 2
    m0 = (alpha + L * Lip) / (1 - alpha) * m0 + L * 3 * eps * norm_rhs1 / alpha + L * 3 * eps * abs(uL)

    return alpha, m0

def run_paralpha(problem, time_params, err_euler):
    # coll = CollGaussRadau_Right(M, 0, 1)
    # Q = coll.Qmat[1:, 1:]

    dt = time_params['dt']
    L = int(time_params['Tend'] / time_params['dt'])
    print(L)

    kmax = 10
    k = 0
    Lip = problem.params.c * dt * max(abs(np.linalg.eigvals(problem.A.todense()))) / problem.dx
    Lip = problem.params.c * dt / problem.dx
    # lip = problem_params['c'] * time_params['dt'] * max(abs(np.linalg.eigvals(problem.A.todense()))) / problem.dx

    print(Lip)

    u = [problem.dtype_u(problem.params.nvars, val=0.0) for _ in range(L)]

    u0 = [problem.u_exact(t=0) for _ in range(L)]

    m0 = abs(problem.u_exact(0) - problem.u_exact(time_params['Tend']))
    err = 99
    t = 0
    while k < kmax and err > err_euler * 5:

        rhs1 = []
        for l in range(L):
            rhs1.append(dt * problem.eval_f(u[l], t + dt * l))
        rhs1[0] += u0[0]

        alpha, m0 = get_alpha(m0, rhs1, u[L-1], Lip, L)
        Ea = sp.eye(k=-1, m=L) + alpha * sp.eye(k=L - 1, m=L)
        d, S = np.linalg.eig(Ea.todense())
        Sinv = np.linalg.inv(S)  # S @ d @ Sinv = Ea

        print(f'Diagonalization error: {np.linalg.norm(S @ np.diag(d) @ Sinv - Ea, np.inf)}')

        rhs1[0] -= alpha * u[-1]

        for i in range(L):
            tmp = problem.dtype_u(problem.params.nvars, val=0.0+0.0j)
            for j in range(L):
                tmp += Sinv[i, j] * rhs1[j]
            u[i] = problem.dtype_u(tmp)

        for l in range(L):
            u[l] = 1.0 / (1.0 - d[l]) * u[l]

        u1 = [problem.dtype_u(u[l], val=0.0) for l in range(L)]
        for i in range(L):
            tmp = problem.dtype_u(problem.params.nvars, val=0.0+0.0j)
            for j in range(L):
                tmp += S[i, j] * u1[j]
            u[i] = problem.dtype_u(tmp)

        err = abs(u[-1] - problem.u_exact(t=time_params['Tend']))
        k += 1

        print('paralpha err = ', k, alpha, m0, err)


if __name__ == '__main__':
    problem, time_params = get_config()
    err_euler = run_explicit_euler(problem, time_params)
    run_paralpha(problem, time_params, err_euler)

    pass