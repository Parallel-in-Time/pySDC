"""
This script introduces ParaDiag for nonlinear problems with the van der Pol oscillator as an example.

ParaDiag works by diagonalizing the "top layer" of Kronecker products that make up the circularized composite
collocation problem.
However, in nonlinear problems, the problem cannot be written as a matrix and therefore we cannot write the composite
collocation problem as a matrix.
There are two approaches for dealing with this. We can do IMEX splitting, where we treat only the linear part implicitly.
The ParaDiag preconditioner is then only made up of the linear implicit part and we can again write this as a matrix and
do the diagonalization just like for linear problems. The non-linear part then comes in via the residual on the right
hand side.
The second approach is to average Jacobians. The non-linear problems are solved with a Newton scheme, where the Jacobian
matrix is computed based on the current solution and then inverted in each Newton iteration. In order to write the
ParaDiag preconditioner as a matrix with Kronecker products and then only diagonalize the outermost part, we need to
have the same Jacobian on all steps.
The ParaDiag iteration then proceeds as follows:
    - (1) Compute residual of composite collocation problem
    - (2) Average the solution across the steps and nodes as preparation for computing the average Jacobian
    - (3) Weighted FFT in time to diagonalize E_alpha
    - (4) Solve for the increment by inverting the averaged Jacobian from (2) on the subproblems on the different steps
          and nodes.
    - (5) Weighted iFFT in time
    - (6) Increment solution
As IMEX ParaDiag is a trivial extension of ParaDiag for linear problems, we focus on the second approach here.
"""

import numpy as np
import scipy.sparse as sp
import sys

from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit as sweeper_class
from pySDC.implementations.problem_classes.Van_der_Pol_implicit import vanderpol

# setup output
out_file = open('data/step_9_B_out.txt', 'w')


def my_print(*args, **kwargs):
    for output in [sys.stdout, out_file]:
        print(*args, **kwargs, file=output)


# setup parameters
L = 4
M = 3
alpha = 1e-4
restol = 1e-8
dt = 0.1

# setup infrastructure
prob = vanderpol(newton_maxiter=1, mu=1e0, crash_at_maxiter=False)
N = prob.init[0]

# make problem work on complex data
prob.init = tuple([*prob.init[:2]] + [np.dtype('complex128')])

# setup global solution array
u = np.zeros((L, M, N), dtype=complex)

# setup collocation problem
sweep = sweeper_class({'num_nodes': M, 'quad_type': 'RADAU-RIGHT'}, None)

# initial conditions
u[0, :, :] = prob.u_exact(t=0)

my_print(
    f'Running ParaDiag test script for van der Pol with mu={prob.mu} and {L} time steps and {M} collocation nodes.'
)


"""
Setup matrices that make up the composite collocation problem. We do not set up the full composite collocation problem
here, however. See https://arxiv.org/abs/2103.12571 for the meaning of the matrices.
"""
I_M = sp.eye(M)

H_M = sp.eye(M).tolil() * 0
H_M[:, -1] = 1

Q = sweep.coll.Qmat[1:, 1:]

E_alpha = sp.diags(
    [
        -1.0,
    ]
    * (L - 1),
    offsets=-1,
).tolil()
E_alpha[0, -1] = -alpha

gamma = alpha ** (-np.arange(L) / L)
D_alpha_diag_vals = np.fft.fft(1 / gamma * E_alpha[:, 0].toarray().flatten(), norm='backward')

J = sp.diags(gamma)
J_inv = sp.diags(1 / gamma)

G = [(D_alpha_diag_vals[l] * H_M + I_M).tocsc() for l in range(L)]  # MxM

# prepare diagonalization of QG^{-1}
w = []
S = []
S_inv = []

for l in range(L):
    # diagonalize QG^-1 matrix
    if M > 1:
        _w, _S = np.linalg.eig(Q @ sp.linalg.inv(G[l]).toarray())
    else:
        _w, _S = np.linalg.eig(Q / (G[l].toarray()))
    _S_inv = np.linalg.inv(_S)
    w.append(_w)
    S.append(_S)
    S_inv.append(_S_inv)

"""
Setup functions for computing matrix-vector productions on the steps and for computing the residual of the composite
collocation problem
"""


def mat_vec(mat, vec):
    """
    Matrix vector product

    Args:
        mat (np.ndarray or scipy.sparse) : Matrix
        vec (np.ndarray) : vector

    Returns:
        np.ndarray: mat @ vec
    """
    res = np.zeros_like(vec)
    for l in range(vec.shape[0]):
        for k in range(vec.shape[0]):
            res[l] += mat[l, k] * vec[k]
    return res


def residual(_u, u0):
    """
    Compute the residual of the composite collocation problem

    Args:
        _u (np.ndarray): Current iterate
        u0 (np.ndarray): Initial conditions

    Returns:
        np.ndarray: LMN size array with the residual
    """
    res = _u * 0j
    for l in range(L):
        # build step local residual

        # communicate initial conditions for each step
        if l == 0:
            res[l, ...] = u0[l, ...]
        else:
            res[l, ...] = _u[l - 1, -1, ...]

        # evaluate and subtract integral over right hand side functions
        f_evals = np.array([prob.eval_f(_u[l, m], 0) for m in range(M)])
        Qf = mat_vec(Q, f_evals)
        res[l, ...] -= _u[l] - dt * Qf

    return res


# do ParaDiag
sol_paradiag = u.copy() * 0j
u0 = u.copy()
niter = 0
res = residual(sol_paradiag, u0)
while np.max(np.abs(res)) > restol:
    # compute all-at-once residual
    res = residual(sol_paradiag, u0)

    # compute solution averaged across the L steps and M nodes. This is the difference to ParaDiag for linear problems.
    u_avg = prob.u_init
    u_avg[:] = np.mean(sol_paradiag, axis=(0, 1))

    # weighted FFT in time
    x = np.fft.fft(mat_vec(J_inv.toarray(), res), axis=0)

    # perform local solves of "collocation problems" on the steps in parallel
    y = np.empty_like(x)
    for l in range(L):

        # perform local solves on the collocation nodes in parallel
        x1 = S_inv[l] @ x[l]
        x2 = np.empty_like(x1)
        for m in range(M):
            x2[m, :] = prob.solve_jacobian(x1[m], w[l][m] * dt, u=u_avg, t=l * dt)
        z = S[l] @ x2
        y[l, ...] = sp.linalg.spsolve(G[l], z)

    # inverse FFT in time and increment
    sol_paradiag += mat_vec(J.toarray(), np.fft.ifft(y, axis=0))

    res = residual(sol_paradiag, u0)
    niter += 1
    assert niter < 99, 'ParaDiag did not converge for nonlinear problem!'
my_print(f'Needed {niter} ParaDiag iterations, stopped at residual {np.max(np.abs(res)):.2e}')
