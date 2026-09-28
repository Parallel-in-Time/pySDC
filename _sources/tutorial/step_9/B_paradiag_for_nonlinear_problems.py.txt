# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Part B: ParaDiag for nonlinear problems
#
# ParaDiag works by diagonalizing the "top layer" of Kronecker products that make up the circularized composite
# collocation problem. For nonlinear problems, though, the composite collocation problem cannot be written as a
# matrix, so the diagonalization needs a linear operator to work with. There are two ways out:
#
# - **IMEX splitting**, where only the linear part is treated implicitly. The ParaDiag preconditioner is then made up
#   of the linear implicit part only, which can be diagonalized just like for linear problems, and the nonlinear part
#   comes in through the residual on the right-hand side.
# - **Averaging the Jacobian**. Nonlinear problems are solved with Newton, where the Jacobian is computed from the
#   current solution and inverted in each Newton iteration. To write the ParaDiag preconditioner with Kronecker
#   products and only diagonalize the outermost part, all steps need the same Jacobian: the average.
#
# As IMEX ParaDiag is a trivial extension of ParaDiag for linear problems, we focus on the second approach here, with
# the van der Pol oscillator as an example. The ParaDiag iteration then proceeds as follows:
#
# 1. compute the residual of the composite collocation problem,
# 2. average the solution across the steps and nodes, to compute the average Jacobian,
# 3. weighted FFT in time to diagonalize $E_\alpha$,
# 4. solve for the increment on the subproblems on the steps and nodes, inverting the averaged Jacobian from (2),
# 5. weighted inverse FFT in time,
# 6. increment the solution.

# %%
import numpy as np
import scipy.sparse as sp

from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit as sweeper_class
from pySDC.implementations.problem_classes.Van_der_Pol_implicit import vanderpol

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

print(f'Running ParaDiag test script for van der Pol with mu={prob.mu} and {L} time steps and {M} collocation nodes.')

# %% [markdown]
# ## The matrices
#
# Those that make up the composite collocation problem, as in [Part A](A_paradiag_for_linear_problems), although we
# do not set up the full problem here. See [the paper](https://arxiv.org/abs/2103.12571) for their meaning. We
# diagonalize $Q G^{-1}$ on every step right away.

# %%
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

# %% [markdown]
# The matrix-vector product on the steps and the residual of the composite collocation problem, as in Part A:


# %%
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


# %% [markdown]
# ## The iteration
#
# The six steps from above. Each ParaDiag iteration does a single Newton iteration on every node, one call of
# `solve_jacobian` per node, so the number of Newton iterations per node equals the number of ParaDiag iterations.

# %%
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
print(f'Needed {niter} ParaDiag iterations, stopped at residual {np.max(np.abs(res)):.2e}')

# %% [markdown]
# :::{admonition} Important things to note
# - Averaging the Jacobian requires communicating the average solution, which is why `average_jacobian` is off by
#   default for linear problems.
# - We do a single Newton iteration per ParaDiag iteration, so the number of Newton iterations per node equals the
#   number of ParaDiag iterations.
# :::
#
# The check the tests run is the one inside the loop: ParaDiag has to converge in fewer than 99 iterations.
