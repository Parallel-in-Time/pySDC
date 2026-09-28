# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Part A: ParaDiag for linear problems
#
# We start with the linear case, where the composite collocation problem really can be written as a matrix and the
# whole method is a few lines of linear algebra. It is recommended to view this code side by side with
# [Gaya's paper on ParaDiag with collocation methods](https://arxiv.org/abs/2103.12571), as the code follows the
# equations there closely without repeating their explanation.
#
# The test problem is the Dahlquist equation, on $N = 2$ unknowns, over $L = 4$ time steps with $M = 3$ collocation
# nodes each.

# %%
import numpy as np
import scipy.sparse as sp

from pySDC.implementations.problem_classes.TestEquation_0D import testequation0d as problem_class
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.sweeper_classes.ParaDiagSweepers import QDiagonalization

# setup parameters
L = 4  # Number of parallel time steps
M = 3  # Number of collocation nodes
N = 2  # Number of spatial degrees of freedom
alpha = 1e-4  # Circular perturbation parameter
restol = 1e-10  # Residual tolerance for the composite collocation problem
dt = 0.1  # step size

print(f'Running ParaDiag test script with {L} time steps, {M} collocation nodes and {N} spatial degrees of freedom')

# setup pySDC infrastructure for Dahlquist problem and quadrature
prob = problem_class(lambdas=-1.0 * np.ones(shape=(N)), u0=1.0)
sweeper_params = params = {'num_nodes': M, 'quad_type': 'RADAU-RIGHT'}
sweep = generic_implicit(sweeper_params, None)

# Setup a global NumPy array and insert initial conditions in the first step
u = np.zeros((L, M, N), dtype=complex)
u[0, :, :] = prob.u_exact(t=0)

# %% [markdown]
# ## The composite collocation problem
#
# We set up the matrices of the composite collocation problem, and note their sizes in the comments. The matrix $E$
# propagates the solution of each step to be the initial condition of the next.

# %%
# Start with identity matrices (I) of various sizes
I_L = sp.eye(L)  # LxL
I_MN = sp.eye((M) * N)  # MNxMN
I_N = sp.eye(N)  # NxN
I_M = sp.eye(M)  # MxM

# E matrix propagates the solution of the steps to be the initial condition for the next step
E = sp.diags(
    [
        -1.0,
    ]
    * (L - 1),
    offsets=-1,
)  # LxL

# %% [markdown]
# The $H$ matrix computes the solution at the end of an individual step from the solutions at the collocation nodes.
# For the RADAU-RIGHT rule we use here, the right node coincides with the end of the interval, so this is simple. We
# start with the $M \times M$ matrix $H_M$ on the node level, and extend it to the spatial dimension with a Kronecker
# product.

# %%
H_M = sp.eye(M).tolil() * 0  # MxM
H_M[:, -1] = 1
H = sp.kron(H_M, I_N)  # MNxMN

# %% [markdown]
# Now the collocation problem. Note that the Kronecker product of $Q$ and $A$ is only possible when there is an $A$,
# i.e. when the problem is linear; nonlinear problems follow in [Part B](B_paradiag_for_nonlinear_problems).

# %%
Q = sweep.coll.Qmat[1:, 1:]  # MxM
C_coll = I_MN - dt * sp.kron(Q, prob.A)  # MNxMN

# Set up the composite collocation / all-at-once problem
C = (sp.kron(I_L, C_coll) + sp.kron(E, H)).tocsc()  # LMNxLMN

# %% [markdown]
# With the full composite collocation problem as one large matrix, we can solve it directly for a reference solution.
# Of course, this is prohibitively expensive for any actual application, and we would never do this in practice.

# %%
sol_direct = sp.linalg.spsolve(C, u.flatten()).reshape(u.shape)

# %% [markdown]
# The normal time-stepping approach solves the composite collocation problem by forward substitution, one step after
# the other. This only solves $MN \times MN$ systems rather than $LMN \times LMN$ ones, but that is still really
# expensive in practice, which is why there is SDC, for example.

# %%
sol_stepping = u.copy()
for l in range(L):
    # solve the current step (sol_stepping[l] currently contains the initial conditions at step l)
    sol_stepping[l, :] = sp.linalg.spsolve(C_coll, sol_stepping[l].flatten()).reshape(sol_stepping[l].shape)

    # place the solution to the current step as the initial conditions to the next step
    if l < L - 1:
        sol_stepping[l + 1, ...] = sol_stepping[l, -1, :]

assert np.allclose(sol_stepping, sol_direct)

# %% [markdown]
# ## ParaDiag
#
# So far, so serial and boring. We now parallelize this with ParaDiag. We solve the composite collocation problem
# with preconditioned Picard iterations,
#
# $$
# C_\alpha \delta = u_0 - C u^k, \qquad u^{k+1} = u^k + \delta ,
# $$
#
# where the right-hand side is the residual of the composite collocation problem. The trick behind ParaDiag is to
# choose the preconditioner $C_\alpha$ as a time-periodic approximation to $C$ that can be diagonalized, and
# therefore inverted in parallel. What changes in $C_\alpha$ compared to $C$ is the matrix $E$, which now also feeds
# the solution of the last step back into the first one, scaled by $\alpha$:

# %%
E_alpha = sp.diags(
    [
        -1.0,
    ]
    * (L - 1),
    offsets=-1,
).tolil()  # LxL
E_alpha[0, -1] = -alpha  # make the problem time-periodic

# %% [markdown]
# To diagonalize $C_\alpha$ on the step level, we need to diagonalize $I_L$ and $E_\alpha$ simultaneously. Both are
# $\alpha$-circulant matrices, which can be diagonalized simultaneously by a weighted Fourier transform. We set up
# the weighting matrices of the Fourier transforms, and compute the diagonal entries of the diagonal version
# $D_\alpha$ of $E_\alpha$. We do not set up the preconditioner itself, because we will not use the expanded version.

# %%
gamma = alpha ** (-np.arange(L) / L)
J = sp.diags(gamma)  # LxL
J_inv = sp.diags(1 / gamma)  # LxL

# compute diagonal entries via Fourier transform
D_alpha_diag_vals = np.fft.fft(1 / gamma * E_alpha[:, 0].toarray().flatten(), norm='backward')

# %% [markdown]
# Two convenience functions, for matrix-vector products on the steps and for the residual of the composite
# collocation problem:


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
    res = np.zeros_like(vec).astype(complex)
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
        for m in range(M):
            res[l, m, ...] -= (_u[l] - dt * Qf)[m]

    return res


# %% [markdown]
# ### Parallel across the steps
#
# First, we parallelize across the $L$ steps, but solve the collocation problems directly and in serial: a weighted
# FFT in time, $L$ independent solves, the inverse transform.

# %%
sol_ParaDiag_L = u.copy()
u0 = u.copy()
niter_ParaDiag_L = 0

res = residual(sol_ParaDiag_L, u0)
while np.linalg.norm(res) > restol:
    # compute weighted FFT in time to go to diagonal base of C_alpha
    x = np.fft.fft(
        mat_vec(J_inv.tolil(), res),
        axis=0,
        norm='ortho',
    )

    # solve the collocation problems in parallel on the steps
    y = np.empty_like(x)
    for l in range(L):
        # construct local matrix of "collocation problem"
        local_matrix = (D_alpha_diag_vals[l] * H + C_coll).tocsc()

        # solve local "collocation problem" directly
        y[l, ...] = sp.linalg.spsolve(local_matrix, x[l, ...].flatten()).reshape(x[l, ...].shape)

    # compute inverse weighted FFT in time to go back from diagonal base of C_alpha
    sol_ParaDiag_L += mat_vec(J.tolil(), np.fft.ifft(y, axis=0, norm='ortho'))

    # update residual
    res = residual(sol_ParaDiag_L, u0)
    niter_ParaDiag_L += 1
print(
    f'Needed {niter_ParaDiag_L} iterations in parallel across the steps ParaDiag. Stopped at residual {np.linalg.norm(res):.2e}'
)
assert np.allclose(sol_ParaDiag_L, sol_direct)

# %% [markdown]
# ### Parallel across the steps and the nodes
#
# The work is distributed across $L$ tasks now, but each of them still solves a perturbed collocation problem
# directly, which is very expensive. So we additionally diagonalize the quadrature matrix $Q$, to distribute the work
# on $LM$ tasks that solve $N \times N$ systems each. We rearrange the contribution of $E_\alpha$ to arrive at a
# problem $(I - \Delta t\, Q G^{-1} A) u = u_0$. After diagonalizing $Q G^{-1}$, we can simply use the implicit Euler
# solves that pySDC problems implement, keeping in mind that they need complex "step sizes".
#
# We set up $G$ and $G^{-1}$, and pySDC sweepers that diagonalize $Q G^{-1}$. We do not use the sweepers themselves
# here, only the diagonalization they compute, to make clearer what is going on.

# %%
G = [(D_alpha_diag_vals[l] * H_M + I_M).tocsc() for l in range(L)]  # MxM
G_inv = [sp.linalg.inv(_G).toarray() for _G in G]  # MxM
sweepers = [QDiagonalization(params={**sweeper_params, 'G_inv': _G_inv}, level=None) for _G_inv in G_inv]


sol_ParaDiag = u.copy().astype(complex)
res = residual(sol_ParaDiag, u0)
niter = 0
while np.max(np.abs(residual(sol_ParaDiag, u0))) > restol:

    # weighted FFT in time
    x = np.fft.fft(
        mat_vec(J_inv.tolil(), res),
        axis=0,
        norm='ortho',
    )

    # perform local solves of "collocation problems" on the steps in parallel
    y = np.empty_like(x)
    for l in range(L):

        # diagonalize QG^-1 matrix
        w, S, S_inv = sweepers[l].w, sweepers[l].S, sweepers[l].S_inv

        # perform local solves on the collocation nodes in parallel
        x1 = S_inv @ x[l]
        x2 = np.empty_like(x1)
        for m in range(M):
            x2[m, :] = prob.solve_system(rhs=x1[m], factor=w[m] * dt, u0=x1[m], t=0)
        z = S @ x2
        y[l, ...] = G_inv[l] @ z

    # inverse weighted FFT in time
    sol_ParaDiag += mat_vec(J.tolil(), np.fft.ifft(y, axis=0, norm='ortho'))

    res = residual(sol_ParaDiag, u0)
    niter += 1
print(
    f'Needed {niter} iterations in parallel and local paradiag with increment formulation, stopped at residual {np.linalg.norm(res):.2e}'
)

# %% [markdown]
# :::{admonition} Important things to note
# - The diagonalization happens across the time steps, not across the collocation nodes; the second variant adds a
#   diagonalization across the nodes on top.
# - The $\alpha$-circulant approximation is what makes the diagonalization possible in the first place.
# :::
#
# The checks the tests run: both variants arrive at the direct solution, in the same number of iterations.

# %%
assert np.allclose(sol_ParaDiag, sol_direct)
assert np.allclose(niter, niter_ParaDiag_L)
