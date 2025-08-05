"""
This script introduces ParaDiag for linear problems.
It is recommended to view this code side by side with `Gaya's paper on ParaDiag with collocation methods
<https://arxiv.org/abs/2103.12571>`_ as the code is close to the equations presented there but offers no explanations
about them.
"""

import numpy as np
import scipy.sparse as sp
import sys

from pySDC.implementations.problem_classes.TestEquation_0D import testequation0d as problem_class
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.sweeper_classes.ParaDiagSweepers import QDiagonalization

# setup output
out_file = open('data/step_9_A_out.txt', 'w')


def my_print(*args, **kwargs):
    for output in [sys.stdout, out_file]:
        print(*args, **kwargs, file=output)


# setup parameters
L = 4  # Number of parallel time steps
M = 3  # Number of collocation nodes
N = 2  # Number of spatial degrees of freedom
alpha = 1e-4  # Circular perturbation parameter
restol = 1e-10  # Residual tolerance for the composite collocation problem
dt = 0.1  # step size

my_print(f'Running ParaDiag test script with {L} time steps, {M} collocation nodes and {N} spatial degrees of freedom')

# setup pySDC infrastructure for Dahlquist problem and quadrature
prob = problem_class(lambdas=-1.0 * np.ones(shape=(N)), u0=1.0)
sweeper_params = params = {'num_nodes': M, 'quad_type': 'RADAU-RIGHT'}
sweep = generic_implicit(sweeper_params)

# Setup a global NumPy array and insert initial conditions in the first step
u = np.zeros((L, M, N), dtype=complex)
u[0, :, :] = prob.u_exact(t=0)

# setup matrices for composite collocation problem. We note the sizes of the matrices in comments after generating them.

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

"""
The H matrix computes the solution at the of an individual step from the solutions at the collocation nodes.
For the RADAU-RIGHT rule we use here, the right node coincides with the end of the interval, so this is simple.
We start with building the MxM matrix H_M on the node level and then extend to the spatial dimension with a Kronecker product.
"""
H_M = sp.eye(M).tolil() * 0  # MxM
H_M[:, -1] = 1
H = sp.kron(H_M, I_N)  # MNxMN

"""
Set up collocation problem.
Note that the Kronecker product from Q and A is only possible when there is an A, i.e. when the problem is linear.
We will discuss non-linear problems in later steps in this tutorial
"""
Q = sweep.coll.Qmat[1:, 1:]  # MxM
C_coll = I_MN - dt * sp.kron(Q, prob.A)  # MNxMN

# Set up the composite collocation / all-at-once problem
C = (sp.kron(I_L, C_coll) + sp.kron(E, H)).tocsc()  # LMNxLMN

"""
Now that we have the full composite collocation problem as one large matrix, we can just solve it directly to get a reference solution.
Of course, this is prohibitively expensive for any actual application and we would never want to do this in practice.
"""
sol_direct = sp.linalg.spsolve(C, u.flatten()).reshape(u.shape)

"""
The normal time-stepping approach is to solve the composite collocation problem with forward substitution
"""
sol_stepping = u.copy()
for l in range(L):
    """
    Solve the current step (sol_stepping[l] currently contains the initial conditions at step l)
    Here, we only solve MNxMN systems rather than LMNxLMN systems. This is still really expensive in practice, which is why there is SDC, for example.
    """
    sol_stepping[l, :] = sp.linalg.spsolve(C_coll, sol_stepping[l].flatten()).reshape(sol_stepping[l].shape)

    # place the solution to the current step as the initial conditions to the next step
    if l < L - 1:
        sol_stepping[l + 1, ...] = sol_stepping[l, -1, :]

assert np.allclose(sol_stepping, sol_direct)


"""
So far, so serial and boring. We will now parallelize this using ParaDiag.
We will solve the composite collocation problem using preconditioned Picard iterations:
    C_alpha delta = u_0 - Cu^k = < residual of the composite collocation problem >
    u^{k+1} = u^k + delta
The trick behind ParaDiag is to choose the preconditioner C_alpha to be a time-periodic approximation to C that can be diagonalized and therefore inverted in parallel.
What we change in C_alpha compared to C is the E matrix that propagates the solutions between steps, which we amend to feed the solution to the last step back into the first step.
"""
E_alpha = sp.diags(
    [
        -1.0,
    ]
    * (L - 1),
    offsets=-1,
).tolil()  # LxL
E_alpha[0, -1] = -alpha  # make the problem time-periodic

"""
In order to diagonalize C_alpha, on the step level, we need to diagonalize I_L and E_alpha simultaneously.
I_L and E_alpha are alpha-circular matrices which can be simultaneously diagonalized by a weighted Fourier transform.
We start by setting the weighting matrices for the Fourier transforms and then compute the diagonal entries of the diagonal version D_alpha of E_alpha.
We refrain from actually setting up the preconditioner because we will not use the expanded version here.
"""
gamma = alpha ** (-np.arange(L) / L)
J = sp.diags(gamma)  # LxL
J_inv = sp.diags(1 / gamma)  # LxL

# compute diagonal entries via Fourier transform
D_alpha_diag_vals = np.fft.fft(1 / gamma * E_alpha[:, 0].toarray().flatten(), norm='backward')


"""
We need some convenience functions for computing matrix vector multiplication and the composite collocation problem residual here
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
            # res[l, m, ...] -= (_u[l] - dt * Qf)[-1]
            res[l, m, ...] -= (_u[l] - dt * Qf)[m]
            # res[l, m, ...] -= np.mean((_u[l] - dt * Qf), axis=0)

    return res


"""
We will start with ParaDiag where we parallelize across the L steps but solve the collocation problems directly in serial.
"""
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
my_print(
    f'Needed {niter_ParaDiag_L} iterations in parallel across the steps ParaDiag. Stopped at residual {np.linalg.norm(res):.2e}'
)
assert np.allclose(sol_ParaDiag_L, sol_direct)

"""
While we have distributed the work across L tasks, we are still solving perturbed collocation problems directly on a single task here.
This is very expensive, and we will now additionally diagonalize the quadrature matrix Q in order to distribute the work on LM tasks, where we solve NxN systems each.
We rearrange the contribution of E_alpha to arrive at a problem (I - dtQG^{-1}A)u = u0.
After diagonalizing QG^{-1}, we can simply utilize the Euler solves that are implemented in pySDC, but need to keep in mind that complex valued "step sizes" are required.

We start by setting up the G and G^{-1} matrices. Then we will setup pySDC sweepers that solve QG^{-1} with diagonalization.
Here, we will not use the sweepers, but just the diagonalization computed there in order to make more clear what is going on.
"""
G = [(D_alpha_diag_vals[l] * H_M + I_M).tocsc() for l in range(L)]  # MxM
G_inv = [sp.linalg.inv(_G).toarray() for _G in G]  # MxM
sweepers = [QDiagonalization(params={**sweeper_params, 'G_inv': _G_inv}) for _G_inv in G_inv]


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
my_print(
    f'Needed {niter} iterations in parallel and local paradiag with increment formulation, stopped at residual {np.linalg.norm(res):.2e}'
)
assert np.allclose(sol_ParaDiag, sol_direct)
assert np.allclose(niter, niter_ParaDiag_L)
