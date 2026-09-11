import numpy as np


def state_to_numpy(state):
    """Convert a pySDC datatype to a flat real-valued NumPy vector."""
    arr = np.asarray(state)
    if np.iscomplexobj(arr):
        # First prototype focuses on real-valued problems.
        arr = arr.real
    return arr.astype(np.float64, copy=False).reshape(-1)


def extract_problem_params(prob, dt=None):
    """Extract a compact fixed-size parameter vector for known prototype problems.

    When *dt* is provided for a Dahlquist-type problem the returned value is
    z = lambda * dt (the stability parameter) instead of bare lambda.
    """
    if hasattr(prob, 'lambdas'):
        lambdas = np.asarray(prob.lambdas).reshape(-1)
        lam = float(np.real(lambdas[0]))
        if dt is not None:
            return np.array([lam * float(dt)], dtype=np.float64)
        return np.array([lam], dtype=np.float64)

    if hasattr(prob, 'nu') and hasattr(prob, 'freq') and hasattr(prob, 'nvars'):
        freq = prob.freq[0] if isinstance(prob.freq, tuple) else prob.freq
        nvars = prob.nvars[0] if isinstance(prob.nvars, tuple) else prob.nvars
        return np.array([float(prob.nu), float(freq), float(nvars)], dtype=np.float64)

    return np.zeros((1,), dtype=np.float64)


def stack_nodes(nodes):
    """Stack node values to shape (num_nodes, state_dim)."""
    return np.stack([state_to_numpy(me) for me in nodes], axis=0)


def compute_residual_vectors(level, u_nodes=None, f_nodes=None):
    """Compute collocation residual vectors and max residual norm for a level state."""
    L = level
    P = L.prob
    coll = L.sweep.coll
    M = coll.num_nodes

    if u_nodes is None:
        u_nodes = [L.u[m + 1] for m in range(M)]
    if f_nodes is None:
        f_nodes = [L.f[m + 1] for m in range(M)]

    residuals = []
    for m in range(M):
        res = P.dtype_u(P.init, val=0.0)
        for j in range(M):
            res += L.dt * coll.Qmat[m + 1, j + 1] * f_nodes[j]
        res += L.u[0] - u_nodes[m]
        if L.tau[m] is not None:
            res += L.tau[m]
        residuals.append(res)

    residual_norm = max(abs(me) for me in residuals)
    residual_array = stack_nodes(residuals)
    return residual_array, float(residual_norm)

