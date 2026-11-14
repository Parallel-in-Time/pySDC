r"""
The classical schemes the paper compares SDC-SBB with, for all samples of an
:class:`~pySDC.projects.SDC4SDEs.problem_classes.SDEProblem` at once.

Both use the increments :math:`\Delta W_r` of the Brownian paths over each step. Milstein is written for commutative
noise, :math:`(b_r \cdot \nabla) b_s = (b_s \cdot \nabla) b_r`, where it needs no iterated integrals.
"""

import numpy as np


def euler_maruyama(P, dt, Tend):
    r"""
    Euler-Maruyama for the Itô form, :math:`X_{n+1} = X_n + a(t_n, X_n) \Delta t + \sum_r b_r(t_n, X_n) \Delta W_r`.

    Args:
        P (SDEProblem): the problem, with its Brownian paths
        dt (float): step size
        Tend (float): end time

    Returns:
        numpy.ndarray: solution at Tend for all samples, shape (n_samples, ndim)
    """
    u, n = P.u_exact(0.0), round(Tend / dt)
    for i in range(n):
        t = i * dt
        P.work_counters['diffusion']()
        dW = P.W(t + dt) - P.W(t)
        u = u + dt * P.ito_drift(u, t) + (dW[..., None] * P.diffusion(u, t)).sum(axis=0)
    return u


def milstein(P, dt, Tend):
    r"""
    Milstein for commutative noise, in Stratonovich form,
    :math:`X_{n+1} = X_n + \underline{a} \Delta t + \sum_r b_r \Delta W_r
    + \frac{1}{2} \sum_{r,s} (b_s \cdot \nabla) b_r \Delta W_r \Delta W_s`.

    Args:
        P (SDEProblem): the problem, with its Brownian paths
        dt (float): step size
        Tend (float): end time

    Returns:
        numpy.ndarray: solution at Tend for all samples, shape (n_samples, ndim)
    """
    u, n = P.u_exact(0.0), round(Tend / dt)
    for i in range(n):
        t = i * dt
        for key in ['drift', 'diffusion', 'diffusion_derivative']:
            P.work_counters[key]()
        dW = P.W(t + dt) - P.W(t)
        u = (
            u
            + dt * P.drift(u, t)
            + (dW[..., None] * P.diffusion(u, t)).sum(axis=0)
            + 0.5 * (dW[:, None, :, None] * dW[None, :, :, None] * P.diffusion_derivative(u, t)).sum(axis=(0, 1))
        )
    return u
