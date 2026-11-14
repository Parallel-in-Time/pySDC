r"""
Brownian paths and their smooth Brownian bridge (SBB) approximation.

On a time step :math:`[t_0, t_0 + \Delta T]`, a Brownian path :math:`W` is written as the line through its end points
plus a Brownian bridge, and the bridge is expanded in a sine series (Karhunen-Loève expansion),

.. math::

    W(t_0 + s) = W(t_0) + \frac{s}{\Delta T} \Delta W
        + \frac{\sqrt{2 \Delta T}}{\pi} \sum_{j=1}^{\infty} \frac{\sin(j \pi s / \Delta T)}{j} \xi_j,
    \qquad \xi_j = \sqrt{\frac{2}{\Delta T}} \int_0^{\Delta T} \cos(j \pi s / \Delta T)\, dW(t_0 + s),

with independent standard normal :math:`\xi_j`. Truncating the sum after :math:`J` terms gives a smooth function
:math:`B^{(J)}` which agrees with :math:`W` at both ends of the step, and whose derivative

.. math::

    \frac{dB^{(J)}}{dt}(t_0 + s) = \frac{\eta_0}{\sqrt{\Delta T}}
        + \sqrt{\frac{2}{\Delta T}} \sum_{j=1}^{J} \cos(j \pi s / \Delta T)\, \xi_j,
    \qquad \eta_0 = \frac{\Delta W}{\sqrt{\Delta T}},

replaces the white noise :math:`dW/dt` in the SDE, see Binkowski, Christian, Götschel, Minion: "Toward Spectral
Deferred Correction Methods for Stochastic Differential Equations" (arXiv:2610.23693).

Paths are stored with shape ``(n_noise, n_samples, n_fine + 1)``: one row per independent Wiener process and per
sample, sampled on a uniform grid of ``n_fine`` steps.
"""

import numpy as np


def brownian_paths(n_samples, n_fine, T=1.0, n_noise=1, seed=None):
    """
    Sample Brownian paths on a uniform grid of [0, T].

    Args:
        n_samples (int): number of independent paths (Monte Carlo samples)
        n_fine (int): number of steps of the grid
        T (float): end time
        n_noise (int): number of independent Wiener processes per sample
        seed: seed of the random number generator

    Returns:
        numpy.ndarray: paths of shape (n_noise, n_samples, n_fine + 1), starting at zero
    """
    rng = np.random.default_rng(seed)
    W = np.zeros((n_noise, n_samples, n_fine + 1))
    W[..., 1:] = np.cumsum(rng.standard_normal((n_noise, n_samples, n_fine)) * np.sqrt(T / n_fine), axis=-1)
    return W


def bridge_coefficients(W, dt, n_terms):
    r"""
    Coefficients :math:`\eta_0` and :math:`\xi_1, \dots, \xi_J` of the smooth Brownian bridge on one time step.

    The integrals :math:`\int \cos(j \pi s / \Delta T)\, dW` are computed exactly for the piecewise linear
    interpolant of the samples, so that :math:`B^{(J)}` converges to that interpolant as :math:`J` grows.

    Args:
        W (numpy.ndarray): path on the step, sampled at ``n + 1`` uniformly spaced points along the last axis
        dt (float): length :math:`\Delta T` of the step
        n_terms (int): number J of terms of the sine series

    Returns:
        eta0 (numpy.ndarray): :math:`\Delta W / \sqrt{\Delta T}`, with the shape of ``W`` without its last axis
        xi (numpy.ndarray): :math:`\xi_1, \dots, \xi_J` along a new last axis
    """
    n = W.shape[-1] - 1
    eta0 = (W[..., -1] - W[..., 0]) / np.sqrt(dt)
    j = np.arange(1, n_terms + 1)
    s = np.arange(n + 1) / n
    # integral of cos(j pi s / dt) over each fine step of length dt / n, divided by that length
    weights = np.diff(np.sin(np.pi * np.outer(s, j)), axis=0) * n / (np.pi * j)
    return eta0, np.sqrt(2.0 / dt) * np.diff(W, axis=-1) @ weights


def bridge(eta0, xi, dt, s):
    r"""
    Evaluate :math:`B^{(J)}(t_0 + s) - W(t_0)` for the coefficients of :func:`bridge_coefficients`.

    Args:
        eta0 (numpy.ndarray): :math:`\eta_0`
        xi (numpy.ndarray): :math:`\xi_1, \dots, \xi_J` along the last axis
        dt (float): length of the step
        s (numpy.ndarray): times since the start of the step

    Returns:
        numpy.ndarray: values with the shape of ``eta0`` followed by the shape of ``s``
    """
    s = np.asarray(s) / dt
    j = np.arange(1, xi.shape[-1] + 1)
    sines = np.sin(np.pi * np.multiply.outer(j, s)) / (np.pi * j).reshape((-1,) + (1,) * s.ndim)
    return np.sqrt(dt) * (np.multiply.outer(eta0, s) + np.sqrt(2.0) * np.tensordot(xi, sines, axes=1))


def smooth_paths(W, T, n_modes):
    r"""
    Replace paths by their truncated Karhunen-Loève expansion on all of [0, T], with ``n_modes`` sine terms.

    The result is smooth, so that the smooth Brownian bridge approximation on each time step converges fast. This
    separates the error of the SDC iteration from the error of approximating a rough path by a smooth one.

    Args:
        W (numpy.ndarray): paths starting at zero, sampled on a uniform grid of [0, T] along the last axis
        T (float): end time
        n_modes (int): number of sine terms

    Returns:
        numpy.ndarray: smooth paths, sampled on the same grid
    """
    eta0, xi = bridge_coefficients(W, T, n_modes)
    return bridge(eta0, xi, T, np.linspace(0, T, W.shape[-1]))
