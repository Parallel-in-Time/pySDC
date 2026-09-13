r"""
A problem whose node-local solve returns a deliberately inexact correction.

The solve itself is exact and the *returned* correction is then perturbed by a random relative
error of size :math:`\eta`. That makes the solver a black box specified only by the accuracy it
delivers, which is what the SDC iteration actually sees -- how a real solver reaches :math:`\eta`,
whether by a low-precision factorisation with iterative refinement, a loose Krylov tolerance or a
couple of multigrid cycles, is invisible here on purpose.

Used to measure the delivered-accuracy requirement quoted in the README.
"""

import numpy as np

from pySDC.projects.DeltaSDC.problems import heat_delta


class heat_inexact(heat_delta):
    """
    Heat equation whose node-local solve delivers only ``eta`` relative accuracy.

    Parameters
    ----------
    eta : float or None, optional
        Relative accuracy of the returned correction, measured in the infinity norm. ``None`` keeps
        the exact solve.
    seed : int, optional
        Seed for the perturbation, so a run is reproducible.
    **kwargs
        Forwarded to :class:`heat_delta`.
    """

    def __init__(self, eta=None, seed=0, **kwargs):
        """Initialization routine"""
        super().__init__(**kwargs)
        self.eta = eta
        self.rng = np.random.default_rng(seed)

    def solve_system(self, rhs, factor, u0, t):
        """
        Solve exactly, then spoil the answer to the requested relative accuracy.

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side.
        factor : float
            Implicit prefactor.
        u0 : dtype_u
            Initial guess, unused for this direct solve.
        t : float
            Current time.

        Returns
        -------
        dtype_u
            The solution, perturbed by a relative error of size ``eta``.
        """
        me = super().solve_system(rhs, factor, u0, t)
        if not self.eta:
            return me

        x = np.asarray(me).reshape(-1)
        noise = self.rng.uniform(-1.0, 1.0, x.size)
        noise /= max(float(np.max(np.abs(noise))), 1e-300)
        me[:] = (x + self.eta * float(np.max(np.abs(x))) * noise).reshape(self.nvars)
        return me
