r"""
A node-local solve that delivers only a chosen relative accuracy, for any problem.

The solve itself is exact and the *returned* value is then perturbed by a random relative error of
size :math:`\eta`, measured in the maximum norm. That makes the solver a black box specified only by
the accuracy it delivers, which is all the SDC iteration sees of it -- how a real solver reaches
:math:`\eta`, whether in a low precision with iterative refinement, with a loose Krylov tolerance or
with a couple of multigrid cycles, is invisible here on purpose.

Both solve routes of the correction-solve contract on :class:`~pySDC.core.problem.Problem` are
spoiled: ``solve_system`` and, where the problem has it, ``solve_system_delta``. ``eta`` can be given
per level, as pySDC spreads lists over the levels, so a fine or a coarse solve can be spoiled alone.
"""

import numpy as np


def inexact(problem_class):
    """
    Subclass ``problem_class`` so that its implicit solves deliver only ``eta`` relative accuracy.

    Parameters
    ----------
    problem_class : type
        Any pySDC problem class whose datatype is backed by a NumPy array.

    Returns
    -------
    type
        The subclass. It takes two more parameters: ``eta`` (float or None, the relative accuracy;
        ``None`` or 0 keeps the exact solve) and ``seed`` (int, for a reproducible perturbation).
    """

    class spoiled(problem_class):
        def __init__(self, eta=None, seed=0, **kwargs):
            super().__init__(**kwargs)
            self._makeAttributeAndRegister('eta', 'seed', localVars=locals())
            self.rng = np.random.default_rng(seed)

        def _spoil(self, value):
            """Perturb ``value`` in place by ``eta`` times its maximum, in a random direction."""
            if not self.eta:
                return value
            x = np.asarray(value).reshape(-1)
            noise = self.rng.uniform(-1.0, 1.0, x.size)
            noise /= max(float(np.max(np.abs(noise))), 1e-300)
            value[:] = (x + self.eta * float(np.max(np.abs(x))) * noise).reshape(np.shape(value))
            return value

        def solve_system(self, *args, **kwargs):
            return self._spoil(super().solve_system(*args, **kwargs))

    if hasattr(problem_class, 'solve_system_delta'):
        spoiled.solve_system_delta = lambda self, *args, **kwargs: self._spoil(
            super(spoiled, self).solve_system_delta(*args, **kwargs)
        )

    spoiled.__name__ = spoiled.__qualname__ = f'{problem_class.__name__}_inexact'
    return spoiled
