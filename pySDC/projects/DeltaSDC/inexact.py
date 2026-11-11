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
``calls`` restricts the spoiling to chosen solve calls, counted from 0 -- with one call per node and
sweep, sweep :math:`j` of an :math:`M`-node run is calls :math:`(j-1)M` to :math:`jM - 1` -- which is
how a perturbation of a single sweep is measured. ``schedule`` instead sets the accuracy per
iteration: a noise level :math:`\eta` as above, or the name of a format the result is rounded to,
relative to its maximum (``'fp64'``, ``'fp32'``, ``'fp16'``).
"""

import numpy as np

FORMATS = {'fp64': 2.0**-53, 'fp32': 2.0**-24, 'fp16': 2.0**-11}
"""Unit roundoff of the formats a result can be rounded to."""


def round_to(x, fmt):
    """Round the float64 array ``x`` to the format ``fmt``, relative to its maximum."""
    scale = max(float(np.max(np.abs(x))), 1e-300)
    y = x / scale
    if fmt == 'fp32':
        y = y.astype(np.float32)
    elif fmt == 'fp16':
        y = y.astype(np.float16)
    elif fmt != 'fp64':
        raise ValueError(f'unknown format {fmt}')
    return y.astype(np.float64) * scale


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
        The subclass. It takes more parameters: ``eta`` (float or None, the relative accuracy;
        ``None`` or 0 keeps the exact solve), ``seed`` (int, for a reproducible perturbation),
        ``calls`` (a collection of call indices to spoil; ``None``, the default, spoils every call),
        and ``schedule`` with ``nodes``: a tuple with one noise level or format name per iteration,
        the last one repeated, overriding ``eta``; ``nodes`` is the number of solves per iteration.
    """

    class spoiled(problem_class):
        def __init__(self, eta=None, seed=0, calls=None, schedule=None, nodes=None, **kwargs):
            super().__init__(**kwargs)
            self._makeAttributeAndRegister('eta', 'seed', 'calls', 'schedule', 'nodes', localVars=locals())
            self.rng = np.random.default_rng(seed)
            self._call = 0

        def _spoil(self, value):
            """Perturb ``value`` in place by ``eta`` times its maximum, in a random direction."""
            call, self._call = self._call, self._call + 1
            eta = self.eta if self.schedule is None else self.schedule[min(call // self.nodes, len(self.schedule) - 1)]
            if not eta or (self.calls is not None and call not in self.calls):
                return value
            x = np.asarray(value).reshape(-1)
            if isinstance(eta, str):
                value[:] = round_to(x, eta).reshape(np.shape(value))
                return value
            noise = self.rng.uniform(-1.0, 1.0, x.size)
            noise /= max(float(np.max(np.abs(noise))), 1e-300)
            value[:] = (x + eta * float(np.max(np.abs(x))) * noise).reshape(np.shape(value))
            return value

        def solve_system(self, *args, **kwargs):
            return self._spoil(super().solve_system(*args, **kwargs))

    if hasattr(problem_class, 'solve_system_delta'):
        spoiled.solve_system_delta = lambda self, *args, **kwargs: self._spoil(
            super(spoiled, self).solve_system_delta(*args, **kwargs)
        )

    spoiled.__name__ = spoiled.__qualname__ = f'{problem_class.__name__}_inexact'
    return spoiled
