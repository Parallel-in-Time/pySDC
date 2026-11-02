r"""
Storage precision raised as the iteration converges.

The fine level's state is the one place precision cannot simply be reduced -- the residual is formed
there by cancelling :math:`\mathcal{O}(1)` quantities. But that is a statement about the *end* of the
run: early on the residual is nowhere near :math:`\varepsilon|u|`, and the rounding a low format
introduces is a perturbed iterate, which the remaining sweeps contract away like any other.

So the state walks up a ladder, ``float16 -> float32 -> float64``, stepping when the correction the
sweep just applied would be lost in the rounding the current format costs:

.. math::
    \text{use the lowest format with}\quad |\delta| > \texttt{safety}\cdot\varepsilon\,|u| .

Both norms are already in hand, so the delta form supplies its own indicator -- no operator norm, no
:math:`\Delta t`, nothing to calibrate per problem. The ladder is walked **monotonically**, so a run
pays at most ``len(ladder) - 1`` conversions however many iterations it takes.

See the project README for the measurements, for why stagnation detection is not used instead, and
for the two things this requires of a run.
"""

import numpy as np

from pySDC.projects.DeltaSDC.mlsdc import delta_implicit_rounded, round_value

SAFETY = 100.0
"""How far the correction must stay above the format's representation error before stepping up."""


class PrecisionCascade:
    """
    Mixin raising a level's storage precision as its corrections shrink.

    Mix into any delta-form sweeper. Reads two sweeper parameters:

    ``state_cascade``
        Applies to the **finest level only**; see :meth:`update_nodes`. Formats from lowest to
        highest as a **tuple**, e.g. ``('float16', 'float32', None)``, where ``None`` means the
        backend's own precision and should be the last entry, since the finest level has to end
        where the residual is formed. A tuple rather than a list: pySDC spreads a list-valued
        parameter one entry per level, which would hand each level a single format name instead of
        a ladder, so a list is refused rather than silently misread.
    ``cascade_safety``
        Multiple of the format's representation error the correction must stay above. Defaults to
        :data:`SAFETY`.

    Attributes
    ----------
    cascade_index : int
        Position in the ladder, which only ever increases.
    cascade_history : list
        The format in force after each sweep, for reporting.
    """

    cascade_index = 0
    cascade_history = None

    def _delta_setup(self):
        """Arm the per-sweep record of corrections the indicator reads."""
        super()._delta_setup()
        self._cascade_deltas = []

    def _solve_correction(self, *args, **kwargs):
        """Record the correction, which is the numerator of the indicator."""
        delta = super()._solve_correction(*args, **kwargs)
        self._cascade_deltas.append(delta)
        return delta

    def ladder(self):
        """
        The requested ladder of formats.

        Returns
        -------
        tuple or None
            ``None`` if no cascade was requested.

        Raises
        ------
        ValueError
            If a single format name arrives instead of a ladder, which is what a list turns into:
            pySDC spreads a list-valued parameter one entry per level, before the sweeper ever sees
            it, so each level receives one format name. Indexing into that would pick out characters
            of the name rather than formats -- ``'float16'[1]`` is ``'l'``, which numpy reads as
            int64 -- so it is refused instead.
        """
        ladder = getattr(self.params, 'state_cascade', None)
        if isinstance(ladder, str):
            raise ValueError(
                f'state_cascade must be a tuple of formats, got the single name {ladder!r}. A list '
                f'does not survive: pySDC spreads it one entry per level, taking the ladder apart '
                f'before the sweeper sees it. Pass a tuple.'
            )
        return ladder or None

    def current_format(self):
        """
        The format in force.

        Returns
        -------
        numpy.dtype or None
            ``None`` once the ladder has reached backend precision, or if no ladder was requested.
        """
        ladder = self.ladder()
        if ladder is None:
            return None
        token = ladder[min(self.cascade_index, len(ladder) - 1)]
        return None if token is None else np.dtype(token)

    def cascade_norms(self):
        r"""
        The two norms the indicator compares, :math:`|\delta|` and :math:`|u|`.

        Split out so a node-parallel sweeper can reduce them across ranks: every rank has to reach
        the same verdict, or the levels would end up stored at different precisions.

        Returns
        -------
        tuple
            ``(|delta|, |u|)`` in the infinity norm.
        """
        delta = max((abs(d) for d in self._cascade_deltas), default=0.0)
        state = max((abs(u) for u in self.level.u if u is not None), default=0.0)
        return delta, state

    def advance_cascade(self):
        r"""
        Step up the ladder while the correction no longer clears the current format.

        Compares :math:`|\delta|` against :math:`\texttt{safety}\,\varepsilon|u|` and steps as far as
        needed, so a ladder entry that was never usable is skipped rather than costing a sweep.

        Returns
        -------
        numpy.dtype or None
            The format in force after stepping.
        """
        ladder = self.ladder()
        if ladder is None:
            return None

        safety = float(getattr(self.params, 'cascade_safety', SAFETY))
        delta, state = self.cascade_norms()

        while self.cascade_index < len(ladder) - 1:
            token = ladder[self.cascade_index]
            if token is None:
                break
            if delta > safety * float(np.finfo(np.dtype(token)).eps) * state:
                break
            self.cascade_index += 1
        return self.current_format()

    def update_nodes(self):
        """
        Sweep, then step the ladder and store the level at whatever format it now calls for.

        Returns
        -------
        None
        """
        super().update_nodes()
        if self.ladder() is None or self.level.level_index > 0:
            # The cascade is for the finest level only. That is the one level whose requirement is
            # absolute -- eps*|u| has to end up below the residual being resolved -- so it is the one
            # with somewhere to climb to. A coarse level is a preconditioner and its requirement is
            # relative, so it belongs at a fixed low precision (`level_precision`) rather than on a
            # ladder. Letting the cascade run there would raise the coarse level back to backend
            # precision after a few sweeps and quietly undo the setting.
            return None

        dtype = self.advance_cascade()
        if self.cascade_history is None:
            self.cascade_history = []
        self.cascade_history.append(None if dtype is None else dtype.name)
        if dtype is not None:
            # index 0 is the step's initial value and the right-hand side at it: input data, not an
            # iterate, so no later sweep corrects it and raising the precision afterwards does not
            # undo a rounding. The cascade may only lower the precision of what it later corrects.
            for name in ('u', 'f'):
                for value in getattr(self.level, name)[1:]:
                    round_value(value, dtype)
        return None


class delta_implicit_cascade(PrecisionCascade, delta_implicit_rounded):
    """Delta-form implicit sweeper whose fine-level state walks up a ladder of formats."""
