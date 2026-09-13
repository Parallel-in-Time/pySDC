r"""
Reduced precision on a level, emulated, on top of pySDC's delta-form hierarchy.

The hierarchy itself is ordinary pySDC -- the delta-form sweepers and ``delta_transfer`` live in
:mod:`pySDC.implementations.sweeper_classes.delta_form` and
:mod:`pySDC.implementations.transfer_classes.BaseTransferDelta`, and are useful without any of this.
What is here is the instrument for measuring what the rewrite buys, and the controls that say it
buys anything at all.

``level_precision``
    Sweeper parameter, per level as usual. Rounds everything the level stores -- ``u``, ``f``,
    ``uold``, ``fold`` and ``tau`` -- through the given precision. The *storage* is then genuinely at
    that precision while the arithmetic stays at the backend type, so this is optimistic about
    iteration counts and saves no memory traffic. Results obtained with it should be labelled as
    emulated; ``dtype`` on the finite-difference problems is the real thing.

:class:`delta_implicit_rounded` is the sweeper for both; what makes an experiment a measurement or
a **control** is the transfer it is paired with. :class:`delta_transfer` hands the coarse level a
restricted residual, :class:`rounding_transfer` is stock MLSDC at the same reduced precision --
which is what the delta hierarchy is claimed to fix.

A problem on a reduced-precision level must provide ``eval_f_increment``. Without it the sweeper
forms :math:`\Delta f` by subtracting two stored right-hand sides, whose cancellation error carries
the operator norm and binds long before anything else does.
"""

import numpy as np

from pySDC.core.base_transfer import BaseTransfer
from pySDC.implementations.sweeper_classes.delta_form import (
    delta_imex_1st_order,
    delta_implicit,
    total_increment,  # noqa: F401  -- re-exported for the node-parallel module
)
from pySDC.implementations.transfer_classes.BaseTransferDelta import delta_transfer as _delta_transfer

LEVEL_FIELDS = ('u', 'f', 'uold', 'fold', 'tau')


COMPOSITE_PARTS = ('impl', 'expl', 'comp1', 'comp2')
"""Attribute names of the split right-hand side datatypes, rounded part by part."""


def round_value(value, dtype):
    """
    Round one datatype instance through ``dtype``, in place.

    Dispatches over the backends the project supports rather than assuming a numpy array, and
    raises rather than silently leaving a value untouched -- an emulation that quietly does nothing
    would report a reduced-precision result that never was.

    Parameters
    ----------
    value : dtype_u or dtype_f
        The value to round.
    dtype : numpy.dtype
        Target precision.

    Raises
    ------
    NotImplementedError
        If the datatype is not one this can round.
    """
    if value is None:
        return
    if any(hasattr(value, part) for part in COMPOSITE_PARTS):  # split right-hand side
        for part in COMPOSITE_PARTS:
            if hasattr(value, part):
                round_value(getattr(value, part), dtype)
    elif hasattr(value, 'getArray'):  # PETSc Vec
        array = value.getArray()
        array[:] = array.astype(dtype)
    elif hasattr(value, 'values') and hasattr(value.values, 'vector'):  # fenics_mesh
        vector = value.values.vector()
        local = vector.get_local()
        vector.set_local(local.astype(dtype).astype(local.dtype))
        vector.apply('insert')
    elif isinstance(value, np.ndarray):
        value[:] = np.asarray(value).astype(dtype)
    else:
        raise NotImplementedError(
            f'level_precision is not supported for {type(value).__name__}: it cannot be rounded '
            f'through another precision in place'
        )


def round_level(lvl):
    """
    Round everything a level stores through its ``level_precision``.

    A no-op when the level did not ask for one, which keeps the default path datatype-agnostic.

    Parameters
    ----------
    lvl : pySDC.core.level.Level
        The level to round.
    """
    token = getattr(lvl.sweep.params, 'level_precision', None)
    if token is None:
        return
    dtype = np.dtype(token)
    for name in LEVEL_FIELDS:
        for value in getattr(lvl, name):
            round_value(value, dtype)


class RoundedLevelMixin:
    """
    Rounds a level after every sweep.

    Mixed into the delta-form hierarchy this is the emulation. Mixed into a *stock* sweeper and
    paired with :class:`rounding_transfer` it is the control instead, running ordinary MLSDC with a
    reduced-precision coarse level, which is what the delta form is claimed to fix -- and without
    that row the other rows say nothing.
    """

    def update_nodes(self):
        """
        Sweep, then round the level.

        Returns
        -------
        None
        """
        super().update_nodes()
        round_level(self.level)
        return None


class delta_transfer(_delta_transfer):
    """Delta-form transfer that rounds each level's storage after touching it."""

    def restrict(self):
        """
        Restrict, then round the coarse level.

        Returns
        -------
        None
        """
        super().restrict()
        round_level(self.coarse)
        # after the rounding, so the reference is the value the level actually holds and a level
        # that receives nothing shifts its residual by exactly zero
        self.coarse.u0_reference = self.coarse.prob.dtype_u(self.coarse.u[0])
        return None

    def prolong(self):
        """
        Prolong, then round the fine level.

        Returns
        -------
        None
        """
        super().prolong()
        round_level(self.fine)
        return None


class delta_implicit_rounded(RoundedLevelMixin, delta_implicit):
    """
    Delta-form sweeper whose level storage is rounded through ``level_precision``.

    Which experiment this is depends entirely on the transfer it is paired with:
    :class:`delta_transfer` makes it the emulation, :class:`rounding_transfer` the control.
    """


class delta_imex_1st_order_rounded(RoundedLevelMixin, delta_imex_1st_order):
    """IMEX counterpart of :class:`delta_implicit_rounded`."""


class rounding_transfer(BaseTransfer):
    """Stock transfer that rounds the coarse level after restriction. For the control only."""

    def restrict(self):
        """
        Restrict as usual, then round the coarse level.

        Returns
        -------
        None
        """
        super().restrict()
        round_level(self.coarse)
        return None
