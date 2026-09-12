r"""
Node-parallel delta-form sweeper.

Kept in its own module because importing it requires ``mpi4py``, which
:mod:`~pySDC.implementations.sweeper_classes.delta_form` does not.

The MPI sweeper assigns one collocation node per rank and therefore uses only the **diagonal** of
:math:`Q^\Delta`. The delta form collapses accordingly: with
:math:`\varepsilon_r = u_0 + \tau_r + \Delta t (Q f^k)_r - u^k_r` for the rank's own node,

.. math::
    \delta_r = \varepsilon_r + \Delta t Q^\Delta_{rr}\,\big(f(u^k_r + \delta_r) - f(u^k_r)\big),

with no accumulation over other nodes. The node-local piece is identical to the serial case, so
:meth:`DeltaFormMixin._solve_correction` is reused unchanged and all three strategies
(``solve_system_delta``, ``linear_implicit``, substitution fallback) work here too.

"""

from mpi4py import MPI

from pySDC.implementations.sweeper_classes.delta_form import DeltaFormMixin
from pySDC.implementations.sweeper_classes.generic_implicit_MPI import generic_implicit_MPI


class delta_implicit_MPI(DeltaFormMixin, generic_implicit_MPI):
    """Delta-form counterpart of :class:`generic_implicit_MPI`. One collocation node per rank."""

    def _set_work_scale(self, values):
        """
        Agree the divisor across the node communicator.

        Each rank holds one collocation node, so left alone every rank would scale its corrections by
        its own node's residual and quantise differently from a serial run doing the same work. The
        scale is conceptually the residual's magnitude, which is a property of the whole sweep, so it
        is reduced. One scalar per sweep.

        Parameters
        ----------
        values : list
            This rank's residual, in backend units.
        """
        super()._set_work_scale(values)
        if self._work_dtype is not None and self._scales_corrections():
            self._work_scale = self.comm.allreduce(self._work_scale, op=MPI.MAX)

    def _residual_nodes(self):
        r"""
        Compute :math:`\varepsilon_r` at this rank's node, in backend precision.

        Returned as a one-element list, so the multi-level machinery -- which is written for a list
        of nodes -- reads the same here as it does serially.

        Returns
        -------
        list
            This rank's collocation residual, as a single-element list.
        """
        lvl = self.level
        eps = self.integrate()
        eps += lvl.u[0]
        eps -= lvl.u[self.rank + 1]
        if lvl.tau[self.rank] is not None:
            eps += lvl.tau[self.rank]
        return [eps]

    def update_nodes(self):
        """
        Perform one delta-form sweep for this rank's collocation node.

        Returns
        -------
        None
        """
        lvl = self.level
        prob = lvl.prob
        assert lvl.status.unlocked
        self._delta_setup()

        rank = self.rank
        t_node = lvl.time + lvl.dt * self.coll.nodes[rank]
        alpha = lvl.dt * self.QI[rank + 1, rank + 1]

        u_old = prob.dtype_u(lvl.u[rank + 1])
        f_old = prob.dtype_f(lvl.f[rank + 1])

        residual = self._residual_nodes()
        self._set_work_scale(residual)
        delta = self._solve_correction(self._to_work(prob, residual[0]), alpha, u_old, f_old, t_node)

        lvl.u[rank + 1] = u_old + self._to_backend(prob, self._to_work(prob, delta))
        lvl.f[rank + 1] = prob.eval_f(lvl.u[rank + 1], t_node)

        # No increment is evaluated here: QI is diagonal on this layout, so nothing accumulates
        # across nodes and the sweep has nothing to accumulate it into.
        lvl.status.updated = True
        return None
