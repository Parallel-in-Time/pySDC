r"""
Node-parallel delta-form sweeper.

Kept in its own module because importing it requires ``mpi4py``, which :mod:`sweepers` does not.

The MPI sweeper assigns one collocation node per rank and therefore uses only the **diagonal** of
:math:`Q^\Delta`. The delta form collapses accordingly: with
:math:`\varepsilon_r = u_0 + \tau_r + \Delta t (Q f^k)_r - u^k_r` for the rank's own node,

.. math::
    \delta_r = \varepsilon_r + \Delta t Q^\Delta_{rr}\,\big(f(u^k_r + \delta_r) - f(u^k_r)\big),

with no accumulation over other nodes. The node-local piece is identical to the serial case, so
:meth:`DeltaFormMixin._solve_correction` is reused unchanged and all three strategies
(``solve_system_delta``, ``linear_implicit``, substitution fallback) work here too.
"""

from pySDC.implementations.sweeper_classes.generic_implicit_MPI import generic_implicit_MPI
from pySDC.projects.DeltaSDC.sweepers import DeltaFormMixin


class delta_implicit_MPI(DeltaFormMixin, generic_implicit_MPI):
    """Delta-form counterpart of :class:`generic_implicit_MPI`. One collocation node per rank."""

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

        # residual at this rank's node, in backend precision
        eps = self.integrate()
        eps += lvl.u[0]
        eps -= u_old
        if lvl.tau[rank] is not None:
            eps += lvl.tau[rank]

        delta = self._solve_correction(self._to_work(prob, eps), alpha, u_old, f_old, t_node)

        lvl.u[rank + 1] = u_old + self._to_backend(prob, self._to_work(prob, delta))
        lvl.f[rank + 1] = prob.eval_f(lvl.u[rank + 1], t_node)

        lvl.status.updated = True
        return None
