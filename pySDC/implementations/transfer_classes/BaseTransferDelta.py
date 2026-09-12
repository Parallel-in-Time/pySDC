r"""
Space-time transfer for the delta-form hierarchy.

:class:`BaseTransfer` makes a coarse level rebuild its own residual from :math:`\mathcal{O}(1)`
coarse data and recovers the coarse-grid correction as :math:`u_G - u_G^{\mathrm{old}}`. Both have
exact algebraic replacements -- :math:`\varepsilon_G = R\varepsilon_F`, which is what the FAS
:math:`\tau` is *for*, and the correction is the sum of the sweep's own increments -- and this
transfer uses those instead. It is what decides whether a level rebuilds its residual or is handed
one: the delta-form sweepers do both, and this is what hands one down.

The FAS :math:`\tau` is **substituted, not discarded**. Writing
:math:`\tau = R(\Delta t\,Q_F f_F) - \Delta t\,Q_G f_G` and putting it into the coarse residual
:math:`\Delta t (Q_G f_G) + u_G[0] - u_G[m] + \tau` cancels the :math:`\Delta t\,Q_G f_G` terms
identically and leaves :math:`R\varepsilon_F`. So a coarse level handed that residual is solving
precisely the FAS-corrected problem -- and must *not* also add :math:`\tau`, which would count it
twice. What is skipped is materialising :math:`\tau` as an array, since the quantity it exists to
produce arrives directly.

Identical to :class:`BaseTransfer` up to round-off at backend precision, verified for two, three and
four levels and for PFASST -- and against a control with :math:`\tau` genuinely zeroed, which does
not converge at all. It is also slightly cheaper, because nothing then reads :math:`\tau` except
:meth:`compute_end_point` in the quadrature-update case: see :meth:`delta_transfer.restrict`.

What the reformulation buys is that every coarse-level quantity becomes proportional to the fine
residual rather than to :math:`|u|`. A coarse level is a preconditioner whose returned correction
tends to zero, so once nothing on it is rebuilt out of :math:`\mathcal{O}(1)` state, its arithmetic
error is proportional to that correction rather than to the solution -- which is what lets it run
below backend precision without capping the accuracy of the iteration.
"""

from pySDC.core.base_transfer import BaseTransfer
from pySDC.core.errors import UnlockError


class delta_transfer(BaseTransfer):
    """
    Space-time transfer that passes a residual down and an accumulated correction up.

    Drop-in for :class:`BaseTransfer`, and identical to it up to round-off when every level is at
    backend precision. Falls back to the stock prolongation when the coarse level has banked no
    corrections, so a hierarchy mixing delta-form and stock sweepers still runs.
    """

    def coarse_reads_tau(self):
        """
        Whether the coarse level reads the FAS :math:`\\tau` at all.

        The sweep does not -- it is handed the restricted fine residual, which is what :math:`\\tau`
        exists to produce -- and neither does :meth:`DeltaFormMixin.compute_residual`. The one
        remaining reader is :meth:`compute_end_point`, and only when the end point comes from the
        quadrature update rather than a copy of the last node. When nothing reads it, building it is
        a coarse ``integrate()`` and a restriction spent on a quantity that is then discarded.

        Returns
        -------
        bool
            Whether :math:`\\tau` has to be built.
        """
        SG = self.coarse.sweep
        return not (SG.coll.right_is_node and not SG.params.do_coll_update)

    def restrict_state(self):
        """
        Restrict ``u`` and re-evaluate ``f``, which is :meth:`BaseTransfer.restrict` without
        :math:`\\tau`.

        Returns
        -------
        None

        Raises
        ------
        UnlockError
            If the fine level has not been unlocked yet.
        """
        F, G = self.fine, self.coarse
        SF, SG, PG = F.sweep, G.sweep, G.prob
        if not F.status.unlocked:
            raise UnlockError('fine level is still locked, cannot use data from there')

        tmp_u = [self.space_transfer.restrict(F.u[m]) for m in range(1, SF.coll.num_nodes + 1)]
        G.u[0] = self.space_transfer.restrict(F.u[0])
        for n in range(1, SG.coll.num_nodes + 1):
            # float(), not the raw np.float64 entry: see DeltaFormMixin._coeff
            G.u[n] = float(self.Rcoll[n - 1, 0]) * tmp_u[0]
            for m in range(1, SF.coll.num_nodes):
                G.u[n] += float(self.Rcoll[n - 1, m]) * tmp_u[m]

        G.f[0] = PG.eval_f(G.u[0], G.time)
        for m in range(1, SG.coll.num_nodes + 1):
            G.f[m] = PG.eval_f(G.u[m], G.time + G.dt * SG.coll.nodes[m - 1])
            G.uold[m] = PG.dtype_u(G.u[m])
            G.fold[m] = PG.dtype_f(G.f[m])

        G.status.unlocked = True
        return None

    def restrict(self):
        """
        Restrict the state, then hand the coarse level the restricted fine residual.

        Returns
        -------
        None
        """
        SF, SG = self.fine.sweep, self.coarse.sweep
        SF._delta_setup()
        SG._delta_setup()
        eps_F = SF._residual_nodes()
        super().restrict() if self.coarse_reads_tau() else self.restrict_state()

        tmp = [self.space_transfer.restrict(eps) for eps in eps_F]
        eps_G = []
        for n in range(SG.coll.num_nodes):
            acc = float(self.Rcoll[n, 0]) * tmp[0]
            for m in range(1, SF.coll.num_nodes):
                acc += float(self.Rcoll[n, m]) * tmp[m]
            eps_G.append(SG._to_work(self.coarse.prob, acc))
        SG.eps_in = eps_G
        SG.delta_acc = None
        # after the rounding, so the reference is the value the level actually holds and a level
        # that receives nothing shifts its residual by exactly zero
        self.coarse.u0_reference = self.coarse.prob.dtype_u(self.coarse.u[0])
        return None

    def prolong(self):
        """
        Add the coarse level's accumulated correction to the fine level.

        If the fine level is itself a coarse level of something above it, its residual and its own
        accumulated correction are advanced by the same update, so neither has to be rebuilt from
        :math:`\\mathcal{O}(1)` state later in the V-cycle.

        Returns
        -------
        None
        """
        if self.coarse.sweep.delta_acc is None:
            return super().prolong()

        F, PF = self.fine, self.fine.prob
        SF, SG = self.fine.sweep, self.coarse.sweep
        SF._delta_setup()
        tmp = [self.space_transfer.prolong(delta) for delta in self.coarse.sweep.delta_acc]

        corr = []
        for n in range(1, SF.coll.num_nodes + 1):
            c = float(self.Pcoll[n - 1, 0]) * tmp[0]
            for m in range(1, SG.coll.num_nodes):
                c += float(self.Pcoll[n - 1, m]) * tmp[m]
            # Quantise the correction at the correction precision, then bring it back to backend
            # units before applying it. `_to_work` alone also divides by the sweep's scale, and
            # adding *that* to the nodal value would be wrong by a factor of the scale.
            c = SF._to_backend(PF, SF._to_work(PF, c))
            corr.append(SF._to_work(PF, c))

            t_node = F.time + F.dt * SF.coll.nodes[n - 1]
            u_old, f_old = PF.dtype_u(F.u[n]), PF.dtype_f(F.f[n])
            F.u[n] += c
            F.f[n] = PF.eval_f(F.u[n], t_node)
            if SF.eps_in is not None:
                # records into SF._dfs, which is what advance_residual reads below
                SF._f_increment(PF, F.f[n], f_old, u_old, c, t_node)

        if SF.eps_in is not None:
            SF.eps_in = SF.advance_residual(SF.eps_in, corr, SF._dfs)
            SF.accumulate(corr)
        return None
