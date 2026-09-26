from pySDC.core.base_transfer import BaseTransfer
from pySDC.core.errors import UnlockError


class base_transfer_mass(BaseTransfer):
    """
    Standard base_transfer class

    Attributes:
        logger: custom logger for sweeper-related logging
        params(__Pars): parameter object containing the custom parameters passed by the user
        fine (pySDC.Level.level): reference to the fine level
        coarse (pySDC.Level.level): reference to the coarse level
    """

    def restrict(self):
        """
        Space-time restriction routine

        The routine applies the spatial restriction operator to teh fine values on the fine nodes, then reevaluates f
        on the coarse level. This is used for the first part of the FAS correction tau via integration. The second part
        is the integral over the fine values, restricted to the coarse level. Finally, possible tau corrections on the
        fine level are restricted as well.
        """

        # get data for easier access
        F = self.fine
        G = self.coarse

        PG = G.prob
        PF = F.prob

        SF = F.sweep
        SG = G.sweep

        # only if the level is unlocked at least by prediction
        if not F.status.unlocked:
            raise UnlockError('fine level is still locked, cannot use data from there')

        # restrict fine values in space
        tmp_u = []
        for m in range(1, SF.coll.num_nodes + 1):
            tmp_u.append(self.space_transfer.project(F.u[m]))

        # restrict collocation values
        G.u[0] = self.space_transfer.project(F.u[0])
        for n in range(1, SG.coll.num_nodes + 1):
            G.u[n] = self.Rcoll[n - 1, 0] * tmp_u[0]
            for m in range(1, SF.coll.num_nodes):
                G.u[n] += self.Rcoll[n - 1, m] * tmp_u[m]

        # re-evaluate f on coarse level
        G.f[0] = PG.eval_f(G.u[0], G.time)
        for m in range(1, SG.coll.num_nodes + 1):
            G.f[m] = PG.eval_f(G.u[m], G.time + G.dt * SG.coll.nodes[m - 1])

        # build coarse level tau correction part
        tauG = G.sweep.integrate()

        for m in range(SG.coll.num_nodes):
            tauG[m] = PG.apply_mass_matrix(G.u[m + 1]) - tauG[m]

        # build fine level tau correction part
        tauF = F.sweep.integrate()

        for m in range(SF.coll.num_nodes):
            tauF[m] = PF.apply_mass_matrix(F.u[m + 1]) - tauF[m]

        # restrict fine level tau correction part in space
        tmp_tau = []
        for m in range(SF.coll.num_nodes):
            tmp_tau.append(self.space_transfer.restrict_dual(tauF[m]))

        # restrict fine level tau correction part in collocation
        tauFG = []
        for n in range(1, SG.coll.num_nodes + 1):
            tauFG.append(self.Rcoll[n - 1, 0] * tmp_tau[0])
            for m in range(1, SF.coll.num_nodes):
                tauFG[-1] += self.Rcoll[n - 1, m] * tmp_tau[m]

        # build tau correction
        for m in range(SG.coll.num_nodes):
            G.tau[m] = tauG[m] - tauFG[m]

        if F.tau[0] is not None:
            # restrict possible tau correction from fine in space
            tmp_tau = []
            for m in range(SF.coll.num_nodes):
                tmp_tau.append(self.space_transfer.restrict_dual(F.tau[m]))

            # restrict possible tau correction from fine in collocation
            for n in range(SG.coll.num_nodes):
                for m in range(SF.coll.num_nodes):
                    G.tau[n] += self.Rcoll[n, m] * tmp_tau[m]
        else:
            pass

        # save u and rhs evaluations for interpolation
        for m in range(1, SG.coll.num_nodes + 1):
            G.uold[m] = PG.dtype_u(G.u[m])
            G.fold[m] = PG.dtype_f(G.f[m])

        # u0 lives in the DUAL space on every coarse level: the finest level supplies it as M u0,
        # and each further level restricts that dual vector with P^T. Without the else branch,
        # level 1 -> 2 kept the primal `project` result from above and L2-projected an already-dual
        # vector, which is why 2 levels worked and 3 did not.
        if F.level_index == 0:
            G.u[0] = self.space_transfer.restrict_dual(PF.apply_mass_matrix(F.u[0]))
        else:
            G.u[0] = self.space_transfer.restrict_dual(F.u[0])

        # works as a predictor
        G.status.unlocked = True

        return None

    def prolong(self):
        """
        Space-time prolongation routine

        This routine applies the spatial prolongation routine to the difference between the computed and the restricted
        values on the coarse level and then adds this difference to the fine values as coarse correction.
        """

        # get data for easier access
        F = self.fine
        G = self.coarse

        PF = F.prob

        SF = F.sweep
        SG = G.sweep

        # only of the level is unlocked at least by prediction or restriction
        if not G.status.unlocked:
            raise UnlockError('coarse level is still locked, cannot use data from there')

        # build coarse correction

        # interpolate values in space first
        tmp_u = []
        for m in range(1, SG.coll.num_nodes + 1):
            tmp_u.append(self.space_transfer.prolong(G.u[m] - G.uold[m]))

        # interpolate values in collocation
        # F.u[0] += tmp_u[0]
        for n in range(1, SF.coll.num_nodes + 1):
            for m in range(SG.coll.num_nodes):
                F.u[n] += self.Pcoll[n - 1, m] * tmp_u[m]

        # re-evaluate f on fine level
        # F.f[0] = PF.eval_f(F.u[0], F.time)
        for m in range(1, SF.coll.num_nodes + 1):
            F.f[m] = PF.eval_f(F.u[m], F.time + F.dt * SF.coll.nodes[m - 1])

        return None

    def prolong_f(self):
        """
        Space-time prolongation routine w.r.t. the rhs f

        Not available for the mass formulation, so this falls back to prolong().

        Under the mass formulation eval_f returns a load vector (f.impl = K u, f.expl = M g), which
        cannot be interpolated -- doing so reads it as a nodal function and makes the coarse
        correction actively wrong. The dimensionally correct dual transfer M_f P M_c^-1 restores the
        right answer but needs a coarse mass solve per node and still costs about seven times the
        iterations of simply re-evaluating f (measured on step_7's heat problem: 14.40 against 2.00
        at restol 5e-10). Re-evaluating is cheaper and more accurate here, so finter is ignored.
        """
        if not getattr(self, '_finter_warned', False):
            self.logger.warning('finter is not supported for the mass formulation, re-evaluating f instead')
            self._finter_warned = True

        return self.prolong()
