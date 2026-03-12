from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.sweeper_classes.imex_1st_order_MPI import imex_1st_order_MPI


class imex_1st_order_diagonal_serial(imex_1st_order):
    def update_nodes(self):
        """
        Update the u- and f-values at the collocation nodes -> corresponds to a single sweep over all nodes

        Returns:
            None
        """
        # get current level and problem description
        L = self.level
        P = L.prob

        # only if the level has been touched before
        assert L.status.unlocked

        # get number of collocation nodes for easier access
        M = self.coll.num_nodes

        # gather all terms which are known already (e.g. from the previous iteration)
        # this corresponds to u0 + QF(u^k) - QIFI(u^k) - QEFE(u^k) + tau

        # get QF(u^k)
        integral = self.integrate()
        for m in range(M):
            # subtract QIFI(u^k)_m + QEFE(u^k)_m
            for j in range(1, M + 1):
                integral[m] -= L.dt * (self.QI[m + 1, j] * L.f[j].impl + self.QE[m + 1, j] * L.f[j].expl)
            # add initial value
            integral[m] += L.u[0]
            # add tau if associated
            if L.tau[m] is not None:
                integral[m] += L.tau[m]

        # do the sweep
        for m in range(0, M) if L.status.sweep < L.params.nsweeps else [M - 1]:
            # build rhs, consisting of the known values from above and new values from previous nodes (at k+1)
            rhs = P.dtype_u(integral[m])
            for j in range(1, m + 1):
                rhs += L.dt * (self.QI[m + 1, j] * L.f[j].impl + self.QE[m + 1, j] * L.f[j].expl)

            # implicit solve with prefactor stemming from QI
            L.u[m + 1] = P.solve_system(
                rhs, L.dt * self.QI[m + 1, m + 1], L.u[m + 1], L.time + L.dt * self.coll.nodes[m]
            )

            # update function values
            if L.status.sweep < L.params.nsweeps:
                L.f[m + 1] = P.eval_f(L.u[m + 1], L.time + L.dt * self.coll.nodes[m])

        # indicate presence of new values at this level
        L.status.updated = True

        return None


class imex_1st_order_MPI_fixed_k(imex_1st_order_MPI):
    def update_nodes(self):
        """
        Update the u- and f-values at the collocation nodes -> corresponds to a single sweep over all nodes

        Returns:
            None
        """

        L = self.level
        P = L.prob

        # only if the level has been touched before
        assert L.status.unlocked

        # get number of collocation nodes for easier access

        # gather all terms which are known already (e.g. from the previous iteration)
        # this corresponds to u0 + QF(u^k) - QdF(u^k) + tau

        # get QF(u^k)
        rhs = self.integrate()

        # subtract QdF(u^k)
        rhs -= L.dt * (self.QI[self.rank + 1, self.rank + 1] * L.f[self.rank + 1].impl)

        # add initial conditions
        rhs += L.u[0]
        # add tau if associated
        if L.tau[self.rank] is not None:
            rhs += L.tau[self.rank]

        # implicit solve with prefactor stemming from the diagonal of Qd
        L.u[self.rank + 1] = P.solve_system(
            rhs,
            L.dt * self.QI[self.rank + 1, self.rank + 1],
            L.u[self.rank + 1],
            L.time + L.dt * self.coll.nodes[self.rank],
        )
        # update function values
        if L.status.sweep < L.params.nsweeps:
            L.f[self.rank + 1] = P.eval_f(L.u[self.rank + 1], L.time + L.dt * self.coll.nodes[self.rank])

        # indicate presence of new values at this level
        L.status.updated = True

        return None
