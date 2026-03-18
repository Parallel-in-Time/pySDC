from pySDC.implementations.sweeper_classes.imex_1st_order_mass import imex_1st_order_mass


class imex_1st_order_mass_NSE(imex_1st_order_mass):
    """
    Custom sweeper class implementing IMEX SDC for solving the incompressible Navier–Stokes equations
    using a projection method.

    """

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

        # This is somewhat ugly, but we have to apply the mass matrix on u0 only on the finest level
        if L.level_index == 0:
            u0 = P.apply_mass_matrix(L.u[0])
        else:
            u0 = L.u[0]

        for m in range(M):
            # subtract QIFI(u^k)_m + QEFE(u^k)_m
            for j in range(M + 1):
                integral[m] -= L.dt * (self.QI[m + 1, j] * L.f[j].impl + self.QE[m + 1, j] * L.f[j].expl)
            # add initial value
            integral[m] += u0
            # add tau if associated
            if L.tau[m] is not None:
                integral[m] += L.tau[m]

        # do the sweep
        for m in range(0, M):
            # build rhs, consisting of the known values from above and new values from previous nodes (at k+1)
            rhs = P.dtype_u(integral[m])
            for j in range(m + 1):
                rhs += L.dt * (self.QI[m + 1, j] * L.f[j].impl + self.QE[m + 1, j] * L.f[j].expl)

            L.u[m + 1], P.pn = P.solve_system(
                rhs,
                L.dt * self.QI[m + 1, m + 1],
                L.u[m + 1],
                L.time + L.dt * self.coll.nodes[m],
                L.dt * self.coll.nodes[m],
            )

            # update function values
            L.f[m + 1] = P.eval_f(L.u[m + 1], L.time + L.dt * self.coll.nodes[m])

        # indicate presence of new values at this level
        L.status.updated = True
        return None
