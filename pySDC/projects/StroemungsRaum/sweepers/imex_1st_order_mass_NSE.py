from pySDC.implementations.sweeper_classes.imex_1st_order_mass import imex_1st_order_mass


class imex_1st_order_mass_NSE(imex_1st_order_mass):
    """
    Custom sweeper class, implements Sweeper.py

    First-order IMEX sweeper using implicit/explicit Euler as base integrator, with mass or weighting matrix
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

        #  store old value for residual computation
        L.uold = L.u.copy()

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

    def compute_residual(self, stage=None):
        """
        Computation of the residual using the collocation matrix Q

        Args:
            stage (str): The current stage of the step the level belongs to
        """

        # get current level and problem description
        L = self.level

        # Check if we want to skip the residual computation to gain performance
        # Keep in mind that skipping any residual computation is likely to give incorrect outputs of the residual!
        if stage in self.params.skip_residual_computation:
            L.status.residual = 0.0 if L.status.residual is None else L.status.residual
            return None

        # compute the residual for each node
        # build QF(u)
        res_norm = []
        res = [0] * (self.coll.num_nodes + 1)
        for m in range(self.coll.num_nodes):

            # compute the residual at node m, using the incremental criterion
            if L.uold[m + 1] is not None:
                res[m] = L.u[m + 1] - L.uold[m + 1]
            else:
                res[m] = L.u[m + 1]

            # Due to different boundary conditions we might have to fix the residual
            if L.prob.fix_bc_for_residual:
                L.prob.fix_residual(res[m])
            # use abs function from data type here
            res_norm.append(abs(res[m]))

        # find maximal residual over the nodes
        L.status.residual = max(res_norm)

        if L.time == 3.1250000000e-04 and L.status.residual == 0.0:
            L.status.residual = 1.0

        # indicate that the residual has seen the new values
        L.status.updated = False

        return None
