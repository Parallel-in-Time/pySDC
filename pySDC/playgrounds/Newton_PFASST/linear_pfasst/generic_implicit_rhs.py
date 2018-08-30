from pySDC.core.Sweeper import sweeper

from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit


class generic_implicit_rhs(generic_implicit):
    """
    Generic implicit sweeper, expecting lower triangular matrix type as input, has RHS to consider

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
        # this corresponds to u0 + QF(u^k) - QdF(u^k) + tau

        # get QF(u^k)
        integral = self.integrate()
        for m in range(M):

            # get -QdF(u^k)_m
            for j in range(M + 1):
                integral[m] -= L.dt * self.QI[m + 1, j] * L.f[j]

            # add initial value
            integral[m] += L.u[0]

            # add rhs vector
            integral[m] += L.rhs[m]

        # do the sweep
        for m in range(0, M):
            # build rhs, consisting of the known values from above and new values from previous nodes (at k+1)
            rhs = P.dtype_u(integral[m])
            for j in range(m + 1):
                rhs += L.dt * self.QI[m + 1, j] * L.f[j]

            # implicit solve with prefactor stemming from the diagonal of Qd
            L.u[m + 1] = P.solve_system(rhs, L.dt * self.QI[m + 1, m + 1], L.u[m + 1],
                                        L.time + L.dt * self.coll.nodes[m])
            # update function values
            L.f[m + 1] = P.eval_f(L.u[m + 1], L.time + L.dt * self.coll.nodes[m])

        # indicate presence of new values at this level
        L.status.updated = True

        return None

    def compute_end_point(self):
        """
        Compute u at the right point of the interval

        The value uend computed here is a full evaluation of the Picard formulation unless do_full_update==False

        Returns:
            None
        """

        # get current level and problem description
        L = self.level
        P = L.prob

        # check if Mth node is equal to right point and do_coll_update is false, perform a simple copy
        if (self.coll.right_is_node and not self.params.do_coll_update):
            # a copy is sufficient
            L.uend = P.dtype_u(L.u[-1])
        else:
            # start with u0 and add integral over the full interval (using coll.weights)
            L.uend = P.dtype_u(L.u[0])
            for m in range(self.coll.num_nodes):
                L.uend += L.dt * self.coll.weights[m] * L.f[m + 1]

        return None

    def predict(self, u0=None, slot=None):
        """
        Predictor to fill values at nodes before first sweep

        Default prediction for the sweepers, only copies the values to all collocation nodes
        and evaluates the RHS of the ODE there
        """

        # get current level and problem description
        L = self.level
        P = L.prob

        # evaluate RHS at left point
        L.u[0] = P.dtype_u(P.init, val=0.0)
        L.f[0] = P.dtype_f(P.init, val=0.0)

        for m in range(1, self.coll.num_nodes + 1):
            L.f[m] = P.eval_f(L.u[m], L.time + L.dt * self.coll.nodes[m - 1])

        # indicate that this level is now ready for sweeps
        L.status.unlocked = True
        L.status.updated = True

    def compute_residual(self):
        """
        Computation of the residual using the collocation matrix Q
        """

        # get current level and problem description
        L = self.level

        # check if there are new values (e.g. from a sweep)
        # assert L.status.updated

        # compute the residual for each node

        # build QF(u)
        res_norm = []
        res = self.integrate()
        for m in range(self.coll.num_nodes):
            # add u0 and subtract u at current node
            res[m] += L.u[0] - L.u[m + 1] + L.rhs[m]
            # use abs function from data type here
            res_norm.append(abs(res[m]))

        # find maximal residual over the nodes
        L.status.residual = max(res_norm)

        # indicate that the residual has seen the new values
        L.status.updated = False

        return None
