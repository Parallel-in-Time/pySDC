from pySDC.core.sweeper import Sweeper


class multi_implicit(Sweeper):
    """
    SDC sweeper for right hand sides with two parts, each treated implicitly with its own solver and preconditioner.

    First-order multi-implicit sweeper for two components

    Attributes:
        Q1: implicit integration matrix for the first component
        Q2: implicit integration matrix for the second component
    """

    def __init__(self, params, level):
        """
        Initialization routine for the custom sweeper

        Args:
            params: parameters for the sweeper
            level (pySDC.Level.level): the level that uses this sweeper
        """

        # Default choice: implicit Euler
        if 'Q1' not in params:
            params['Q1'] = 'IE'
        if 'Q2' not in params:
            params['Q2'] = 'IE'

        # call parent's initialization routine
        super(multi_implicit, self).__init__(params, level)

        # Integration matrices
        self.Q1 = self.get_Qdelta_implicit(qd_type=self.params.Q1)
        self.Q2 = self.get_Qdelta_implicit(qd_type=self.params.Q2)

    def integrate(self):
        """
        Integrates the right-hand side (two components)

        Returns:
            list of dtype_u: containing the integral as values
        """

        # get current level and problem description
        L = self.level
        P = L.prob
        M = self.coll.num_nodes

        # sum the components once per node, not once per entry of Q
        F = [L.f[j].comp1 + L.f[j].comp2 for j in range(1, M + 1)]

        me = []

        # integrate RHS over all collocation nodes
        for m in range(1, M + 1):
            # new instance of dtype_u, initialize values with 0
            me.append(P.dtype_u(P.init, val=0.0))
            for j in range(1, M + 1):
                me[-1] += L.dt * self.coll.Qmat[m, j] * F[j - 1]

        return me

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
        Q, Q1, Q2 = self.coll.Qmat, self.Q1, self.Q2

        # the components of F(u^k), taken once: the sweep replaces L.f[m], so these stay the old values
        F1 = [L.f[j].comp1 for j in range(1, M + 1)]
        F2 = [L.f[j].comp2 for j in range(1, M + 1)]

        # do the sweep
        for m in range(1, M + 1):
            t = L.time + L.dt * self.coll.nodes[m - 1]

            # u0 + QF(u^k) - Q1F1(u^k) + tau, plus Q1F1(u^k+1) from the nodes already updated
            rhs = P.dtype_u(L.u[0])
            for j in range(1, M + 1):
                rhs += L.dt * (Q[m, j] - Q1[m, j]) * F1[j - 1]
                rhs += L.dt * Q[m, j] * F2[j - 1]
            if L.tau[m - 1] is not None:
                rhs += L.tau[m - 1]
            for j in range(1, m):
                rhs += L.dt * Q1[m, j] * L.f[j].comp1

            # implicit solve with prefactor stemming from Q1
            L.u[m] = P.solve_system_1(rhs, L.dt * Q1[m, m], L.u[m], t)

            # substract Q2F2(u^k) and add Q2F2(u^k+1)
            rhs = P.dtype_u(L.u[m])
            for j in range(1, M + 1):
                rhs -= L.dt * Q2[m, j] * F2[j - 1]
            for j in range(1, m):
                rhs += L.dt * Q2[m, j] * L.f[j].comp2

            L.u[m] = P.solve_system_2(rhs, L.dt * Q2[m, m], L.u[m], t)  # TODO: is L.u[m] a good guess?

            # update function values
            L.f[m] = P.eval_f(L.u[m], t)

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
        if self.coll.right_is_node and not self.params.do_coll_update:
            # a copy is sufficient
            L.uend = P.dtype_u(L.u[-1])
        else:
            # start with u0 and add integral over the full interval (using coll.weights)
            L.uend = P.dtype_u(L.u[0])
            for m in range(self.coll.num_nodes):
                L.uend += L.dt * self.coll.weights[m] * (L.f[m + 1].comp1 + L.f[m + 1].comp2)
            # add up tau correction of the full interval (last entry)
            if L.tau[-1] is not None:
                L.uend += L.tau[-1]

        return None
