from pySDC.core.Sweeper import sweeper


class multi_implicit(sweeper):
    """
    Custom sweeper class, implements Sweeper.py

    First-order multi-implicit sweeper for two components

    Attributes:
        Q1: implicit integration matrix for the first component
        Q2: implicit integration matrix for the second component
    """

    def __init__(self, params):
        """
        Initialization routine for the custom sweeper

        Args:
            params: parameters for the sweeper
        """

        # Default choice: implicit Euler
        if 'Q1' not in params:
            params['Q1'] = 'IE'
        if 'Q2' not in params:
            params['Q2'] = 'IE'

        # call parent's initialization routine
        super(multi_implicit, self).__init__(params)

        # Integration matrices
        self.Q1 = self.get_Qdelta_implicit(coll=self.coll, qd_type=self.params.Q1)
        self.Q2 = self.get_Qdelta_implicit(coll=self.coll, qd_type=self.params.Q2)

    def integrate(self):
        """
        Integrates the right-hand side (two components)

        Returns:
            list of dtype_u: containing the integral as values
        """

        # get current level and problem description
        L = self.level
        P = L.prob

        me = []

        # integrate RHS over all collocation nodes
        for m in range(1, self.coll.num_nodes + 1):
            # new instance of dtype_u, initialize values with 0
            me.append(P.dtype_u(P.init, val=0.0))
            for j in range(1, self.coll.num_nodes + 1):
                me[-1] += L.dt * self.coll.Qmat[m, j] * (L.f[j].comp1 + L.f[j].comp2)

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

        # gather all terms which are known already (e.g. from the previous iteration)

        # get QF(u^k)
        integral = self.integrate()
        for m in range(M):
            # subtract Q1F1(u^k)_m
            for j in range(1, M + 1):
                integral[m] -= L.dt * self.Q1[m + 1, j] * L.f[j].comp1
            # add initial value
            integral[m] += L.u[0]
            # add tau if associated
            if L.tau[m] is not None:
                integral[m] += L.tau[m]

        # store Q2F2(u^k) for later usage
        Q2int = []
        for m in range(M):
            Q2int.append(P.dtype_u(P.init, val=0.0))
            for j in range(1, M + 1):
                Q2int[-1] += L.dt * self.Q2[m + 1, j] * L.f[j].comp2

        # do the sweep
        for m in range(0, M):
            # build rhs, consisting of the known values from above and new values from previous nodes (at k+1)
            rhs = P.dtype_u(integral[m])
            for j in range(1, m + 1):
                rhs += L.dt * self.Q1[m + 1, j] * L.f[j].comp1

            # implicit solve with prefactor stemming from Q1
            L.u[m + 1] = P.solve_system_1(rhs, L.dt * self.Q1[m + 1, m + 1], L.u[m + 1],
                                          L.time + L.dt * self.coll.nodes[m])

            # substract Q2F2(u^k) and add Q2F(u^k+1)
            rhs = L.u[m + 1] - Q2int[m]
            for j in range(1, m + 1):
                rhs += L.dt * self.Q2[m + 1, j] * L.f[j].comp2

            L.u[m + 1] = P.solve_system_2(rhs, L.dt * self.Q2[m + 1, m + 1], L.u[m + 1],  # TODO: is this a good guess?
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
