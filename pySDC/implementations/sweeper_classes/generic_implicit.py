from pySDC.core.Sweeper import sweeper


class generic_implicit(sweeper):
    """
    Generic implicit sweeper, expecting lower triangular matrix type as input

    Attributes:
        QI: lower triangular matrix
    """

    def __init__(self, params):
        """
        Initialization routine for the custom sweeper

        Args:
            params: parameters for the sweeper
        """

        if 'QI' not in params:
            params['QI'] = 'IE'

        super().__init__(params)

        self.QI = self.get_Qdelta_implicit(self.coll, qd_type=self.params.QI)

    def update_nodes(self):
        """
        Update the u- and f-values at the collocation nodes -> corresponds to a single sweep over all nodes

        Returns:
            None
        """
        M = self.coll.num_nodes

        # only if the level has been touched before
        assert self.level.status.unlocked

        rhs = self.build_right_hand_side()

        for m in range(0, M):
            self.sweep(rhs, m)

        self.level.status.updated = True

        return None

    def build_right_hand_side(self):
        rhs = self.initialize_right_hand_side_buffer()
        self.add_Q_minus_QD_times_F(rhs)
        self.add_initial_conditions(rhs)
        self.add_tau_correction(rhs)
        return rhs

    def initialize_right_hand_side_buffer(self):
        problem = self.level.prob

        return [problem.dtype_u(problem.init, val=0.0) for _ in range(self.coll.num_nodes)]

    def add_Q_minus_QD_times_F(self, rhs):
        self.add_matrix_times_f_evaluations_to(self.coll.Qmat - self.QI, rhs)

    def add_matrix_times_f_evaluations_to(self, matrix, rhs):
        for m in range(1, self.coll.num_nodes + 1):
            for j in range(1, self.coll.num_nodes + 1):
                rhs[m - 1] += self.level.dt * matrix[m, j] * self.level.f[j]

    def add_initial_conditions(self, rhs):
        for i in range(len(rhs)):
            rhs[i] += self.level.u[0]

    def add_tau_correction(self, rhs):
        for m in range(self.coll.num_nodes):
            if self.level.tau[m] is not None:
                rhs[m] += self.level.tau[m]

    def sweep(self, rhs, current_node):
        L = self.level
        P = L.prob
        m = current_node

        self.add_new_information_from_forward_substitution(rhs, m)

        L.u[m + 1] = P.solve_system(
            rhs[m], L.dt * self.QI[m + 1, m + 1], L.u[m + 1], L.time + L.dt * self.coll.nodes[m]
        )
        L.f[m + 1] = P.eval_f(L.u[m + 1], L.time + L.dt * self.coll.nodes[m])

    def add_new_information_from_forward_substitution(self, rhs, current_node):
        for j in range(1, current_node + 1):
            rhs[current_node] += self.level.dt * self.QI[current_node + 1, j] * self.level.f[j]

    def integrate(self):
        """
        Integrates the right-hand side

        Returns:
            list of dtype_u: containing the integral as values
        """
        integral = self.initialize_right_hand_side_buffer()
        self.add_matrix_times_f_evaluations_to(self.coll.Qmat, integral)
        return integral

    def compute_end_point(self):
        """
        Compute u at the right point of the interval

        The value uend computed here is a full evaluation of the Picard formulation unless do_full_update==False

        Returns:
            None
        """

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
                L.uend += L.dt * self.coll.weights[m] * L.f[m + 1]
            # add up tau correction of the full interval (last entry)
            if L.tau[-1] is not None:
                L.uend += L.tau[-1]

        return None
