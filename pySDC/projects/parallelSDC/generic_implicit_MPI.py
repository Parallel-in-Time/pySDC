from mpi4py import MPI

from pySDC.core.Sweeper import sweeper


class generic_implicit_MPI(sweeper):
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

        # call parent's initialization routine
        super(generic_implicit_MPI, self).__init__(params)

        # get QI matrix
        self.QI = self.get_Qdelta_implicit(self.coll, qd_type=self.params.QI)

        self.rank = self.params.comm.Get_rank()

    def integrate(self):
        """
        Integrates the right-hand side

        Returns:
            list of dtype_u: containing the integral as values
        """

        # get current level and problem description
        L = self.level
        P = L.prob

        me = P.dtype_u(P.init, val=0.0)
        for m in range(self.coll.num_nodes):
            if m == self.rank:
                self.params.comm.Reduce(L.dt * self.coll.Qmat[m + 1, self.rank + 1] * L.f[self.rank + 1],
                                        me, root=m, op=MPI.SUM)
            else:
                self.params.comm.Reduce(L.dt * self.coll.Qmat[m + 1, self.rank + 1] * L.f[self.rank + 1],
                                        None, root=m, op=MPI.SUM)

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

        # gather all terms which are known already (e.g. from the previous iteration)
        # this corresponds to u0 + QF(u^k) - QdF(u^k) + tau

        # get QF(u^k)
        rhs = self.integrate()

        rhs -= L.dt * self.QI[self.rank + 1, self.rank + 1] * L.f[self.rank + 1]

        # add initial value
        rhs += L.u[0]
        # add tau if associated
        if L.tau[self.rank] is not None:
            rhs += L.tau[self.rank]

        # build rhs, consisting of the known values from above and new values from previous nodes (at k+1)

        # implicit solve with prefactor stemming from the diagonal of Qd
        L.u[self.rank + 1] = P.solve_system(rhs, L.dt * self.QI[self.rank + 1, self.rank + 1], L.u[self.rank + 1],
                                            L.time + L.dt * self.coll.nodes[self.rank])
        # update function values
        L.f[self.rank + 1] = P.eval_f(L.u[self.rank + 1], L.time + L.dt * self.coll.nodes[self.rank])

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

        # check if Mth node is equal to right point and do_coll_update is false, perform a simple copy
        if self.coll.right_is_node and not self.params.do_coll_update:
            # a copy is sufficient
            L.uend = self.params.comm.bcast(L.u[self.rank + 1], root=self.params.comm.Get_size() - 1)
        else:
            raise NotImplementedError('require last node to be identical with right interval boundary')

        return None

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
        res = self.integrate()
        res += L.u[0] - L.u[self.rank + 1]
        # add tau if associated
        if L.tau[self.rank] is not None:
            res += L.tau[self.rank]
        # use abs function from data type here
        res_norm = abs(res)

        # find maximal residual over the nodes
        L.status.residual = self.params.comm.allreduce(res_norm, op=MPI.MAX)

        # indicate that the residual has seen the new values
        L.status.updated = False

        return None

    def predict(self):
        """
        Predictor to fill values at nodes before first sweep

        Default prediction for the sweepers, only copies the values to all collocation nodes
        and evaluates the RHS of the ODE there
        """

        # get current level and problem description
        L = self.level
        P = L.prob

        # evaluate RHS at left point
        L.f[0] = P.eval_f(L.u[0], L.time)

        if self.params.initial_guess == 'spread':
            L.u[self.rank + 1] = P.dtype_u(L.u[0])
            L.f[self.rank + 1] = P.eval_f(L.u[self.rank + 1], L.time + L.dt * self.coll.nodes[self.rank])
        else:
            L.u[self.rank + 1] = P.dtype_u(init=P.init, val=0.0)
            L.f[self.rank + 1] = P.dtype_f(init=P.init, val=0.0)

        # indicate that this level is now ready for sweeps
        L.status.unlocked = True
        L.status.updated = True
