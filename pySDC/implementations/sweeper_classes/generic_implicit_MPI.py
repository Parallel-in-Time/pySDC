from mpi4py import MPI

from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.core.Sweeper import sweeper
import logging


class SweeperMPI(sweeper):
    """
    MPI based sweeper where each rank administers one collocation node. Adapt sweepers to MPI by use of multiple inheritance.
    See for example the `generic_implicit_MPI` sweeper, which has a class definition:

    ```
    class generic_implicit_MPI(SweeperMPI, generic_implicit):
    ```

    this means in inherits both from `SweeperMPI` and `generic_implicit`. The hierarchy works such that functions are first
    called from `SweeperMPI` and then from `generic_implicit`. For instance, in the `__init__` function, the `SweeperMPI`
    class adds a communicator and nothing else. The `generic_implicit` implicit class adds a preconditioner and so on.
    It's a bit confusing because `self.params` is overwritten in the second call to the `__init__` of the core `sweeper`
    class, but the `SweeperMPI` class adds parameters to the `params` dictionary, which will again be added in
    `generic_implicit`.
    """

    def __init__(self, params):
        self.logger = logging.getLogger('sweeper')

        if 'comm' not in params.keys():
            params['comm'] = MPI.COMM_WORLD
            self.logger.debug('Using MPI.COMM_WORLD for the communicator because none was supplied in the params.')
        super().__init__(params)

        if self.params.comm.size != self.coll.num_nodes:
            raise NotImplementedError(
                f'The communicator in the {type(self).__name__} sweeper needs to have one rank for each node as of now! That means we need {self.coll.num_nodes} nodes, but got {self.params.comm.size} processes.'
            )

    @property
    def comm(self):
        return self.params.comm

    @property
    def rank(self):
        return self.comm.rank

    def compute_end_point(self):
        """
        Compute u at the right point of the interval

        The value uend computed here is a full evaluation of the Picard formulation unless do_full_update==False

        Returns:
            None
        """

        L = self.level
        P = L.prob
        L.uend = P.dtype_u(P.init, val=0.0)

        # check if Mth node is equal to right point and do_coll_update is false, perform a simple copy
        if self.coll.right_is_node and not self.params.do_coll_update:
            # a copy is sufficient
            L.uend[:] = self.params.comm.bcast(L.u[self.rank + 1], root=self.params.comm.Get_size() - 1)
        else:
            raise NotImplementedError('require last node to be identical with right interval boundary')

        return None

    def compute_residual(self, stage=None):
        """
        Computation of the residual using the collocation matrix Q

        Args:
            stage (str): The current stage of the step the level belongs to
        """

        L = self.level

        # Check if we want to skip the residual computation to gain performance
        # Keep in mind that skipping any residual computation is likely to give incorrect outputs of the residual!
        if stage in self.params.skip_residual_computation:
            L.status.residual = 0.0 if L.status.residual is None else L.status.residual
            return None

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


class generic_implicit_MPI(SweeperMPI, generic_implicit):
    """
    Generic implicit sweeper parallelized across the nodes.
    Please supply a communicator as `comm` to the parameters!

    Attributes:
        rank (int): MPI rank
    """

    def integrate(self):
        """
        Integrates the right-hand side

        Returns:
            list of dtype_u: containing the integral as values
        """

        L = self.level
        P = L.prob

        me = P.dtype_u(P.init, val=0.0)
        for m in range(self.coll.num_nodes):
            recvBuf = me if m == self.rank else None
            self.params.comm.Reduce(
                L.dt * self.coll.Qmat[m + 1, self.rank + 1] * L.f[self.rank + 1], recvBuf, root=m, op=MPI.SUM
            )

        return me

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

        rhs -= L.dt * self.QI[self.rank + 1, self.rank + 1] * L.f[self.rank + 1]

        # add initial value
        rhs += L.u[0]
        # add tau if associated
        if L.tau[self.rank] is not None:
            rhs += L.tau[self.rank]

        # build rhs, consisting of the known values from above and new values from previous nodes (at k+1)

        # implicit solve with prefactor stemming from the diagonal of Qd
        L.u[self.rank + 1] = P.solve_system(
            rhs,
            L.dt * self.QI[self.rank + 1, self.rank + 1],
            L.u[self.rank + 1],
            L.time + L.dt * self.coll.nodes[self.rank],
        )
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

        L = self.level
        P = L.prob
        L.uend = P.dtype_u(P.init, val=0.0)

        # check if Mth node is equal to right point and do_coll_update is false, perform a simple copy
        if self.coll.right_is_node and not self.params.do_coll_update:
            super().compute_end_point()
        else:
            L.uend = P.dtype_u(L.u[0])
            self.params.comm.Allreduce(L.dt * self.coll.weights[self.rank] * L.f[self.rank + 1], L.uend, op=MPI.SUM)
            L.uend += L.u[0]

            # add up tau correction of the full interval (last entry)
            if L.tau[-1] is not None:
                L.uend += L.tau[-1]
        return None
