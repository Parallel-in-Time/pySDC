from mpi4py import MPI

from pySDC.core.Errors import ParameterError
from pySDC.implementations.sweeper_classes.generic_implicit_MPI import SweeperMPI, generic_implicit_MPI
from pySDC.projects.DAE.sweepers.fully_implicit_DAE import fully_implicit_DAE


class SweeperDAEMPI(SweeperMPI):
    r"""
    MPI base class for DAE sweepers where each rank administers one collocation node.

    This class provides all the stuff to be done in parallel during the simulation. When implementing a new MPI-DAE sweeper multiple inheritance is used here

    >>> class fully_implicit_DAE_MPI(SweeperDAEMPI, generic_implicit_MPI, fully_implicit_DAE):

    Be careful with ordering of the class where the child class should inherit from! Due to multple inheritance several methods are overwritten here. In the example above (which is the class below) the class first inherits the methods 

    - compute_residual()
    - predict()
    - compute_end_point()

    from ``SweeperDAEMPI``. From second inherited class ``generic_implicit_MPI`` the methods

    - integrate()
    - update_nodes()

    are inherited, which then overwritten by the child class.

    Parameters
    ----------
    params : dict
        Parameters passed to the sweeper.
    """

    def __init__(self, params):
        super().__init__(params)

    def compute_residual(self, stage=None):
        r"""
        Uses the absolute value of the DAE system

        .. math::
            ||F(t, u, u')||

        for computing the residual in a chosen norm. If norm computes residual by using all values on collocation nodes, result is broadcasted to all processes; when norm only uses the value on last node, the result is collected on last process!

        Parameters
        ----------
        stage : str, optional
            The current stage of the step the level belongs to.
        """

        L = self.level
        P = L.prob

        # Check if we want to skip the residual computation to gain performance
        # Keep in mind that skipping any residual computation is likely to give incorrect outputs of the residual!
        if stage in self.params.skip_residual_computation:
            L.status.residual = 0.0 if L.status.residual is None else L.status.residual
            return None

        res = P.eval_f(L.u[self.rank + 1], L.f[self.rank + 1], L.time + L.dt * self.coll.nodes[self.rank])
        res_norm = abs(res)  # use abs function from data type here

        # find maximal residual over the nodes
        if L.params.residual_type == 'full_abs':
            L.status.residual = self.comm.allreduce(res_norm, op=MPI.MAX)
        elif L.params.residual_type == 'last_abs':
            L.status.residual = self.comm.bcast(res_norm, root=self.comm.size - 1)
        elif L.params.residual_type == 'full_rel':
            L.status.residual = self.comm.allreduce(res_norm / abs(L.u[0]), op=MPI.MAX)
        elif L.params.residual_type == 'last_rel':
            L.status.residual = self.comm.bcast(res_norm / abs(L.u[0]), root=self.comm.size - 1)
        else:
            raise NotImplementedError(f'residual type \"{L.params.residual_type}\" not implemented!')

        # indicate that the residual has seen the new values
        L.status.updated = False

        return None

    def predict(self):
        r"""
        Predictor to fill values at nodes before first sweep. It can decide whether the

            - initial condition is spread to each node ('initial_guess' = 'spread'),
            - or zero values are spread to each node ('initial_guess' = 'zero').

        Default prediction for sweepers, only copies values to all collocation nodes. This function
        overrides the base implementation by always initialising ``level.f`` to zero. This is necessary since
        ``level.f`` stores the solution derivative in the fully implicit case, which is not initially known.
        """

        L = self.level
        P = L.prob

        # set initial guess for gradient to zero
        L.f[0] = P.dtype_f(init=P.init, val=0.0)

        if self.params.initial_guess == 'spread':
            L.u[self.rank + 1] = P.dtype_u(L.u[0])
            L.f[self.rank + 1] = P.dtype_f(init=P.init, val=0.0)
        elif self.params.initial_guess == 'zero':
            L.u[self.rank + 1] = P.dtype_u(init=P.init, val=0.0)
            L.f[self.rank + 1] = P.dtype_f(init=P.init, val=0.0)
        else:
            raise ParameterError(f'initial_guess option {self.params.initial_guess} not implemented')

        # indicate that this level is now ready for sweeps
        L.status.unlocked = True
        L.status.updated = True

    def compute_end_point(self):
        """
        Compute u at the right point of the interval.

        The value uend computed here is a full evaluation of the Picard formulation unless do_full_update==False

        Returns
        -------
        None
        """

        L = self.level
        P = L.prob
        L.uend = P.dtype_u(P.init, val=0.0)

        # check if Mth node is equal to right point and do_coll_update is false, perform a simple copy
        if self.coll.right_is_node and not self.params.do_coll_update:
            # a copy is sufficient
            root = self.comm.Get_size() - 1
            if self.comm.rank == root:
                L.uend = P.dtype_u(L.u[-1])
            # broadcast diff parts and alg parts separately
            self.comm.Bcast(L.uend.diff, root=root)
            self.comm.Bcast(L.uend.alg, root=root)
        else:
            raise NotImplementedError()

        return None


class fully_implicit_DAE_MPI(SweeperDAEMPI, generic_implicit_MPI, fully_implicit_DAE):
    """
    Fully implicit DAE sweeper parallelized across the nodes.
    Please supply a communicator as `comm` to the parameters!
    """

    def integrate(self, last_only=False):
        r"""
        Integrates the gradient. ``me`` serves as buffer, and root process ``m`` stores
        the result of integral at node :math:`\tau_m` there.

        Parameters
        ----------
        last_only : bool, optional
            Integrate only the last node for the residual or all of them.

        Returns
        -------
        me : list of dtype_u
            Containing the integral as values.
        """

        L = self.level
        P = L.prob

        me = P.dtype_u(P.init, val=0.0)
        for m in [self.coll.num_nodes - 1] if last_only else range(self.coll.num_nodes):
            recvBufDiff = me.diff[:] if m == self.rank else None
            recvBufAlg = me.alg[:] if m == self.rank else None
            integral = L.dt * self.coll.Qmat[m + 1, self.rank + 1] * L.f[self.rank + 1]
            self.comm.Reduce(
                integral.diff, recvBufDiff, root=m, op=MPI.SUM
            )
            self.comm.Reduce(
                integral.alg, recvBufAlg, root=m, op=MPI.SUM
            )

        return me

    def update_nodes(self):
        r"""
        Updates values of ``u`` and ``f`` at collocation nodes. This correspond to a single iteration of the
        preconditioned Richardson iteration in **"ordinary"** SDC.
        """

        L = self.level
        P = L.prob

        # only if the level has been touched before
        assert L.status.unlocked

        integral = self.integrate()
        integral -= L.dt * self.QI[self.rank + 1, self.rank + 1] * L.f[self.rank + 1]
        integral += L.u[0]

        u_approx = P.dtype_u(integral)

        L.f[self.rank + 1][:] = P.solve_system(
            fully_implicit_DAE.F,
            u_approx,
            L.dt * self.QI[self.rank + 1, self.rank + 1],
            L.f[self.rank + 1],
            L.time + L.dt * self.coll.nodes[self.rank],
        )

        integral = self.integrate()
        L.u[self.rank + 1] = L.u[0] + integral
        # indicate presence of new values at this level
        L.status.updated = True

        return None
