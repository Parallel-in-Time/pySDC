import numpy as np

from pySDC.core.convergence_controller import ConvergenceController
from pySDC.implementations.convergence_controller_classes.store_uold import StoreUOld


class CheckIterationEstimator(ConvergenceController):
    r"""
    Stop iterating once an estimate says the desired tolerance has been reached.

    The estimate is the standard contraction argument. With :math:`d_k` the largest change between
    the last two iterates,

    .. math::
        \tilde{L} = \min\left(\frac{d_k}{d_{k-1}}, 0.9\right), \quad
        \alpha = \frac{d_1}{1 - \tilde{L}}, \quad
        K = 1.05 \frac{\log(\text{errtol} / \alpha)}{\log \tilde{L}},

    and once :math:`\lceil K \rceil` is no larger than the current iteration the block stops.

    :math:`d_k` is the largest change anywhere in the block, so every step reaches the same verdict at
    the same iteration. That maximum is the only thing here that is not step-local, and it is all the
    two transports do differently: an ``MPI_Allreduce`` with ``MPI_MAX`` when there is one step per
    rank, and a maximum over the steps when the block is in one process.

    Taking it over the whole block rather than over the steps up to the one asking is what makes the
    two agree. Checking convergence is a pass over the steps in order, so a verdict reached at the
    last step arrives too late for the steps already visited, and the block would run an extra
    iteration that the same problem on one step per rank does not -- a collective reaches every rank
    at once, a loop does not. The maximum is therefore taken on the first step visited, which is also
    the only moment it can be taken: `StoreUOld` overwrites ``uold`` step by step as the block is
    walked, so by the last step the earlier ones can no longer say how far they moved.

    Everything else lives on the steps rather than on this object, which is what lets one class serve
    both transports -- an object here is shared by a whole block in one case and private to a rank in
    the other.
    """

    def setup(self, controller, params, description, **kwargs):
        """
        Define default parameters here.

        Args:
            controller (pySDC.Controller): The controller
            params (dict): Parameters for the convergence controller
            description (dict): The description object used to instantiate the controller

        Returns:
            dict: The updated parameters
        """
        return {'control_order': -50, **super().setup(controller, params, description, **kwargs)}

    def check_parameters(self, controller, params, description, **kwargs):
        """
        Check whether we have a tolerance to aim for, and whether the run is shaped so we can.

        Args:
            controller (pySDC.Controller): The controller
            params (dict): The params passed for this specific convergence controller
            description (dict): The description object used to instantiate the controller

        Returns:
            bool: Whether the parameters are compatible
            str: The error message
        """
        if 'errtol' not in params.keys():
            return False, 'Please give the iteration estimator a tolerance in the form of `errtol`. Thanks!'

        if self.params.useMPI and not controller.params.all_to_done:
            return False, (
                'The iteration estimator predicts when the whole block has converged and stops it in '
                'one go, so every rank has to take part in every iteration for the running maximum to '
                'be well defined. Please set `all_to_done = True` in the controller parameters.'
            )

        return True, ''

    def dependencies(self, controller, description, **kwargs):
        """
        Need the solution of the previous iteration, and a maximum under MPI.

        Args:
            controller (pySDC.Controller): The controller
            description (dict): The description object used to instantiate the controller

        Returns:
            None
        """
        controller.add_convergence_controller(StoreUOld, description=description)

        if self.params.useMPI:
            from mpi4py import MPI

            self.MPI_MAX = MPI.MAX

        super().dependencies(controller, description, **kwargs)
        return None

    @staticmethod
    def local_difference(S):
        """
        How far this step moved in the last iteration.

        Args:
            S (pySDC.Step): The step to measure

        Returns:
            float: the largest change across the collocation nodes
        """
        L = S.levels[0]
        return max(abs(L.uold[m] - L.u[m]) for m in range(1, L.sweep.coll.num_nodes + 1))

    def setup_status_variables(self, controller, **kwargs):
        """
        Store the differences on the steps, where they belong.

        Args:
            controller (pySDC.Controller): The controller

        Returns:
            None
        """
        self.add_status_variable_to_step('diff_local', 0.0)
        self.add_status_variable_to_step('diff_block', 0.0)
        self.add_status_variable_to_step('diff_old', 0.0)
        self.add_status_variable_to_step('diff_first', 0.0)
        return None

    def check_iteration_status(self, controller, S, comm=None, **kwargs):
        """
        Estimate the number of iterations still needed and stop the block if it has had enough.

        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.Step): The current step
            comm (mpi4py.MPI.Intracomm): Communicator, or None when the block is in one process

        Returns:
            None
        """
        L = S.levels[0]
        S.status.diff_local = self.local_difference(S)

        if comm is not None:
            diff_new = comm.allreduce(S.status.diff_local, op=self.MPI_MAX)
        else:
            block = kwargs.get('MS', controller.steps)
            # Every step has to reach the same verdict in the same pass, so the maximum is taken over
            # the whole block and taken once. It has to be taken here, on the first step visited,
            # because `StoreUOld` overwrites `uold` step by step as the block is walked -- by the time
            # the last step is visited the earlier ones can no longer say how far they moved. Handing
            # the answer to every step keeps it off this object, which a whole block shares.
            if S.status.slot == block[0].status.slot:
                diff_block = max(self.local_difference(T) for T in block)
                for T in block:
                    T.status.diff_block = diff_block
            diff_new = S.status.diff_block

        if S.status.iter == 1:
            S.status.diff_old = diff_new
            S.status.diff_first = diff_new
        elif S.status.iter > 1:
            Ltilde_loc = min(diff_new / S.status.diff_old, 0.9)
            S.status.diff_old = diff_new
            alpha = 1 / (1 - Ltilde_loc) * S.status.diff_first
            Kest_loc = np.log(self.params.errtol / alpha) / np.log(Ltilde_loc) * 1.05  # Safety factor!
            self.debug(
                f'LOCAL: {L.time:8.4f}, {S.status.iter}: {int(np.ceil(Kest_loc))}, '
                f'{Ltilde_loc:8.6e}, {Kest_loc:8.6e}, {Ltilde_loc ** S.status.iter * alpha:8.6e}',
                S,
            )

            # every step saw the same maximum, so every step reaches the same verdict
            if np.ceil(Kest_loc) <= S.status.iter:
                S.status.force_done = True

        return None


# the estimator used to be available only with the whole block in one process, and was named for it
CheckIterationEstimatorNonMPI = CheckIterationEstimator
