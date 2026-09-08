import numpy as np

from pySDC.helpers.ParaDiagHelper import get_G_inv_matrix
from pySDC.implementations.controller_classes.ParaDiag import ParaDiag
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI


class controller_ParaDiag_nonMPI(ParaDiag, controller_nonMPI):
    """

    ParaDiag controller, running serialized version.

    This is `controller_nonMPI` with a different iteration: where PFASST sweeps and cascades through
    the levels, ParaDiag diagonalizes across the steps. Everything around the iteration -- blocks,
    windowing, restarts, convergence -- is the driver it inherits, which is why the dispatcher it
    uses is still called `pfasst`.

    This controller uses the increment formulation. That is to say, we setup the residual of the all at once problem,
    put it on the right hand side, invert the ParaDiag preconditioner on the left-hand side to compute the increment
    and then add the increment onto the solution. For this reason, we need to replace the solution values in the steps
    with the residual values before the solves and then put the solution plus increment back into the steps. This is a
    bit counter to what you expect when you access the `u` variable in the levels, but it is mathematically advantageous.
    """

    def __init__(self, num_procs, controller_params, description):
        """
        Initialization routine for ParaDiag controller

        Args:
           num_procs: number of parallel time steps (still serial, though), can be 1
           controller_params: parameter set for the controller and the steps
           description: all the parameters to set up the rest (levels, problems, transfer, ...)
        """
        self.prepare_ParaDiag_params(controller_params, description)

        self.sweeper_params = description['sweeper_params']
        self._G_inv_alpha = self.resolve_alpha(controller_params['alpha'], 0)

        # the steps are copies of the first one, so give that one its own G^-1 before they are built
        description['sweeper_params']['G_inv'] = get_G_inv_matrix(
            0, num_procs, self._G_inv_alpha, description['sweeper_params']
        )

        super().__init__(num_procs, controller_params, description)

        self.n_steps = num_procs

        if len(self.MS[0].levels) > 1:
            raise NotImplementedError('This controller does not support multiple levels')

        # every step but the first still has the first one's G^-1
        self.set_G_inv(self._G_inv_alpha)

    # ------------------------------------------------------------------ the ParaDiag iteration

    def apply_matrix(self, mat, quantity):
        """
        Apply a matrix on the step level. Needs to be square. Puts the result back into the controller.

        Args:
            mat: square LxL matrix with L number of steps
        """
        L = len(self.MS)
        assert np.allclose(mat.shape, L)
        assert len(mat.shape) == 2

        level = self.MS[0].levels[0]
        M = level.sweep.params.num_nodes
        prob = level.prob

        # buffer for storing the result
        res = [
            None,
        ] * L

        if quantity == 'residual':
            me = [S.levels[0].residual for S in self.MS]
        elif quantity == 'increment':
            me = [S.levels[0].increment for S in self.MS]
        else:
            raise NotImplementedError

        # compute matrix-vector product
        for i in range(mat.shape[0]):
            res[i] = [prob.u_init for _ in range(M)]
            for j in range(mat.shape[1]):
                for m in range(M):
                    res[i][m] += mat[i, j] * me[j][m]

        # put the result in the "output"
        for i in range(mat.shape[0]):
            for m in range(M):
                me[i][m] = res[i][m]

    def compute_all_at_once_residual(self, local_MS_running):
        """
        This requires to communicate the solutions at the end of the steps to be the initial conditions for the next
        steps. Afterwards, the residual can be computed locally on the steps.

        Args:
            local_MS_running (list): list of currently running steps
        """

        for S in local_MS_running:
            # communicate initial conditions
            S.levels[0].sweep.compute_end_point()

            for hook in self.hooks:
                hook.pre_comm(step=S, level_number=0)

            if not S.status.first:
                S.levels[0].u[0] = S.prev.levels[0].uend

            for hook in self.hooks:
                hook.post_comm(step=S, level_number=0, add_to_stats=True)

            # compute residuals locally
            S.levels[0].sweep.compute_residual()

    def set_G_inv(self, alpha):
        """
        Give every step the G^-1 that belongs to where it sits in the block.

        Args:
            alpha (float): the alpha this G^-1 is built from
        """
        L = len(self.MS)
        for l, S in enumerate(self.MS):
            S.levels[0].sweep.set_G_inv(get_G_inv_matrix(l, L, alpha, self.sweeper_params))

    def update_G_inv(self, k=0):
        """
        Rebuild G^-1 on every step if alpha changed with the iteration.

        Args:
            k (int): 0-based ParaDiag iteration index
        """
        alpha = self.get_alpha(k)
        if alpha == self._G_inv_alpha:
            return
        self._G_inv_alpha = alpha
        self.set_G_inv(alpha)

    def update_solution(self, local_MS_running):
        """
        Since we solve for the increment, we need to update the solution between iterations by adding the increment.

        Args:
            local_MS_running (list): list of currently running steps
        """
        for S in local_MS_running:
            for m in range(S.levels[0].sweep.coll.num_nodes):
                S.levels[0].u[m + 1] += S.levels[0].increment[m]

    def prepare_Jacobians(self, local_MS_running):
        # get solutions for constructing average Jacobians
        if self.params.average_jacobian:
            level = local_MS_running[0].levels[0]
            M = level.sweep.coll.num_nodes

            u_avg = [level.prob.dtype_u(level.prob.init, val=0)] * M

            # communicate average solution
            for S in local_MS_running:
                for m in range(M):
                    u_avg[m] += S.levels[0].u[m + 1] / self.n_steps

            # store the averaged solution in the steps
            for S in local_MS_running:
                S.levels[0].u_avg = u_avg

    def it_ParaDiag(self, local_MS_running):
        """
        Do a single ParaDiag iteration. Does the following steps
         - (1) Compute the residual of the all-at-once / composite collocation problem
         - (2) Compute an FFT in time to diagonalize the preconditioner
         - (3) Solve the collocation problems locally on the steps for the increment
         - (4) Compute iFFT in time to go back to the original base
         - (5) Update the solution by adding increment

        Note that this is the only place where we compute the all-at-once residual because it requires communication and
        swaps the solution values for the residuals. So after the residual tolerance is reached, one more ParaDiag
        iteration will be done.

        Args:
            local_MS_running (list): list of currently running steps
        """

        for S in local_MS_running:
            for hook in self.hooks:
                hook.pre_sweep(step=S, level_number=0)

        # `it_check` has already incremented the counter, so the first sweep is k = 0
        k = max(local_MS_running[0].status.iter - 1, 0)
        self.update_G_inv(k)

        # communicate average residual for setting up Jacobians for non-linear problems
        self.prepare_Jacobians(local_MS_running)

        # compute the all-at-once residual to use as right hand side
        self.compute_all_at_once_residual(local_MS_running)

        # weighted FFT of the residual in time
        self.FFT_in_time(quantity='residual', k=k)

        # perform local solves of "collocation problems" on the steps (can be done in parallel)
        for S in local_MS_running:
            assert len(S.levels) == 1, 'Multi-level SDC not implemented in ParaDiag'
            S.levels[0].sweep.update_nodes()

        # inverse FFT of the increment in time
        self.iFFT_in_time(quantity='increment', k=k)

        # get the next iterate by adding increment to previous iterate
        self.update_solution(local_MS_running)

        for S in local_MS_running:
            for hook in self.hooks:
                hook.post_sweep(step=S, level_number=0)

        # update stage
        for S in local_MS_running:
            S.status.stage = 'IT_CHECK'
