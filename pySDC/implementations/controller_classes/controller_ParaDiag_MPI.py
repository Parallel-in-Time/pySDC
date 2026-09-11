import numpy as np
from mpi4py import MPI

from pySDC.core.errors import ControllerError
from pySDC.helpers.ParaDiagHelper import get_G_inv_matrix
from pySDC.implementations.controller_classes.ParaDiag import ParaDiag
from pySDC.implementations.controller_classes.controller_MPI import controller_MPI


class controller_ParaDiag_MPI(ParaDiag, controller_MPI):
    """
    ParaDiag controller with MPI parallelism across time steps: one step per rank.

    This is `controller_MPI` with a different iteration: where PFASST sweeps and cascades through
    the levels, ParaDiag diagonalizes across the steps. Everything around the iteration -- blocks,
    windowing, restarts, convergence -- is the driver it inherits, which is why the dispatcher it
    uses is still called `pfasst`.

    Everything here is written from a single processor's point of view. A rank owns exactly one step
    and never inspects another rank's data; the places where ParaDiag genuinely needs information
    from the whole block are expressed as communication:

    - ``prepare_Jacobians``          -> Allreduce(SUM) over the step communicator
    - ``compute_all_at_once_residual`` -> point-to-point exchange with the previous/next rank
    - ``apply_matrix`` (the weighted FFT/iFFT in time) -> a ring reduction, see below
    - convergence                   -> the inherited `it_check`, which allreduces because
                                       `all_to_done` is forced on for ParaDiag

    Note that ParaDiag steps can only converge together, so every rank always participates in every
    iteration. That is not a policy choice: a rank that stopped early would never enter the
    collectives below and the run would hang. `step_is_active` says the same thing about blocks --
    a block is never run partially, so the driver's windowing never splits one.
    """

    def __init__(self, controller_params, description, comm=None):
        """
        Args:
            controller_params: parameter set for the controller and the steps
            description: all the parameters to set up the rest (levels, problems, transfer, ...)
            comm: MPI communicator, one rank per time step
        """
        comm = MPI.COMM_WORLD if comm is None else comm
        self.prepare_ParaDiag_params(controller_params, description)

        self.sweeper_params = description['sweeper_params']

        # each step needs its own G^-1, determined by where it sits in the block
        self._G_inv_alpha = self.resolve_alpha(controller_params['alpha'], 0)
        description['sweeper_params']['G_inv'] = get_G_inv_matrix(
            comm.rank, comm.size, self._G_inv_alpha, description['sweeper_params']
        )

        super().__init__(controller_params, description, comm)

        self.n_steps = comm.size

        if len(self.S.levels) > 1:
            raise ControllerError('Multi-level SDC not implemented in ParaDiag!')

    # ------------------------------------------------------------------ collectives

    def apply_matrix(self, mat, quantity):
        """
        Apply a square L x L matrix across the steps, where L is the number of ranks.

        Each rank needs ``res_i = sum_j mat[i, j] * me_j`` but only holds ``me_i``. This is done as a
        ring reduction: the values circulate once around the communicator and each rank accumulates
        its own row as they pass. That keeps the working set at O(M) fields per rank, independent of
        L -- an allgather would instead need O(L * M) fields on every rank, which is exactly the
        gather this controller must not do.

        The ring costs L - 1 rounds. A butterfly would need only log2(L), but only because the matrix
        this is called with is a DFT; for a general matrix it needs the O(L * M) gather just ruled
        out. That belongs with an FFT-shaped interface rather than this one, and pays off from about
        L = 16 upwards.

        Args:
            mat: square matrix with as many rows as there are ranks
            quantity (str): 'residual' or 'increment', the level attribute to transform in place
        """
        comm = self.comm
        L, rank = comm.size, comm.rank
        assert np.allclose(mat.shape, L), f'need a {L}x{L} matrix, got {mat.shape}'

        lvl = self.S.levels[0]
        M = lvl.sweep.params.num_nodes
        prob = lvl.prob

        if quantity == 'residual':
            me = lvl.residual
        elif quantity == 'increment':
            me = lvl.increment
        else:
            raise NotImplementedError(f'Cannot apply matrix to {quantity!r}')

        res = [prob.u_init for _ in range(M)]
        # all M nodes travel in one contiguous buffer, so the ring costs L - 1 messages rather than
        # M * (L - 1). Same volume, M times fewer message latencies.
        held = np.array([me[m] for m in range(M)])
        buf = np.empty_like(held)

        nxt, prv = (rank + 1) % L, (rank - 1) % L
        for k in range(L):
            # after k rotations I am holding the value that started on rank (rank - k) % L
            src = (rank - k) % L
            for m in range(M):
                res[m] += mat[rank, src] * held[m]

            if k < L - 1:
                comm.Sendrecv(held, dest=nxt, sendtag=k, recvbuf=buf, source=prv, recvtag=k)
                held, buf = buf, held

        for m in range(M):
            me[m] = res[m]

    def prepare_Jacobians(self):
        """Average the solution across all steps, for constructing average Jacobians."""
        if not self.params.average_jacobian:
            return

        lvl = self.S.levels[0]
        M = lvl.sweep.coll.num_nodes

        u_avg = []
        for m in range(M):
            contribution = lvl.prob.dtype_u(lvl.u[m + 1])
            total = lvl.prob.dtype_u(lvl.prob.init, val=0)
            self.comm.Allreduce(contribution, total, op=MPI.SUM)
            u_avg.append(total / self.n_steps)

        lvl.u_avg = u_avg

    def compute_all_at_once_residual(self):
        """
        Compute the residual of the composite collocation problem.

        Needs the previous step's end point as this step's initial condition, which is the only
        point-to-point communication in a ParaDiag iteration.
        """
        S, comm = self.S, self.comm
        lvl = S.levels[0]

        lvl.sweep.compute_end_point()

        for hook in self.hooks:
            hook.pre_comm(step=S, level_number=0)

        req = None
        if not S.status.last:
            req = lvl.uend.isend(dest=(comm.rank + 1) % comm.size, tag=S.status.iter, comm=comm)
        if not S.status.first:
            lvl.u[0].irecv(source=(comm.rank - 1) % comm.size, tag=S.status.iter, comm=comm).Wait()
        if req is not None:
            req.Wait()

        for hook in self.hooks:
            hook.post_comm(step=S, level_number=0, add_to_stats=True)

        lvl.sweep.compute_residual()

    def update_G_inv(self, k=0):
        """
        Rebuild this rank's G^-1 if alpha changed with the iteration.

        Args:
            k (int): 0-based ParaDiag iteration index
        """
        alpha = self.get_alpha(k)
        if alpha == self._G_inv_alpha:
            return
        self._G_inv_alpha = alpha
        self.S.levels[0].sweep.set_G_inv(get_G_inv_matrix(self.comm.rank, self.comm.size, alpha, self.sweeper_params))

    def update_solution(self):
        """Add the increment to get the next iterate. Purely local."""
        lvl = self.S.levels[0]
        for m in range(lvl.sweep.coll.num_nodes):
            lvl.u[m + 1] += lvl.increment[m]

    # ------------------------------------------------------------------ the ParaDiag iteration

    def it_ParaDiag(self, comm, num_procs):
        """A single ParaDiag iteration, from this rank's point of view."""
        S = self.S

        for hook in self.hooks:
            hook.pre_sweep(step=S, level_number=0)

        # `it_check` has already incremented the counter, so the first sweep is k = 0
        k = max(S.status.iter - 1, 0)
        self.update_G_inv(k)

        self.prepare_Jacobians()
        self.compute_all_at_once_residual()

        self.FFT_in_time(quantity='residual', k=k)
        S.levels[0].sweep.update_nodes()  # local solve, embarrassingly parallel
        self.iFFT_in_time(quantity='increment', k=k)

        self.update_solution()

        for hook in self.hooks:
            hook.post_sweep(step=S, level_number=0)

        S.status.stage = 'IT_CHECK'
