import numpy as np
from mpi4py import MPI

from pySDC.core.controller import ParaDiagController
from pySDC.core.errors import ControllerError
from pySDC.core.step import Step
from pySDC.helpers.ParaDiagHelper import get_G_inv_matrix
from pySDC.implementations.convergence_controller_classes.basic_restarting import BasicRestarting


class controller_ParaDiag_MPI(ParaDiagController):
    """
    ParaDiag controller with MPI parallelism across time steps: one step per rank.

    Everything here is written from a single processor's point of view. A rank owns exactly one step
    and never inspects another rank's data; the places where ParaDiag genuinely needs information
    from the whole block are expressed as communication:

    - ``prepare_Jacobians``          -> Allreduce(SUM) over the step communicator
    - ``compute_all_at_once_residual`` -> point-to-point exchange with the previous/next rank
    - ``apply_matrix`` (the weighted FFT/iFFT in time) -> a ring reduction, see below
    - convergence                   -> allreduce(LAND), because ParaDiag converges collectively

    Note that ParaDiag steps can only converge together, so every rank always participates in every
    iteration. That is not a policy choice: a rank that stopped early would never enter the
    collectives below and the run would hang.
    """

    def __init__(self, controller_params, description, comm=None):
        """
        Args:
            controller_params: parameter set for the controller and the steps
            description: all the parameters to set up the rest (levels, problems, transfer, ...)
            comm: MPI communicator, one rank per time step
        """
        comm = MPI.COMM_WORLD if comm is None else comm
        super().__init__(controller_params=controller_params, description=description, n_steps=comm.size, useMPI=True)

        self.comm = comm

        # each step needs its own G^-1, determined by where it sits in the block
        description['sweeper_params']['G_inv'] = get_G_inv_matrix(
            comm.rank, comm.size, self.params.alpha, description['sweeper_params']
        )
        self.S = Step(description)
        self.S.status.time_size = comm.size

        self.base_convergence_controllers += [BasicRestarting.get_implementation(useMPI=True)]
        for convergence_controller in self.base_convergence_controllers:
            self.add_convergence_controller(convergence_controller, description)

        if len(self.S.levels) > 1:
            raise ControllerError('Multi-level SDC not implemented in ParaDiag!')

        if self.params.dump_setup and comm.rank == 0:
            self.dump_setup(step=self.S, controller_params=controller_params, description=description)

        for C in [self.convergence_controllers[i] for i in self.convergence_controller_order]:
            C.setup_status_variables(self, comm=comm)

    # ------------------------------------------------------------------ collectives

    def apply_matrix(self, mat, quantity):
        """
        Apply a square L x L matrix across the steps, where L is the number of ranks.

        Each rank needs ``res_i = sum_j mat[i, j] * me_j`` but only holds ``me_i``. This is done as a
        ring reduction: the values circulate once around the communicator and each rank accumulates
        its own row as they pass. That keeps the working set at O(M) fields per rank, independent of
        L -- an allgather would instead need O(L * M) fields on every rank, which is exactly the
        gather this controller must not do.

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
        held = [prob.dtype_u(me[m]) for m in range(M)]
        buf = [prob.dtype_u(me[m]) for m in range(M)]

        nxt, prv = (rank + 1) % L, (rank - 1) % L
        for k in range(L):
            # after k rotations I am holding the value that started on rank (rank - k) % L
            src = (rank - k) % L
            for m in range(M):
                res[m] += mat[rank, src] * held[m]

            if k < L - 1:
                for m in range(M):
                    comm.Sendrecv(held[m], dest=nxt, sendtag=k, recvbuf=buf[m], source=prv, recvtag=k)
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

    def update_solution(self):
        """Add the increment to get the next iterate. Purely local."""
        lvl = self.S.levels[0]
        for m in range(lvl.sweep.coll.num_nodes):
            lvl.u[m + 1] += lvl.increment[m]

    # ------------------------------------------------------------------ stages

    def spread(self):
        """Spreading phase"""
        S = self.S
        for hook in self.hooks:
            hook.pre_step(step=S, level_number=0)

        S.levels[0].sweep.predict()
        S.levels[0].sweep.compute_residual()
        S.status.stage = 'IT_CHECK'

        for C in [self.convergence_controllers[i] for i in self.convergence_controller_order]:
            C.post_spread_processing(self, S, comm=self.comm)

    def it_check(self):
        """Check for convergence. ParaDiag converges collectively, hence the allreduce."""
        S, comm = self.S, self.comm

        if S.status.iter > 0:
            for hook in self.hooks:
                hook.post_iteration(step=S, level_number=0)

        for C in [self.convergence_controllers[i] for i in self.convergence_controller_order]:
            C.post_iteration_processing(self, S, comm=comm)
            C.convergence_control(self, S, comm=comm)

        for hook in self.hooks:
            hook.pre_comm(step=S, level_number=0)
        S.status.done = comm.allreduce(S.status.done, op=MPI.LAND)
        for hook in self.hooks:
            hook.post_comm(step=S, level_number=0, add_to_stats=True)

        if not S.status.done:
            S.status.iter += 1
            for hook in self.hooks:
                hook.pre_iteration(step=S, level_number=0)
            for C in [self.convergence_controllers[i] for i in self.convergence_controller_order]:
                C.pre_iteration_processing(self, S, comm=comm)
            S.status.stage = 'IT_PARADIAG'
        else:
            S.levels[0].sweep.compute_end_point()
            for hook in self.hooks:
                hook.post_step(step=S, level_number=0)
            S.status.stage = 'DONE'

    def it_ParaDiag(self):
        """A single ParaDiag iteration, from this rank's point of view."""
        S = self.S

        for hook in self.hooks:
            hook.pre_sweep(step=S, level_number=0)

        self.prepare_Jacobians()
        self.compute_all_at_once_residual()

        self.FFT_in_time(quantity='residual')
        S.levels[0].sweep.update_nodes()  # local solve, embarrassingly parallel
        self.iFFT_in_time(quantity='increment')

        self.update_solution()

        for hook in self.hooks:
            hook.post_sweep(step=S, level_number=0)

        S.status.stage = 'IT_CHECK'

    def ParaDiag(self):
        """Dispatch on the current stage. All ranks are always in the same stage."""
        stage = self.S.status.stage
        self.logger.debug(stage)

        switcher = {'SPREAD': self.spread, 'IT_CHECK': self.it_check, 'IT_PARADIAG': self.it_ParaDiag}
        assert stage in switcher, f'Got unexpected stage {stage!r}'
        switcher[stage]()

        return self.S.status.done

    # ------------------------------------------------------------------ driver

    def restart_block(self, time, u0):
        """Reset this rank's step for a new block."""
        S, comm = self.S, self.comm

        S.status.slot = comm.rank
        S.reset_step()
        S.status.first = comm.rank == 0
        S.status.last = comm.rank == comm.size - 1
        S.init_step(u0)
        S.status.done = False
        S.status.prev_done = False
        S.status.iter = 0
        S.status.stage = 'SPREAD'
        S.status.force_done = False
        S.status.time_size = comm.size

        for lvl in S.levels:
            lvl.tag = None
            lvl.status.sweep = 1
            lvl.status.time = time

        for C in [self.convergence_controllers[i] for i in self.convergence_controller_order]:
            C.reset_status_variables(self, comm=comm)

    def run(self, u0, t0, Tend):
        """
        Main driver.

        ParaDiag always runs whole blocks -- every rank participates in every iteration -- so unlike
        the pipelined controllers there is no communicator splitting as steps finish.
        """
        comm = self.comm
        for hook in self.hooks:
            hook.reset_stats()

        all_dt = comm.allgather(self.S.dt)
        block_dt = sum(all_dt)
        time = t0 + sum(all_dt[: comm.rank])
        block_start = t0

        if block_start >= Tend - 10 * np.finfo(float).eps:
            raise ControllerError('Nothing to do, check t0, dt and Tend!')

        if block_start + block_dt > Tend + 10 * np.finfo(float).eps and comm.rank == 0:
            self.logger.warning(
                'Warning: This controller will solve past your desired end time until the end of its block!'
            )

        self.restart_block(time, u0)
        uend = u0

        for hook in self.hooks:
            hook.post_setup(step=None, level_number=None)
        for hook in self.hooks:
            hook.pre_run(step=self.S, level_number=0)

        while block_start < Tend - 10 * np.finfo(float).eps:
            while not self.ParaDiag():
                pass

            uend = self.S.levels[0].uend.bcast(root=comm.size - 1, comm=comm)
            tend = comm.bcast(self.S.time + self.S.dt, root=comm.size - 1)

            for C in [self.convergence_controllers[i] for i in self.convergence_controller_order]:
                C.post_step_processing(self, self.S, comm=comm)
                C.prepare_next_block(self, self.S, self.S.status.time_size, tend, Tend, comm=comm)

            block_start = tend
            if block_start < Tend - 10 * np.finfo(float).eps:
                all_dt = comm.allgather(self.S.dt)
                time = block_start + sum(all_dt[: comm.rank])
                self.restart_block(time, uend)

        for hook in self.hooks:
            hook.post_run(step=self.S, level_number=0)
        for C in [self.convergence_controllers[i] for i in self.convergence_controller_order]:
            C.post_run_processing(self, self.S, comm=comm)

        return uend, self.return_stats()
