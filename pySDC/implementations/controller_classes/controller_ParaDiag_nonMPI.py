import itertools
import numpy as np

from pySDC.core.controller import ParaDiagController
from pySDC.core import step as stepclass
from pySDC.core.errors import ControllerError
from pySDC.implementations.convergence_controller_classes.basic_restarting import BasicRestarting
from pySDC.helpers.ParaDiagHelper import get_G_inv_matrix


class controller_ParaDiag_nonMPI(ParaDiagController):
    """

    ParaDiag controller, running serialized version.

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
        super().__init__(controller_params, description, useMPI=False, n_steps=num_procs)

        self.MS = []

        for l in range(num_procs):
            G_inv = get_G_inv_matrix(l, num_procs, self.params.alpha, description['sweeper_params'])
            description['sweeper_params']['G_inv'] = G_inv

            self.MS.append(stepclass.Step(description))

        self.base_convergence_controllers += [BasicRestarting.get_implementation(useMPI=False)]
        for convergence_controller in self.base_convergence_controllers:
            self.add_convergence_controller(convergence_controller, description)

        if self.params.dump_setup:
            self.dump_setup(step=self.MS[0], controller_params=controller_params, description=description)

        if len(self.MS[0].levels) > 1:
            raise NotImplementedError('This controller does not support multiple levels')

        for C in [self.convergence_controllers[i] for i in self.convergence_controller_order]:
            C.reset_buffers_nonMPI(self)
            C.setup_status_variables(self, MS=self.MS)

    def ParaDiag(self, local_MS_active):
        """
        Main function for ParaDiag

        For the workflow of this controller, see https://arxiv.org/abs/2103.12571

        This method changes self.MS directly by accessing active steps through local_MS_active.

        Args:
            local_MS_active (list): all active steps

        Returns:
            boot: Whether all steps are done
        """

        # if all stages are the same (or DONE), continue, otherwise abort
        stages = [S.status.stage for S in local_MS_active if S.status.stage != 'DONE']
        if stages[1:] == stages[:-1]:
            stage = stages[0]
        else:
            raise ControllerError('not all stages are equal')

        self.logger.debug(stage)

        MS_running = [S for S in local_MS_active if S.status.stage != 'DONE']

        switcher = {
            'SPREAD': self.spread,
            'IT_CHECK': self.it_check,
            'IT_PARADIAG': self.it_ParaDiag,
        }

        assert stage in switcher.keys(), f'Got unexpected stage {stage!r}'
        switcher[stage](MS_running)

        return all(S.status.done for S in local_MS_active)

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

        # communicate average residual for setting up Jacobians for non-linear problems
        self.prepare_Jacobians(local_MS_running)

        # compute the all-at-once residual to use as right hand side
        self.compute_all_at_once_residual(local_MS_running)

        # weighted FFT of the residual in time
        self.FFT_in_time(quantity='residual')

        # perform local solves of "collocation problems" on the steps (can be done in parallel)
        for S in local_MS_running:
            assert len(S.levels) == 1, 'Multi-level SDC not implemented in ParaDiag'
            S.levels[0].sweep.update_nodes()

        # inverse FFT of the increment in time
        self.iFFT_in_time(quantity='increment')

        # get the next iterate by adding increment to previous iterate
        self.update_solution(local_MS_running)

        for S in local_MS_running:
            for hook in self.hooks:
                hook.post_sweep(step=S, level_number=0)

        # update stage
        for S in local_MS_running:
            S.status.stage = 'IT_CHECK'

    def it_check(self, local_MS_running):
        """
        Key routine to check for convergence/termination

        Args:
            local_MS_running (list): list of currently running steps
        """

        for S in local_MS_running:
            if S.status.iter > 0:
                for hook in self.hooks:
                    hook.post_iteration(step=S, level_number=0)

            # decide if the step is done, needs to be restarted and other things convergence related
            for C in [self.convergence_controllers[i] for i in self.convergence_controller_order]:
                C.post_iteration_processing(self, S, MS=local_MS_running)
                C.convergence_control(self, S, MS=local_MS_running)

        for S in local_MS_running:
            if not S.status.first:
                for hook in self.hooks:
                    hook.pre_comm(step=S, level_number=0)
                S.status.prev_done = S.prev.status.done  # "communicate"
                for hook in self.hooks:
                    hook.post_comm(step=S, level_number=0, add_to_stats=True)
                S.status.done = S.status.done and S.status.prev_done

            if self.params.all_to_done:
                for hook in self.hooks:
                    hook.pre_comm(step=S, level_number=0)
                S.status.done = all(T.status.done for T in local_MS_running)
                for hook in self.hooks:
                    hook.post_comm(step=S, level_number=0, add_to_stats=True)

            if not S.status.done:
                # increment iteration count here (and only here)
                S.status.iter += 1
                for hook in self.hooks:
                    hook.pre_iteration(step=S, level_number=0)
                for C in [self.convergence_controllers[i] for i in self.convergence_controller_order]:
                    C.pre_iteration_processing(self, S, MS=local_MS_running)

                # Do another ParaDiag iteration
                S.status.stage = 'IT_PARADIAG'
            else:
                S.levels[0].sweep.compute_end_point()
                for hook in self.hooks:
                    hook.post_step(step=S, level_number=0)
                S.status.stage = 'DONE'

        for C in [self.convergence_controllers[i] for i in self.convergence_controller_order]:
            C.reset_buffers_nonMPI(self)

    def spread(self, local_MS_running):
        """
        Spreading phase

        Args:
            local_MS_running (list): list of currently running steps
        """

        for S in local_MS_running:

            # first stage: spread values
            for hook in self.hooks:
                hook.pre_step(step=S, level_number=0)

            # call predictor from sweeper
            S.levels[0].sweep.predict()

            # compute the residual
            S.levels[0].sweep.compute_residual()

            # update stage
            S.status.stage = 'IT_CHECK'

            for C in [self.convergence_controllers[i] for i in self.convergence_controller_order]:
                C.post_spread_processing(self, S, MS=local_MS_running)

    def run(self, u0, t0, Tend):
        """
        Main driver for running the serial version of ParaDiag

        Args:
           u0: initial values
           t0: starting time
           Tend: ending time

        Returns:
            end values on the last step
            stats object containing statistics for each step, each level and each iteration
        """

        # some initializations and reset of statistics
        uend = None
        num_procs = len(self.MS)
        for hook in self.hooks:
            hook.reset_stats()

        # initial ordering of the steps: 0,1,...,Np-1
        slots = list(range(num_procs))

        # initialize time variables of each step
        time = [t0 + sum(self.MS[j].dt for j in range(p)) for p in slots]

        # determine which steps are still active (time < Tend)
        active = [time[p] < Tend - 10 * np.finfo(float).eps for p in slots]
        if not all(active) and any(active):
            self.logger.warning(
                'Warning: This controller will solve past your desired end time until the end of its block!'
            )
            active = [
                True,
            ] * len(active)

        if not any(active):
            raise ControllerError('Nothing to do, check t0, dt and Tend.')

        # compress slots according to active steps, i.e. remove all steps which have times above Tend
        active_slots = list(itertools.compress(slots, active))

        # initialize block of steps with u0
        self.restart_block(active_slots, time, u0)

        for hook in self.hooks:
            hook.post_setup(step=None, level_number=None)

        # call pre-run hook
        for S in self.MS:
            for hook in self.hooks:
                hook.pre_run(step=S, level_number=0)

        # main loop: as long as at least one step is still active (time < Tend), do something
        while any(active):
            MS_active = [self.MS[p] for p in active_slots]
            done = False
            while not done:
                done = self.ParaDiag(MS_active)

            restarts = [S.status.restart for S in MS_active]
            restart_at = np.where(restarts)[0][0] if True in restarts else len(MS_active)
            if True in restarts:  # restart part of the block
                # initial condition to next block is initial condition of step that needs restarting
                uend = self.MS[restart_at].levels[0].u[0]
                time[active_slots[0]] = time[restart_at]
                self.logger.info(f'Starting next block with initial conditions from step {restart_at}')

            else:  # move on to next block
                # initial condition for next block is last solution of current block
                uend = self.MS[active_slots[-1]].levels[0].uend
                time[active_slots[0]] = time[active_slots[-1]] + self.MS[active_slots[-1]].dt

            for S in MS_active[:restart_at]:
                for C in [self.convergence_controllers[i] for i in self.convergence_controller_order]:
                    C.post_step_processing(self, S, MS=MS_active)

            for C in [self.convergence_controllers[i] for i in self.convergence_controller_order]:
                [C.prepare_next_block(self, S, len(active_slots), time, Tend, MS=MS_active) for S in self.MS]

            # setup the times of the steps for the next block
            for i in range(1, len(active_slots)):
                time[active_slots[i]] = time[active_slots[i] - 1] + self.MS[active_slots[i] - 1].dt

            # determine new set of active steps and compress slots accordingly
            active = [time[p] < Tend - 10 * np.finfo(float).eps for p in slots]
            if not all(active) and any(active):
                self.logger.warning(
                    'Warning: This controller will solve past your desired end time until the end of its block!'
                )
                active = [
                    True,
                ] * len(active)
            active_slots = list(itertools.compress(slots, active))

            # restart active steps (reset all values and pass uend to u0)
            self.restart_block(active_slots, time, uend)

        # call post-run hook
        for S in self.MS:
            for hook in self.hooks:
                hook.post_run(step=S, level_number=0)

        for S in self.MS:
            for C in [self.convergence_controllers[i] for i in self.convergence_controller_order]:
                C.post_run_processing(self, S, MS=MS_active)

        return uend, self.return_stats()

    def restart_block(self, active_slots, time, u0):
        """
        Helper routine to reset/restart block of (active) steps

        Args:
            active_slots: list of active steps
            time: list of new times
            u0: initial value to distribute across the steps

        """

        for j in range(len(active_slots)):
            # get slot number
            p = active_slots[j]

            # store current slot number for diagnostics
            self.MS[p].status.slot = p
            # store link to previous step
            self.MS[p].prev = self.MS[active_slots[j - 1]]

            self.MS[p].reset_step()

            # determine whether I am the first and/or last in line
            self.MS[p].status.first = active_slots.index(p) == 0
            self.MS[p].status.last = active_slots.index(p) == len(active_slots) - 1

            # initialize step with u0
            self.MS[p].init_step(u0)

            # setup G^{-1} for new number of active slots
            # self.MS[j].levels[0].sweep.set_G_inv(get_G_inv_matrix(j, len(active_slots), self.params.alpha, self.description['sweeper_params']))

            # reset some values
            self.MS[p].status.done = False
            self.MS[p].status.prev_done = False
            self.MS[p].status.iter = 0
            self.MS[p].status.stage = 'SPREAD'
            self.MS[p].status.force_done = False
            self.MS[p].status.time_size = len(active_slots)

            for l in self.MS[p].levels:
                l.tag = None
                l.status.sweep = 1

        for p in active_slots:
            for lvl in self.MS[p].levels:
                lvl.status.time = time[p]

        for C in [self.convergence_controllers[i] for i in self.convergence_controller_order]:
            C.reset_status_variables(self, active_slots=active_slots)
