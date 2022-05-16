import numpy as np
import itertools
from pySDC.core.Errors import ControllerError, ParameterError
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from error_estimator import ErrorEstimator_nonMPI


class controller_nonMPI_resilient(controller_nonMPI):
    """

    PFASST controller, running serialized version of PFASST in blocks (MG-style)

    """

    def __init__(self, num_procs, controller_params, description):
        # additional parameters
        controller_params['use_embedded_estimate'] = description['error_estimator_params'].get('use_embedded_estimate',
                                                                                               False)
        controller_params['HotRod'] = controller_params.get('HotRod', False)
        description['error_estimator_params']['HotRod'] = controller_params.get('HotRod', False)
        description['error_estimator_params']['use_adaptivity'] = controller_params.get('use_adaptivity', False)

        # additional parameters
        if controller_params['HotRod']:
            controller_params['HotRod_tol'] = controller_params.get('HotRod_tol', np.inf)
        controller_params['use_adaptivity'] = controller_params.get('use_adaptivity', False)
        if controller_params['use_adaptivity']:
            if 'e_tol' not in description['level_params'].keys():
                raise ParameterError('Please supply "e_tol" in the level parameters')
            if 'restol' in description['level_params'].keys():
                if description['level_params']['restol'] > 0:
                    description['level_params']['restol'] = 0
                    print(f'I want to do always maxiter={description["step_params"]["maxiter"]} iterations to have a consta\
nt order in time for adaptivity. Setting restol=0')

        # call parent's initialization routine
        super(controller_nonMPI_resilient, self).__init__(num_procs, controller_params, description)

        self.error_estimator = ErrorEstimator_nonMPI(self, description.get('error_estimator_params', {}))
        self.store_uold = self.params.use_iteration_estimator or self.params.use_embedded_estimate or self.params.HotRod
        self.restart = [False] * num_procs

    def it_check(self, local_MS_running):
        """
        Key routine to check for convergence/termination

        Args:
            local_MS_running (list): list of currently running steps
        """

        for S in local_MS_running:

            # send updated values forward
            self.hooks.pre_comm(step=S, level_number=0)
            if not S.status.last:
                self.logger.debug('Process %2i provides data on level %2i with tag %s'
                                  % (S.status.slot, 0, S.status.iter))
                self.send(S.levels[0], tag=(0, S.status.iter, S.status.slot))

            # receive values
            if not S.status.prev_done and not S.status.first:
                self.logger.debug('Process %2i receives from %2i on level %2i with tag %s' %
                                  (S.status.slot, S.prev.status.slot, 0, S.status.iter))
                self.recv(S.levels[0], S.prev.levels[0], tag=(0, S.status.iter, S.prev.status.slot))
            self.hooks.post_comm(step=S, level_number=0)

            S.levels[0].sweep.compute_residual()

        if self.params.use_iteration_estimator:
            self.check_iteration_estimator(local_MS_running)

        self.error_estimator.estimate(local_MS_running)

        if self.params.use_adaptivity:
            self.adaptivity(local_MS_running)

        self.resilence(local_MS_running)

        for S in local_MS_running:

            S.status.done = self.check_convergence(S)

            if S.status.iter > 0:
                self.hooks.post_iteration(step=S, level_number=0)

        for S in local_MS_running:
            if not S.status.first:
                self.hooks.pre_comm(step=S, level_number=0)
                S.status.prev_done = S.prev.status.done  # "communicate"
                self.hooks.post_comm(step=S, level_number=0, add_to_stats=True)
                S.status.done = S.status.done and S.status.prev_done

            if self.params.all_to_done:
                self.hooks.pre_comm(step=S, level_number=0)
                S.status.done = all([T.status.done for T in local_MS_running])
                self.hooks.post_comm(step=S, level_number=0, add_to_stats=True)

            if not S.status.done:
                # increment iteration count here (and only here)
                S.status.iter += 1
                self.hooks.pre_iteration(step=S, level_number=0)

                if self.store_uold:
                    # store pervious iterate to compute difference later on
                    S.levels[0].uold[:] = S.levels[0].u[:]

                if len(S.levels) > 1:  # MLSDC or PFASST
                    S.status.stage = 'IT_DOWN'
                else:  # SDC or MSSDC
                    if len(local_MS_running) == 1 or self.params.mssdc_jac:  # SDC or parallel MSSDC (Jacobi-like)
                        S.status.stage = 'IT_FINE'
                    else:
                        S.status.stage = 'IT_COARSE'  # serial MSSDC (Gauss-like)
            else:
                S.levels[0].sweep.compute_end_point()
                self.hooks.post_step(step=S, level_number=0)
                S.status.stage = 'DONE'

    def run(self, u0, t0, Tend):
        """
        Main driver for running the serial version of SDC, MSSDC, MLSDC and PFASST (virtual parallelism)

        Args:
           u0: initial values
           t0: starting time
           Tend: ending time

        Returns:
            end values on the finest level
            stats object containing statistics for each step, each level and each iteration
        """

        # some initializations and reset of statistics
        uend = None
        num_procs = len(self.MS)
        self.hooks.reset_stats()

        # initial ordering of the steps: 0,1,...,Np-1
        slots = list(range(num_procs))

        # initialize time variables of each step
        time = [t0 + sum(self.MS[j].dt for j in range(p)) for p in slots]

        # determine which steps are still active (time < Tend)
        active = [time[p] < Tend - 10 * np.finfo(float).eps for p in slots]

        if not any(active):
            raise ControllerError('Nothing to do, check t0, dt and Tend.')

        # compress slots according to active steps, i.e. remove all steps which have times above Tend
        active_slots = list(itertools.compress(slots, active))

        # initialize block of steps with u0
        self.restart_block(active_slots, time, u0)

        self.hooks.post_setup(step=None, level_number=None)

        # call pre-run hook
        for S in self.MS:
            self.hooks.pre_run(step=S, level_number=0)

        # main loop: as long as at least one step is still active (time < Tend), do something
        while any(active):

            MS_active = [self.MS[p] for p in active_slots]
            done = False
            while not done:
                done = self.pfasst(MS_active)

            # restart the entire block from scratch if a single step needs to be restarted
            if True in self.restart:  # recompute current block
                # restart active steps (reset all values and pass u0 to u0)
                if len(self.MS) > 1:
                    raise NotImplementedError('restart only implemented for 1 rank just yet')
                self.restart_block(active_slots, time, self.MS[active_slots[0]].levels[0].u[0])

            else:  # move on to next block
                # uend is uend of the last active step in the list
                uend = self.MS[active_slots[-1]].levels[0].uend

                self.error_estimator.store_values(MS_active)

                for p in active_slots:
                    time[p] += num_procs * self.MS[p].dt

                # determine new set of active steps and compress slots accordingly
                active = [time[p] < Tend - 10 * np.finfo(float).eps for p in slots]
                active_slots = list(itertools.compress(slots, active))

                # restart active steps (reset all values and pass uend to u0)
                self.restart_block(active_slots, time, uend)

        # call post-run hook
        for S in self.MS:
            self.hooks.post_run(step=S, level_number=0)

        return uend, self.hooks.return_stats()

    def resilence(self, local_MS_running):
        if self.params.HotRod:
            self.hotrod(local_MS_running)

    def hotrod(self, local_MS_running):
        for S in local_MS_running:
            if S.status.iter == S.params.maxiter:
                for l in S.levels:
                    l.u[:] = l.uold[:]

    def adaptivity(self, MS):
        """
        Method to compute time step size adaptively based on embedded error estimate
        """

        # loop through steps and compute local error and optimal step size from there
        for i in range(len(MS)):
            S = MS[i]

            # check if we performed the desired amount of sweeps
            if S.status.iter < S.params.maxiter:
                continue

            L = S.levels[0]

            # compute next step size
            order = S.status.iter  # embedded error estimate is same order as time marching
            h_opt = L.params.dt * 0.9 * (L.params.e_tol / L.status.e_embedded)**(1. / order)

            # distribute step sizes
            if len(MS) > 1:
                raise NotImplementedError('Adaptivity only implemented for 1 rank just yet')
            else:
                L.params.dt = h_opt

            # check whether to move on or restart
            if L.status.e_embedded >= L.params.e_tol:
                self.restart[i] = True
            else:
                self.restart[i] = False
