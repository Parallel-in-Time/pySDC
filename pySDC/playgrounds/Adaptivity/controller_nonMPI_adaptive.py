import itertools
import copy as cp
import numpy as np
import dill

from pySDC.core.Controller import controller
from pySDC.core import Step as stepclass
from pySDC.core.Errors import ControllerError, CommunicationError, ParameterError

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI


class controller_nonMPI_adaptive(controller_nonMPI):
    """

    PFASST controller, running serialized version of PFASST in blocks (MG-style)
    -> added adaptivity on top of controller_nonMPI

    """

    def __init__(self, num_procs, controller_params, description):
        """
        Initialization routine for PFASST controller

        Args:
           num_procs: number of parallel time steps (still serial, though), can be 1
           controller_params: parameter set for the controller and the steps
           description: all the parameters to set up the rest (levels, problems, transfer, ...)
        """

        # additional parameters
        if 'use_adaptivity' not in controller_params.keys():
            controller_params['use_adaptivity'] = True
        if 'e_tol' not in description['level_params'].keys():
            raise ParameterError(f'Please supply "e_tol" in the level parameters')
        if 'restol' in description['level_params'].keys():
            if description['level_params']['restol'] > 0:
                description['level_params']['restol'] = 0
                print(f'I want to do always maxiter={description["step_params"]["maxiter"]} iterations to have a constant order in time for adaptivity. Setting restol=0')
         
        # call parent's initialization routine
        super(controller_nonMPI_adaptive, self).__init__(num_procs, controller_params, description)

 
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

            # check whether any step wants to be recomputed
            restart = [self.MS[p].status.stage == 'RESTART' for p in active_slots]
             
            # restart the entire block from scratch if a single step needs to be restarted
            if True in restart: # recompute current block
                # restart active steps (reset all values and pass u0 to u0)
                if len(self.MS) > 1:
                    raise NotImplementedError(f'restart only implemented for 1 rank just yet')
                self.restart_block(active_slots, time, self.MS[active_slots[0]].levels[0].u[0])
                 
            else: # move on to next block
                # uend is uend of the last active step in the list
                uend = self.MS[active_slots[-1]].levels[0].uend

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

        if self.params.use_adaptivity:
            self.adaptivity(local_MS_running)

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

            if S.status.stage == 'RESTART':
                pass
            elif not S.status.done:
                # increment iteration count here (and only here)
                S.status.iter += 1
                self.hooks.pre_iteration(step=S, level_number=0)

                if self.params.use_iteration_estimator or self.params.use_adaptivity:
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

    def adaptivity(self, MS):
        """
        Method to compute time step size adaptively based on embedded error estimate
        """

        # loop through steps and compute local error and optimal step size from there
        for S in MS:
            # check if we performed the desired amount of sweeps
            if S.status.iter < S.params.maxiter:
                continue

            L = S.levels[0]

            # do embedded method on the last collocation node
            local_error = abs(L.uold[-1] - L.u[-1])

            # compute next step size
            order = S.status.iter # embedded error estimate is same order as time marching
            h_opt = L.params.dt * 0.9 * (L.params.e_tol/local_error)**(1./order)

            # distribute step sizes
            if len(MS) > 1:
                raise NotImplementedError(f'Adaptivity only implemented for 1 rank just yet')
            else:
                L.params.dt = h_opt

            # check whether to move on or restart
            if local_error >= L.params.e_tol:
                S.status.stage = 'RESTART'

    def abort(self, local_MS_active):
        """
        Dummy function with the only purpose to set the status to done regardless of convergece
        """
        for S in local_MS_active:
            S.status.done = True

    def pfasst(self, local_MS_active):
        """
        Main function including the stages of SDC, MLSDC and PFASST (the "controller")

        For the workflow of this controller, check out one of our PFASST talks or the pySDC paper

        This method changes self.MS directly by accessing active steps through local_MS_active. Nothing is returned.

        Args:
            local_MS_active (list): all active steps
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
            'PREDICT': self.predict,
            'IT_CHECK': self.it_check,
            'IT_FINE': self.it_fine,
            'IT_DOWN': self.it_down,
            'IT_COARSE': self.it_coarse,
            'IT_UP': self.it_up,
            'RESTART' : self.abort,
        }

        switcher.get(stage, self.default)(MS_running)

        return all([S.status.done for S in local_MS_active])
