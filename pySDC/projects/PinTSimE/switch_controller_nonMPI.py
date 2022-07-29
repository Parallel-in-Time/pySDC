import numpy as np
import scipy as sp

from pySDC.core.Errors import ParameterError
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI


class switch_controller_nonMPI(controller_nonMPI):
    """
    Controller class including switch handling
    """

    def __init__(self, num_procs, controller_params, description):
        """
        Initialization routine for PFASST controller with switch handling
        Args:
           num_procs: number of parallel time steps (still serial, though), can be 1
           controller_params: parameter set for the controller and the steps
           description: all the parameters to set up the rest (levels, problems, transfer, ...)
        """

        # call parent's initialization routine
        super(switch_controller_nonMPI, self).__init__(num_procs, controller_params, description)

        # prepare variable for switch estimation
        if self.params.use_switch_estimator:
            if 'V_ref' not in description['problem_params'].keys():
                raise ParameterError('Please supply "V_ref" in the problem parameters')

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
                self.logger.debug(
                    'Process %2i provides data on level %2i with tag %s' % (S.status.slot, 0, S.status.iter)
                )
                self.send(S.levels[0], tag=(0, S.status.iter, S.status.slot))

            # receive values
            if not S.status.prev_done and not S.status.first:
                self.logger.debug(
                    'Process %2i receives from %2i on level %2i with tag %s'
                    % (S.status.slot, S.prev.status.slot, 0, S.status.iter)
                )
                self.recv(S.levels[0], S.prev.levels[0], tag=(0, S.status.iter, S.prev.status.slot))
            self.hooks.post_comm(step=S, level_number=0)

            S.levels[0].sweep.compute_residual()

        if self.params.use_iteration_estimator:
            self.check_iteration_estimator(local_MS_running)

        self.error_estimator.estimate(local_MS_running)

        if self.params.use_adaptivity:
            self.adaptivity(local_MS_running)

        self.resilience(local_MS_running)

        if self.params.use_switch_estimator:
            self.switch_estimator(local_MS_running)

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
                    # store previous iterate to compute difference later on
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

    def switch_estimator(self, MS):
        """
        Method to estimate a discrete event (switch)
        """

        for i in range(len(MS)):
            S = MS[i]
            L = S.levels[0]

            switch_detected = False
            for m in range(len(L.u)):
                if L.u[m][1] - L.prob.params.V_ref < 0:
                    m_guess = m - 1
                    switch_detected = True
                    break

            if switch_detected:
                t_interp = [L.time + L.dt * L.sweep.coll.nodes[m] for m in range(len(L.sweep.coll.nodes))]

                vC = []
                for m in range(1, len(L.u)):
                    vC.append(L.u[m][1])

                p = sp.interpolate.interp1d(t_interp, vC, 'cubic', bounds_error=False)

                def switch_examiner(x):
                    """
                    Routine to define root problem
                    """

                    return L.prob.params.V_ref - p(x)

                t_switch, info, _, _ = sp.optimize.fsolve(switch_examiner, t_interp[m_guess], full_output=True)
                t_switch = t_switch[0]
                print('Switch located at time: {}'.format(t_switch))
                if L.time < t_switch < L.time + L.dt and not np.isclose(t_switch - L.time, L.dt, atol=1e-2):
                    S.status.restart = True
                    L.status.dt_new = t_switch - L.time
                    S.status.force_done = True
