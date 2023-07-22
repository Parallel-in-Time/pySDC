import numpy as np
import scipy as sp

from pySDC.core.Collocation import CollBase
from pySDC.core.ConvergenceController import ConvergenceController, Status


class SwitchEstimator(ConvergenceController):
    """
    Class to predict the time point of the event and setting a new step size. For the first time, this is a nonMPI version,
    because a MPI version is not yet developed.
    """

    def setup(self, controller, params, description):
        """
        Function sets default variables to handle with the switch at the beginning.

        Parameters
        ----------
        controller : pySDC.Controller
            The controller doing all the stuff in a computation.
        params : dict
            The parameters passed for this specific convergence controller.
        description : dict
            The description object used to instantiate the controller.

        Returns
        -------
        convergence_controller_params : dict
            The updated params dictionary.
        """

        # for RK4 sweeper, sweep.coll.nodes now consists of values of ButcherTableau
        # for this reason, collocation nodes will be generated here
        coll = CollBase(
            num_nodes=description['sweeper_params']['num_nodes'],
            quad_type=description['sweeper_params']['quad_type'],
        )

        defaults = {
            'control_order': 100,
            'tol': description['level_params']['dt'],
            'nodes': coll.nodes,
        }
        return {**defaults, **params}

    def setup_status_variables(self, controller, **kwargs):
        """
        Adds switching specific variables to status variables.

        Parameters
        ----------
        controller : pySDC.Controller
            The controller doing all the stuff in a computation.
        """

        self.status = Status(['switch_detected', 't_switch'])

    def reset_status_variables(self, controller, **kwargs):
        """
        Resets status variables.

        Parameters
        ----------
        controller : pySDC.Controller
            The controller doing all the stuff in a computation.
        """

        self.setup_status_variables(controller, **kwargs)

    def get_new_step_size(self, controller, S, **kwargs):
        """
        Determine a new step size when an event is found such that the event occurs at the time step.

        Parameters
        ----------
        controller : pySDC.Controller
            The controller doing all the stuff in a computation.
        S : pySDC.Step
            The current step.
        """

        L = S.levels[0]

        if S.status.iter == S.params.maxiter:
            self.status.switch_detected, m_guess, state_function = L.prob.get_switching_info(L.u, L.time)

            if self.status.switch_detected:
                t_interp = [L.time + L.dt * self.params.nodes[m] for m in range(len(self.params.nodes))]

                # only find root if vc_switch[0], vC_switch[-1] have opposite signs (intermediate value theorem)
                if state_function[0] * state_function[-1] < 0:
                    self.status.t_switch = self.get_switch(t_interp, state_function, m_guess)

                    if L.time <= self.status.t_switch <= L.time + L.dt:
                        dt_switch = self.status.t_switch - L.time
                        if not np.isclose(self.status.t_switch - L.time, L.dt, atol=self.params.tol):
                            self.log(
                                f"Located Switch at time {self.status.t_switch:.6f} is outside the range of tol={self.params.tol:.4e}",
                                S,
                            )

                        else:
                            self.log(
                                f"Switch located at time {self.status.t_switch:.6f} inside tol={self.params.tol:.4e}", S
                            )

                            L.prob.t_switch = self.status.t_switch
                            controller.hooks[0].add_to_stats(
                                process=S.status.slot,
                                time=L.time,
                                level=L.level_index,
                                iter=0,
                                sweep=L.status.sweep,
                                type='switch',
                                value=self.status.t_switch,
                            )

                            L.prob.count_switches()

                        dt_planned = L.status.dt_new if L.status.dt_new is not None else L.params.dt

                        # when a switch is found, time step to match with switch should be preferred
                        if self.status.switch_detected:
                            L.status.dt_new = dt_switch

                        else:
                            L.status.dt_new = min([dt_planned, dt_switch])

                    else:
                        self.status.switch_detected = False

                else:
                    self.status.switch_detected = False

    def determine_restart(self, controller, S, **kwargs):
        """
        Check if the step needs to be restarted due to a predicting switch.

        Parameters
        ----------
        controller : pySDC.Controller
            The controller doing all the stuff in a computation.
        S : pySDC.Step
            The current step.
        """

        if self.status.switch_detected:
            S.status.restart = True
            S.status.force_done = True

        super().determine_restart(controller, S, **kwargs)

    def post_step_processing(self, controller, S, **kwargs):
        """
        After a step is done, some variables will be prepared for predicting a possibly new switch.
        If no Adaptivity is used, the next time step will be set as the default one from the front end.

        Parameters
        ----------
        controller : pySDC.Controller
            The controller doing all the stuff in a computation.
        S : pySDC.Step
            The current step.
        """

        L = S.levels[0]

        if self.status.t_switch is None:
            L.status.dt_new = L.status.dt_new if L.status.dt_new is not None else L.params.dt_initial

        super().post_step_processing(controller, S, **kwargs)

    @staticmethod
    def get_switch(t_interp, state_function, m_guess):
        """
        Routine to do the interpolation and root finding stuff.

        Parameters
        ----------
        t_interp : list
            Collocation nodes in a step.
        state_function : list
            Contains values of state function at these collocation nodes.
        m_guess : float
            Index at which the difference drops below zero.

        Returns
        -------
        t_switch : float
           Time point of the founded switch.
        """

        p = sp.interpolate.interp1d(t_interp, state_function, 'cubic', bounds_error=False)

        SwitchResults = sp.optimize.root_scalar(
            p,
            method='brentq',
            bracket=[t_interp[0], t_interp[m_guess]],
            x0=t_interp[m_guess],
            xtol=1e-10,
        )
        t_switch = SwitchResults.root

        return t_switch
