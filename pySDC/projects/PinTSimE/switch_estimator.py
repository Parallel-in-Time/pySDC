import numpy as np
import scipy as sp

from pySDC.core.Collocation import CollBase
from pySDC.core.ConvergenceController import ConvergenceController


class SwitchEstimator(ConvergenceController):
    """
    Class to predict the time point of the switch and setting a new step size

    For the first time, this is a nonMPI version, because a MPI version is not yet developed.
    """

    def setup(self, controller, params, description):
        """
        Function sets default variables to handle with the switch at the beginning.

        Args:
            controller (pySDC.Controller): The controller
            params (dict): The params passed for this specific convergence controller
            description (dict): The description object used to instantiate the controller

        Returns:
            (dict): The updated params dictionary
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
            'coll_nodes_local': coll.nodes,
            'switch_detected': False,
            'switch_detected_step': False,
            't_switch': None,
            'count_switches': 0,
            'dt_initial': description['level_params']['dt'],
        }
        return {**defaults, **params}

    def get_new_step_size(self, controller, S):
        """
        Determine a new step size when a switch is found such that the switch happens at the time step.

        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.Step): The current step

        Returns:
            None
        """

        self.params.switch_detected = False  # reset between steps

        L = S.levels[0]

        if not type(L.prob.params.V_ref) == int and not type(L.prob.params.V_ref) == float:
            # if V_ref is not a scalar, but an (np.)array
            V_ref = np.zeros(np.shape(L.prob.params.V_ref)[0], dtype=float)
            for m in range(np.shape(L.prob.params.V_ref)[0]):
                V_ref[m] = L.prob.params.V_ref[m]
        else:
            V_ref = np.array([L.prob.params.V_ref], dtype=float)

        if S.status.iter == S.params.maxiter and self.count_switches < np.shape(V_ref)[0]:
            # if S.status.iter > 0 and self.count_switches < np.shape(V_ref)[0]:
            for m in range(len(L.u)):
                if L.u[m][self.params.count_switches + 1] - V_ref[self.params.count_switches] <= 0:
                    self.params.switch_detected = True
                    m_guess = m - 1
                    break

            if self.params.switch_detected:
                t_interp = [
                    L.time + L.dt * self.params.coll_nodes_local[m] for m in range(len(self.params.coll_nodes_local))
                ]

                vC_switch = []
                for m in range(1, len(L.u)):
                    vC_switch.append(L.u[m][self.params.count_switches + 1] - V_ref[self.params.count_switches])

                # only find root if vc_switch[0], vC_switch[-1] have opposite signs (intermediate value theorem)
                if vC_switch[0] * vC_switch[-1] < 0:

                    self.params.t_switch = self.get_switch(t_interp, vC_switch, m_guess)

                    # if the switch is not find, we need to do ... ?
                    if L.time < self.params.t_switch < L.time + L.dt:

                        if not np.isclose(self.params.t_switch - L.time, L.dt, atol=self.params.tol):
                            dt_switch = self.params.t_switch - L.time
                            self.log(
                                f"Located Switch at time {self.params.t_switch:.6f} is outside the range of tol={self.params.tol:.4f}",
                                S,
                            )

                        else:
                            dt_switch = self.params.t_switch - L.time
                            self.log(
                                f"Switch located at time {self.params.t_switch:.6f} inside tol={self.params.tol:.4f}", S
                            )

                            L.prob.t_switch = self.params.t_switch
                            controller.hooks.add_to_stats(
                                process=S.status.slot,
                                time=L.time,
                                level=L.level_index,
                                iter=0,
                                sweep=L.status.sweep,
                                type='switch',
                                value=self.params.t_switch,
                            )

                            self.params.switch_detected_step = True

                        dt_planned = L.status.dt_new if L.status.dt_new is not None else L.params.dt

                        # when a switch is found, time step to match with switch should be preferred
                        if self.params.switch_detected:
                            L.status.dt_new = dt_switch

                        else:
                            L.status.dt_new = min([dt_planned, dt_switch])

                    else:
                        self.params.switch_detected = False

                else:
                    self.params.switch_detected = False

    def determine_restart(self, controller, S):
        """
        Check if the step needs to be restarted due to a predicting switch.

        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.Step): The current step

        Returns:
            None
        """

        if self.params.switch_detected:
            S.status.restart = True
            S.status.force_done = True

        super(SwitchEstimator, self).determine_restart(controller, S)

    def post_step_processing(self, controller, S):
        """
        After a step is done, some variables will be prepared for predicting a possibly new switch.
        If no Adaptivity is used, the next time step will be set as the default one from the front end.

        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.Step): The current step

        Returns:
            None
        """

        L = S.levels[0]

        if self.params.switch_detected_step:
            if L.time + L.dt >= self.params.t_switch:
                self.count_switches += 1
                self.params.t_switch = None
                L.prob.t_switch = self.params.t_switch
                self.params.switch_detected_step = False

                dt_planned = L.status.dt_new if L.status.dt_new is not None else self.params.dt_initial
                L.status.dt_new = dt_planned

        super(SwitchEstimator, self).post_step_processing(controller, S)

    @staticmethod
    def get_switch(t_interp, vC_switch, m_guess):
        """
        Routine to do the interpolation and root finding stuff.

        Args:
            t_interp (list): collocation nodes in a step
            vC_switch (list): differences vC - V_ref at these collocation nodes
            m_guess (np.float): Index at which the difference drops below zero

        Returns:
            t_switch (np.float): time point of th switch
        """

        p = sp.interpolate.interp1d(t_interp, vC_switch, 'cubic', bounds_error=False)

        SwitchResults = sp.optimize.root_scalar(
            p,
            method='brentq',
            bracket=[t_interp[0], t_interp[m_guess]],
            x0=t_interp[m_guess],
            xtol=1e-10,
        )
        t_switch = SwitchResults.root

        return t_switch
