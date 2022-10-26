import numpy as np
import scipy as sp

from pySDC.core.Collocation import CollBase
from pySDC.core.ConvergenceController import ConvergenceController


class SwitchEstimator(ConvergenceController):
    """
    Method to estimate a discrete event (switch)
    """

    def setup(self, controller, params, description):
        # for RK4 sweeper, sweep.coll.nodes now consists of values of ButcherTableau
        # for this reason, collocation nodes will be generated here
        coll = CollBase(
            num_nodes=description['sweeper_params']['num_nodes'],
            quad_type=description['sweeper_params']['quad_type'],
        )
        self.coll_nodes_local = coll.nodes
        self.switch_detected = False
        self.switch_detected_step = False
        self.t_switch = None
        self.count_switches = 0
        self.dt_initial = description['level_params']['dt']
        return {'control_order': 100, **params}

    def get_new_step_size(self, controller, S):
        def regulaFalsiMethod(a0, b0, f, tol, maxIter=50):
            """
            Regula falsi method to find the root for the switch
            Args:
                a0, b0 (np.float):              points to start the method
                f (callable function):          function values
                tol (np.float):                 when tol is reached, the secant method breaks
                maxIter (np.int):               maximum number of iterations to find root

            Return:
                The root of f
            """
            count = 0
            while count <= maxIter:
                c0 = a0 - ((b0 - a0) / (f(b0) - f(a0))) * f(a0)

                if f(a0) * f(c0) > 0:
                    a0 = c0
                    b0 = b0

                if f(b0) * f(c0) > 0:
                    a0 = a0
                    b0 = c0

                count += 1

                cm = a0 - ((b0 - a0) / (f(b0) - f(a0))) * f(a0)

                if abs(cm - c0) < tol:
                    print("Regula falsi method: Number of iterations: ", count, "-- Root: ", c0)
                    break

            return c0

        self.switch_detected = False  # reset between steps

        L = S.levels[0]

        if not type(L.prob.params.V_ref) == int and not type(L.prob.params.V_ref) == float:
            # if V_ref is not a scalar, but an (np.)array
            V_ref = np.zeros(np.shape(L.prob.params.V_ref)[0], dtype=float)
            for m in range(np.shape(L.prob.params.V_ref)[0]):
                V_ref[m] = L.prob.params.V_ref[m]
        else:
            V_ref = np.array([L.prob.params.V_ref], dtype=float)

        if S.status.iter > 0 and self.count_switches < np.shape(V_ref)[0]:
            for m in range(len(L.u)):
                if L.u[m][self.count_switches + 1] - V_ref[self.count_switches] <= 0:
                    self.switch_detected = True
                    m_guess = m - 1
                    break

            if self.switch_detected:
                t_interp = [L.time + L.dt * self.coll_nodes_local[m] for m in range(len(self.coll_nodes_local))]

                vC_switch = []
                for m in range(1, len(L.u)):
                    vC_switch.append(L.u[m][self.count_switches + 1] - V_ref[self.count_switches])

                # only find root if vc_switch[0], vC_switch[-1] have opposite signs (intermediate value theorem)
                if vC_switch[0] * vC_switch[-1] < 0:
                    p = sp.interpolate.interp1d(t_interp, vC_switch, 'cubic', bounds_error=False)

                    SwitchResults = sp.optimize.root_scalar(
                        p,
                        method='brentq',
                        bracket=[t_interp[0], t_interp[m_guess]],
                        x0=t_interp[m_guess],
                        xtol=1e-10,
                    )
                    self.t_switch = SwitchResults.root  # t_switch[0]

                    # if the switch is not find, we need to do ... ?
                    if L.time < self.t_switch < L.time + L.dt:
                        r = 1
                        tol = self.dt_initial / r

                        if not np.isclose(self.t_switch - L.time, L.dt, atol=tol):
                            L.status.dt_new = self.t_switch - L.time

                        else:
                            print('Switch located at time: {}'.format(self.t_switch))
                            L.status.dt_new = self.t_switch - L.time
                            L.prob.params.set_switch[self.count_switches] = self.switch_detected
                            L.prob.params.t_switch[self.count_switches] = self.t_switch
                            controller.hooks.add_to_stats(
                                process=S.status.slot,
                                time=self.t_switch,
                                level=L.level_index,
                                iter=0,
                                sweep=L.status.sweep,
                                type='switch{}'.format(self.count_switches + 1),
                                value=p(self.t_switch),
                            )
                            # self.switch_detected_step = self.switch_detected
                            self.switch_detected_step = True

                    else:
                        self.switch_detected = False

                else:
                    self.switch_detected = False

    def determine_restart(self, controller, S):
        if self.switch_detected:
            print("Restart")
            S.status.restart = True
            S.status.force_done = True

        super(SwitchEstimator, self).determine_restart(controller, S)

    def post_step_processing(self, controller, S):
        L = S.levels[0]

        if self.switch_detected_step:
            if L.prob.params.set_switch[self.count_switches] and L.time + L.dt > self.t_switch:
                self.count_switches += 1
                self.t_switch = None
                self.switch_detected_step = False

                L.status.dt_new = self.dt_initial

        super(SwitchEstimator, self).post_step_processing(controller, S)
