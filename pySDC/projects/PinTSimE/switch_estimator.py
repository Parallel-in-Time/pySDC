import numpy as np
import scipy as sp

from pySDC.core.ConvergenceController import ConvergenceController


class SwitchEstimator(ConvergenceController):
    """
    Method to estimate a discrete event (switch)
    """

    def setup(self, controller, params, description):
        self.switch_detected = False
        return {'control_order': 100, **params}

    def get_new_step_size(self, controller, S):
        self.switch_detected = False  # reset between steps

        L = S.levels[0]

        for m in range(len(L.u)):
            if L.u[m][1] - L.prob.params.V_ref < 0:
                m_guess = m - 1
                self.switch_detected = True
                break

        if self.switch_detected:
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
            self.t_switch = t_switch[0]

            if L.time < self.t_switch < L.time + L.dt and not np.isclose(self.t_switch - L.time, L.dt, atol=1e-2):
                print('Switch located at time: {}'.format(self.t_switch))
                L.status.dt_new = self.t_switch - L.time
            else:
                self.switch_detected = False

    def determine_restart(self, controller, S):
        if self.switch_detected:
            S.status.restart = True
            S.status.force_done = True

        super(SwitchEstimator, self).determine_restart(controller, S)
