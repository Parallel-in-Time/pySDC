import numpy as np
import scipy as sp
import matplotlib
matplotlib.use('TkAgg')

from pySDC.core.ConvergenceController import ConvergenceController


class SwitchEstimator(ConvergenceController):
    """
        Method to estimate a discrete event (switch)
    """

    def setup(self, controller, params, description):
        self.switch_detected = False
        self.dt_adapted = False
        self.t_switch = None
        return {'control_order': 100, **params}

    def get_new_step_size(self, controller, S):
        self.switch_detected = False  # reset between steps

        L = S.levels[0]

        if S.status.iter > 0:
            for m in range(len(L.u)):
                if L.u[m][1] - L.prob.params.V_ref <= 0:
                    self.switch_detected = True
                    m_guess = m - 1
                    break

            if self.switch_detected:
                t_interp = [L.time + L.dt * L.sweep.coll.nodes[m] for m in range(len(L.sweep.coll.nodes))]

                vC_switch = []
                vC_plot = []
                for m in range(1, len(L.u)):
                    vC_switch.append(L.u[m][1] - L.prob.params.V_ref)
                    vC_plot.append(L.u[m][1])

                # only find root if vc_switch[0], vC_switch[-1] have opposite signs (intermediate value theorem)
                if vC_switch[0] * vC_switch[-1] < 0:
                    p = sp.interpolate.interp1d(t_interp, vC_switch, 'cubic', bounds_error=False)

                    t_switch, info, _, _ = sp.optimize.fsolve(p, t_interp[m_guess], full_output=True)
                    self.t_switch = t_switch[0]

                    # if the switch is not find, we need to do ... ?
                    if L.time < self.t_switch < L.time + L.dt:
                        # for coarser time steps, tolerance have to be adapted with a ratio
                        if 1 > L.dt > 1e-1:
                            r = L.dt / 1e-1

                        else:
                            r = 1

                        tol = L.dt / r

                        if not np.isclose(self.t_switch - L.time, L.dt, atol=tol):
                            L.status.dt_new = self.t_switch - L.time

                        else:
                            print('Switch located at time: {}'.format(self.t_switch))
                            L.status.dt_new = self.t_switch - L.time
                            L.prob.params.set_switch = True
                            L.prob.params.t_switch = self.t_switch
                            controller.hooks.add_to_stats(process=S.status.slot, time=self.t_switch,
                                                          level=L.level_index, iter=0, sweep=L.status.sweep,
                                                          type='switch', value=p(self.t_switch))

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
        self.dt_adapted = False
        S.levels[0].status.dt_new = S.levels[0].params.dt_initial

        super(SwitchEstimator, self).post_step_processing(controller, S)
