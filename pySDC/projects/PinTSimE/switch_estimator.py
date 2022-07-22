import numpy as np

from pySDC.core.ConvergenceController import ConvergenceController
from pySDC.core.Lagrange import LagrangeApproximation


class SwitchEstimator(ConvergenceController):
    """
        Method to estimate a discrete event (switch)
    """

    def setup(self, controller, params, description):
        self.switch_detected = False
        self.dt_adapted = False
        return {'control_order': 100, **params}

    def get_new_step_size(self, controller, S):
        self.switch_detected = False  # reset between steps

        L = S.levels[0]

        if S.status.iter > 0:
            for m in range(len(L.u)):
                if L.u[m][1] - L.prob.params.V_ref < 0:
                    self.switch_detected = True
                    break

            if self.switch_detected:
                t_interp = [L.time + L.dt * L.sweep.coll.nodes[m] for m in range(len(L.sweep.coll.nodes))]

                vC_switch = []
                for m in range(1, len(L.u)):
                    vC_switch.append(L.u[m][1] - L.prob.params.V_ref)

                approx = LagrangeApproximation(t_interp, weightComputation='AUTO')

                def switch_examiner(x):
                    """
                        Routine to define root problem
                    """

                    return approx.getInterpolationMatrix(x).dot(vC_switch)

                vC_interp = switch_examiner(t_interp)
                if vC_switch[0] * vC_switch[-1] < 0 and not self.dt_adapted:
                    print(vC_interp, t_interp)
                    for m in range(len(t_interp)):
                        if not np.isclose(vC_interp[m], 0, atol=L.dt * 1e-4):
                            L.status.dt_new = 0.5 * L.dt

                        else:
                            self.t_switch = t_interp[m]
                            print(vC_interp[m])
                            print('Switch located at time: {}'.format(self.t_switch))
                            L.status.dt_new = self.t_switch - L.time
                            self.dt_adapted = True

                else:
                    self.switch_detected = False

    def determine_restart(self, controller, S):
        if self.switch_detected:
            S.status.restart = True
            S.status.force_done = True

        super(SwitchEstimator, self).determine_restart(controller, S)

    def post_step_processing(self, controller, S):
        self.dt_adapted = False
        S.levels[0].status.dt_new = S.levels[0].params.dt_initial

        super(SwitchEstimator, self).determine_restart(controller, S)
