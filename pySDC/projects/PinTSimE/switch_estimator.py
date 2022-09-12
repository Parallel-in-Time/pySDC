import numpy as np
import scipy as sp

from pySDC.core.ConvergenceController import ConvergenceController


class SwitchEstimator(ConvergenceController):
    """
    Method to estimate a discrete event (switch)
    """

    def setup(self, controller, params, description):
        self.switch_detected = False
        self.dt_adapted = False
        self.t_switch = None
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

        if S.status.iter > 0:
            for m in range(len(L.u)):
                if L.u[m][1] - L.prob.params.V_ref <= 0:
                    self.switch_detected = True
                    break

            if self.switch_detected:
                t_interp = [L.time + L.dt * L.sweep.coll.nodes[m] for m in range(len(L.sweep.coll.nodes))]

                vC_switch = []
                for m in range(1, len(L.u)):
                    vC_switch.append(L.u[m][1] - L.prob.params.V_ref)

                # only find root if vc_switch[0], vC_switch[-1] have opposite signs (intermediate value theorem)
                if vC_switch[0] * vC_switch[-1] < 0:
                    p = sp.interpolate.interp1d(t_interp, vC_switch, 'cubic', bounds_error=False)
                    
                    self.t_switch = regulaFalsiMethod(t_interp[0], t_interp[-1], p, 1e-12)

                    # if the switch is not find, we need to do ... ?
                    if L.time < self.t_switch < L.time + L.dt:
                        # for coarser time steps, tolerance have to be adapted with a ratio
                        # if 1 > self.dt_initial > 1e-1:
                        #    r = L.dt / 1e-1
                        #    r = self.dt_initial / 1e-1

                        # else:
                        #    r = 1
                        r = 1
                        tol = self.dt_initial / r

                        if not np.isclose(self.t_switch - L.time, L.dt, atol=tol):
                            L.status.dt_new = self.t_switch - L.time

                        else:
                            print('Switch located at time: {}'.format(self.t_switch))
                            L.status.dt_new = self.t_switch - L.time
                            L.prob.params.set_switch = self.switch_detected
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

        L = S.levels[0]

        if self.switch_detected:
            if L.time + L.dt > self.t_switch:
                L.prob.params.set_switch = False  # allows to detecting more discrete events

        # if self.t_switch is not None and L.time > self.t_switch:
        L.status.dt_new = self.dt_initial

        super(SwitchEstimator, self).post_step_processing(controller, S)
