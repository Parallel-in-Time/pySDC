from pySDC.core.ConvergenceController import ConvergenceController


class CheckConvergence(ConvergenceController):
    def setup(self, controller, params, description):
        return {'control_order': -100, **params}

    def check_iteration_status(self, controller, S):
        """
        Routine to determine whether to stop iterating (currently testing the residual + the max. number of iterations)

        Args:
            controller (pySDC.Controller.controller): controller
            S (pySDC.Step.step): current step
        """

        # do all this on the finest level
        L = S.levels[0]

        # get residual and check against prescribed tolerance (plus check number of iterations
        res = L.status.residual
        converged = S.status.iter >= S.params.maxiter or res <= L.params.restol or S.status.force_done
        if converged is not None:
            S.status.done = converged
