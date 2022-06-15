from pySDC.core.ConvergenceController import ConvergenceController


class CheckConvergence(ConvergenceController):

    def __init__(self, controller, description):
        super(CheckConvergence, self).__init__(controller, description)
        self.params.order = -100

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
