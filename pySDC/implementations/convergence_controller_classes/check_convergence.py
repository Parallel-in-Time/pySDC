from pySDC.core.ConvergenceController import ConvergenceController


class CheckConvergence(ConvergenceController):
    '''
    Perform simple checks on convergence for SDC iterations.

    Iteration is terminated via one of two criteria:
     - Residual tolerance
     - Maximum number of iterations
    '''

    def setup(self, controller, params, description):
        '''
        Define default parameters here

        Args:
            controller (pySDC.Controller): The controller
            params (dict): The params passed for this specific convergence controller
            description (dict): The description object used to instantiate the controller

        Returns:
            (dict): The updated params dictionary
        '''
        return {'control_order': -100, **params}

    def check_iteration_status(self, controller, S):
        """
        Routine to determine whether to stop iterating (currently testing the residual + the max. number of iterations)

        Args:
            controller (pySDC.Controller.controller): The controller
            S (pySDC.Step.step): The current step

        Returns:
            None
        """

        # do all this on the finest level
        L = S.levels[0]

        # get residual and check against prescribed tolerance (plus check number of iterations
        res = L.status.residual
        converged = S.status.iter >= S.params.maxiter or res <= L.params.restol or S.status.force_done
        if converged is not None:
            S.status.done = converged

        return None
