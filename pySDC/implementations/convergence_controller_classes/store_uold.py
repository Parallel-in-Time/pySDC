from pySDC.core.ConvergenceController import ConvergenceController


class StoreUOld(ConvergenceController):
    """
    Class to store the solution of the last iteration in a variable called 'uold' of the levels.

    Default control order is 90.
    """

    def setup(self, controller, params, description, **kwargs):
        """
        Define parameters here

        Args:
            controller (pySDC.Controller): The controller
            params (dict): The params passed for this specific convergence controller
            description (dict): The description object used to instantiate the controller

        Returns:
            (dict): The updated params dictionary
        """
        return {"control_order": +90, **params}

    def post_iteration_processing(self, controller, S, **kwargs):
        """
        Store the solution at the current iteration

        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.Step): The current step

        Return:
            None
        """
        for L in S.levels:
            L.uold[:] = L.u[:]

        return None

    def post_spread_processing(self, controller, S, **kwargs):
        """
        Store the initial conditions in u_old in the spread phase.

        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.Step): The current step

        Return:
            None
        """
        self.post_iteration_processing(controller, S, **kwargs)
        return None
