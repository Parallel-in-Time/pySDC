import numpy as np
from pySDC.core.ConvergenceController import ConvergenceController, Status
from pySDC.core.Lagrange import LagrangeApproximation


class InterpolateBetweenRestarts(ConvergenceController):
    """
    Interpolate the solution and right hand side to the new set of collocation nodes after a restart.
    The idea is that when you adjust the step size between restarts, you already know what the new quadrature method
    is going to be and possibly interpolating the current iterate to these results in a better initial guess than
    spreading the initial conditions or whatever you usually like to do.
    """

    def setup(self, controller, params, description, **kwargs):
        """
        Store the initial guess used in the sweeper when no restart has happened

        Args:
            controller (pySDC.Controller.controller): The controller
            params (dict): Parameters for the convergence controller
            description (dict): The description object used to instantiate the controller
        """
        defaults = {
            'control_order': 50,
        }
        return {**defaults, **super().setup(controller, params, description, **kwargs)}

    def setup_status_variables(self, controller, **kwargs):
        """
        Add variables to the sweeper containing the interpolated solution and right hand side.

        Args:
            controller (pySDC.Controller.controller): The controller
        """
        self.status = Status(['u_inter', 'f_inter', 'perform_interpolation'])

        self.status.u_inter = []
        self.status.f_inter = []
        self.status.perform_interpolation = False

    def post_spread_processing(self, controller, step, **kwargs):
        """
        Spread the interpolated values to the collocation nodes. This overrides whatever the sweeper uses for prediction.

        Args:
            controller (pySDC.Controller.controller): The controller
            step (pySDC.Step.step): The current step
        """
        if self.status.perform_interpolation:
            for i in range(len(step.levels)):
                level = step.levels[i]
                for m in range(len(level.u)):
                    level.u[m][:] = self.status.u_inter[i][m][:]
                    level.f[m][:] = self.status.f_inter[i][m][:]

            # reset the status variables
            self.status.perform_interpolation = False
            self.status.u_inter = []
            self.status.f_inter = []

    def post_iteration_processing(self, controller, step, **kwargs):
        """
        Interpolate the solution and right hand sides and store them in the sweeper, where they will be distributed
        accordingly in the prediction step.

        This function is called after every iteration instead of just after the step because we might choose to stop
        iterating as soon as we have decided to restart. If we let the step continue to iterate, this is not the most
        efficient implementation and you may choose to write a different convergence controller.

        The interpolation is based on Thibaut's magic.

        Args:
            controller (pySDC.Controller): The controller
            step (pySDC.Step.step): The current step
        """
        if step.status.restart and all(level.status.dt_new for level in step.levels):
            for level in step.levels:
                nodes_old = level.sweep.coll.nodes.copy()
                nodes_new = level.sweep.coll.nodes.copy() * level.status.dt_new / level.params.dt

                interpolator = LagrangeApproximation(points=np.append(0, nodes_old))
                self.status.u_inter += [(interpolator.getInterpolationMatrix(np.append(0, nodes_new)) @ level.u[:])[:]]
                self.status.f_inter += [(interpolator.getInterpolationMatrix(np.append(0, nodes_new)) @ level.f[:])[:]]

                self.status.perform_interpolation = True

                self.log(
                    f'Interpolating before restart from dt={level.params.dt:.2e} to dt={level.status.dt_new:.2e}', step
                )
        else:
            self.status.perform_interpolation = False
