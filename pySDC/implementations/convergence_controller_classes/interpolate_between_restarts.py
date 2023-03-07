import numpy as np
from pySDC.core.ConvergenceController import ConvergenceController
from pySDC.core.Lagrange import LagrangeApproximation
from pySDC.core.Collocation import CollBase


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

        Returns:
            None
        """
        defaults = {
            'control_order': 50,
            'initial_guess': description['sweeper_params'].get('initial_guess', 'spread'),
        }
        return {**defaults, **super().setup(controller, params, description, **kwargs)}

    def setup_status_variables(self, controller, **kwargs):
        """
        Add variables to the sweeper containing the interpolated solution and right hand side.

        Args:
            controller (pySDC.Controller.controller): The controller

        Returns:
            None
        """
        for S in controller.MS if not self.params.useMPI else [controller.S]:
            self.add_variable(S, 'u_inter', where=['levels', '_level__sweep'])
            self.add_variable(S, 'f_inter', where=['levels', '_level__sweep'])

    def post_iteration_processing(self, controller, S, **kwargs):
        """
        Interpolate the solution and right hand sides and store them in the sweeper, where they will be distributed
        accordingly in the prediction step.

        This function is called after every iteration instead of just after the step because we might choose to stop
        iterating as soon as we have decided to restart. If we let the step continue to iterate, this is not the most
        efficient implementation and you may choose to write a different convergence controller.

        The interpolation is based on Thibaut's magic.

        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.Step.step): The current step

        Returns:
            None
        """

        if S.status.restart and any([L.status.dt_new for L in S.levels]):

            for L in S.levels:
                nodes_old = L.sweep.coll.nodes.copy()
                nodes_new = L.sweep.coll.nodes.copy() * L.status.dt_new / L.params.dt

                interpolator = LagrangeApproximation(points=np.append(0, nodes_old))
                L.sweep.u_inter = (interpolator.getInterpolationMatrix(np.append(0, nodes_new)) @ L.u[:])[:]
                L.sweep.f_inter = (interpolator.getInterpolationMatrix(np.append(0, nodes_new)) @ L.f[:])[:]

                L.sweep.params.initial_guess = 'interpolate'

                self.log(f'Interpolating before restart from dt={L.params.dt:.2e} to dt={L.status.dt_new:.2e}', S)

        else:  # if there is no restart planned, please predict as usual
            for L in S.levels:
                L.sweep.params.initial_guess = self.params.initial_guess
