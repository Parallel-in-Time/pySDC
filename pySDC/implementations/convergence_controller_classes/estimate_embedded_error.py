import numpy as np

from pySDC.core.ConvergenceController import ConvergenceController, Pars, Status
from pySDC.implementations.convergence_controller_classes.store_uold import StoreUOld

from pySDC.implementations.sweeper_classes.Runge_Kutta import RungeKutta


class EstimateEmbeddedError(ConvergenceController):
    """
    The embedded error is obtained by computing two solutions of different accuracy and pretending the more accurate
    one is an exact solution from the point of view of the less accurate solution. In practice, we like to compute the
    solutions with different order methods, meaning that in SDC we can just subtract two consecutive sweeps, as long as
    you make sure your preconditioner is compatible, which you have to just try out...
    """

    @classmethod
    def get_implementation(cls, flavor='standard', useMPI=False):
        """
        Retrieve the implementation for a specific flavor of this class.

        Args:
            flavor (str): The implementation that you want

        Returns:
            cls: The child class that implements the desired flavor
        """
        if flavor == 'standard':
            return cls
        elif flavor == 'linearized':
            if useMPI:
                return EstimateEmbeddedErrorLinearizedMPI
            else:
                return EstimateEmbeddedErrorLinearizedNonMPI
        elif flavor == 'collocation':
            return EstimateEmbeddedErrorCollocation
        else:
            raise NotImplementedError(f'Flavor {flavor} of EstimateEmbeddedError is not implemented!')

    def setup(self, controller, params, description, **kwargs):
        """
        Add a default value for control order to the parameters and check if we are using a Runge-Kutta sweeper

        Args:
            controller (pySDC.Controller): The controller
            params (dict): Parameters for the convergence controller
            description (dict): The description object used to instantiate the controller

        Returns:
            dict: Updated parameters
        """
        sweeper_type = 'RK' if RungeKutta in description['sweeper_class'].__bases__ else 'SDC'
        return {
            "control_order": -80,
            "sweeper_type": sweeper_type,
            **super().setup(controller, params, description, **kwargs),
        }

    def dependencies(self, controller, description, **kwargs):
        """
        Load the convergence controller that stores the solution of the last sweep unless we are doing Runge-Kutta.
        Add the hook for recording the error.

        Args:
            controller (pySDC.Controller): The controller
            description (dict): The description object used to instantiate the controller

        Returns:
            None
        """
        if RungeKutta not in description["sweeper_class"].__bases__:
            controller.add_convergence_controller(StoreUOld, description=description)

        from pySDC.implementations.hooks.log_embedded_error_estimate import LogEmbeddedErrorEstimate

        controller.add_hook(LogEmbeddedErrorEstimate)
        return None

    def estimate_embedded_error_serial(self, L):
        """
        Estimate the serial embedded error, which may need to be modified for a parallel estimate.

        Depending on the type of sweeper, the lower order solution is stored in a different place.

        Args:
            L (pySDC.level): The level

        Returns:
            dtype_u: The embedded error estimate
        """
        if self.params.sweeper_type == "RK":
            # lower order solution is stored in the second to last entry of L.u
            return abs(L.u[-2] - L.u[-1])
        elif self.params.sweeper_type == "SDC":
            # order rises by one between sweeps, making this so ridiculously easy
            return abs(L.uold[-1] - L.u[-1])
        else:
            raise NotImplementedError(
                f"Don't know how to estimate embedded error for sweeper type \
{self.params.sweeper_type}"
            )

    def setup_status_variables(self, controller, **kwargs):
        """
        Add the embedded error variable to the error function.

        Args:
            controller (pySDC.Controller): The controller
        """
        if 'comm' in kwargs.keys():
            steps = [controller.S]
        else:
            if 'active_slots' in kwargs.keys():
                steps = [controller.MS[i] for i in kwargs['active_slots']]
            else:
                steps = controller.MS
        where = ["levels", "status"]
        for S in steps:
            self.add_variable(S, name='error_embedded_estimate', where=where, init=None)

    def reset_status_variables(self, controller, **kwargs):
        self.setup_status_variables(controller, **kwargs)

    def post_iteration_processing(self, controller, S, **kwargs):
        """
        Estimate the local error here.

        If you are doing MSSDC, this is the global error within the block in Gauss-Seidel mode.
        In Jacobi mode, I haven't thought about what this is.

        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.Step): The current step

        Returns:
            None
        """

        if S.status.iter > 0 or self.params.sweeper_type == "RK":
            for L in S.levels:
                L.status.error_embedded_estimate = max([self.estimate_embedded_error_serial(L), np.finfo(float).eps])

        return None


class EstimateEmbeddedErrorLinearizedNonMPI(EstimateEmbeddedError):
    def __init__(self, controller, params, description, **kwargs):
        """
        Initialisation routine. Add the buffers for communication.

        Args:
            controller (pySDC.Controller): The controller
            params (dict): Parameters for the convergence controller
            description (dict): The description object used to instantiate the controller
        """
        super().__init__(controller, params, description, **kwargs)
        self.buffers = Pars({'e_em_last': 0.0})

    def reset_buffers_nonMPI(self, controller, **kwargs):
        """
        Reset buffers for imitated communication.

        Args:
            controller (pySDC.controller): The controller

        Returns:
            None
        """
        self.buffers.e_em_last = 0.0
        return None

    def post_iteration_processing(self, controller, S, **kwargs):
        """
        Compute embedded error estimate on the last node of each level
        In serial this is the local error, but in block Gauss-Seidel MSSDC this is a semi-global error in each block

        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.Step): The current step

        Returns:
            None
        """
        if len(S.levels) > 1 and len(controller.MS) > 1:
            raise NotImplementedError(
                "Embedded error estimate only works for serial multi-level or parallel single \
level"
            )

        if S.status.iter > 0 or self.params.sweeper_type == "RK":
            for L in S.levels:
                temp = self.estimate_embedded_error_serial(L)
                L.status.error_embedded_estimate = max([abs(temp - self.buffers.e_em_last), np.finfo(float).eps])

            self.buffers.e_em_last = temp * 1.0

        return None


class EstimateEmbeddedErrorLinearizedMPI(EstimateEmbeddedError):
    def __init__(self, controller, params, description, **kwargs):
        """
        Initialisation routine. Add the buffers for communication.

        Args:
            controller (pySDC.Controller): The controller
            params (dict): Parameters for the convergence controller
            description (dict): The description object used to instantiate the controller
        """
        super().__init__(controller, params, description, **kwargs)
        self.buffers = Pars({'e_em_last': 0.0})

    def post_iteration_processing(self, controller, S, **kwargs):
        """
        Compute embedded error estimate on the last node of each level
        In serial this is the local error, but in block Gauss-Seidel MSSDC this is a semi-global error in each block

        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.Step): The current step

        Returns:
            None
        """
        comm = kwargs['comm']

        if S.status.iter > 0 or self.params.sweeper_type == "RK":
            for L in S.levels:
                # get accumulated local errors from previous steps
                if not S.status.first:
                    if not S.status.prev_done:
                        self.buffers.e_em_last = self.recv(comm, S.status.slot - 1)
                else:
                    self.buffers.e_em_last = 0.0

                # estimate accumulated local error
                temp = self.estimate_embedded_error_serial(L)

                # estimate local error as difference of accumulated errors
                L.status.error_embedded_estimate = max([abs(temp - self.buffers.e_em_last), np.finfo(float).eps])

                # send the accumulated local errors forward
                if not S.status.last:
                    self.send(comm, dest=S.status.slot + 1, data=temp, blocking=True)

        return None


class EstimateEmbeddedErrorCollocation(ConvergenceController):
    """
    Estimates an embedded error based on changing the underlying quadrature rule. The error estimate is stored as
    `error_embedded_estimate_collocation` in the status of the level. Note that we only compute the estimate on the
    finest level. The error is stored as a tuple with the first index denoting to which iteration it belongs. This
    is useful since the error estimate is not available immediately after, but only when the next collocation problem
    is converged to make sure the two solutions are of different accuracy.

    Changing the collocation method between iterations happens using the `AdaptiveCollocation` convergence controller.
    Please refer to that for documentation on how to use this. Just pass the parameters for that convergence controller
    as `adaptive_coll_params` to the parameters for this one and they will be passed on when the `AdaptiveCollocation`
    convergence controller is automatically added while loading dependencies.
    """

    def setup(self, controller, params, description, **kwargs):
        """
        Add a default value for control order to the parameters

        Args:
            controller (pySDC.Controller): The controller
            params (dict): Parameters for the convergence controller
            description (dict): The description object used to instantiate the controller

        Returns:
            dict: Updated parameters
        """
        defaults = {
            "control_order": 210,
            "adaptive_coll_params": {},
            **super().setup(controller, params, description, **kwargs),
        }
        return defaults

    def dependencies(self, controller, description, **kwargs):
        """
        Load the `AdaptiveCollocation` convergence controller to switch between collocation problems between iterations.

        Args:
            controller (pySDC.Controller): The controller
            description (dict): The description object used to instantiate the controller
        """
        from pySDC.implementations.convergence_controller_classes.adaptive_collocation import AdaptiveCollocation

        controller.add_convergence_controller(
            AdaptiveCollocation, params=self.params.adaptive_coll_params, description=description
        )
        from pySDC.implementations.hooks.log_embedded_error_estimate import LogEmbeddedErrorEstimate

        controller.add_hook(LogEmbeddedErrorEstimate)

    def post_iteration_processing(self, controller, step, **kwargs):
        """
        Compute the embedded error as the difference between the interpolated and the current solution on the finest
        level.

        Args:
            controller (pySDC.Controller.controller): The controller
            step (pySDC.Step.step): The current step
        """
        if step.status.done:
            lvl = step.levels[0]
            lvl.sweep.compute_end_point()
            self.status.u += [lvl.uend]
            self.status.iter += [step.status.iter]

            if len(self.status.u) > 1:
                lvl.status.error_embedded_estimate_collocation = (
                    self.status.iter[-2],
                    max([np.finfo(float).eps, abs(self.status.u[-1] - self.status.u[-2])]),
                )

    def setup_status_variables(self, controller, **kwargs):
        """
        Add the embedded error variable to the levels and add a status variable for previous steps.

        Args:
            controller (pySDC.Controller): The controller
        """
        self.status = Status(['u', 'iter'])
        self.status.u = []  # the solutions of converged collocation problems
        self.status.iter = []  # the iteration in which the solution converged

        if 'comm' in kwargs.keys():
            steps = [controller.S]
        else:
            if 'active_slots' in kwargs.keys():
                steps = [controller.MS[i] for i in kwargs['active_slots']]
            else:
                steps = controller.MS
        where = ["levels", "status"]
        for S in steps:
            self.add_variable(S, name='error_embedded_estimate_collocation', where=where, init=None)

    def reset_status_variables(self, controller, **kwargs):
        self.setup_status_variables(controller, **kwargs)
