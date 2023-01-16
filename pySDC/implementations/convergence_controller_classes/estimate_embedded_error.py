import numpy as np

from pySDC.core.ConvergenceController import ConvergenceController, Pars
from pySDC.implementations.convergence_controller_classes.store_uold import StoreUOld
from pySDC.implementations.hooks.log_embedded_error_estimate import LogEmbeddedErrorEstimate

from pySDC.implementations.sweeper_classes.Runge_Kutta import RungeKutta


class EstimateEmbeddedError(ConvergenceController):
    """
    The embedded error is obtained by computing two solutions of different accuracy and pretending the more accurate
    one is an exact solution from the point of view of the less accurate solution. In practice, we like to compute the
    solutions with different order methods, meaning that in SDC we can just subtract two consecutive sweeps, as long as
    you make sure your preconditioner is compatible, which you have to just try out...
    """

    def __init__(self, controller, params, description, **kwargs):
        """
        Initialisation routine. Add the buffers for communication.

        Args:
            controller (pySDC.Controller): The controller
            params (dict): Parameters for the convergence controller
            description (dict): The description object used to instantiate the controller
        """
        super(EstimateEmbeddedError, self).__init__(controller, params, description, **kwargs)
        self.buffers = Pars({'e_em_last': 0.0})
        controller.add_hook(LogEmbeddedErrorEstimate)

    @classmethod
    def get_implementation(cls, flavor):
        """
        Retrieve the implementation for a specific flavor of this class.

        Args:
            flavor (str): The implementation that you want

        Returns:
            cls: The child class that implements the desired flavor
        """
        if flavor == 'MPI':
            return EstimateEmbeddedErrorMPI
        elif flavor == 'nonMPI':
            return EstimateEmbeddedErrorNonMPI
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
        return {"control_order": -80, "sweeper_type": sweeper_type, **params}

    def dependencies(self, controller, description, **kwargs):
        """
        Load the convergence controller that stores the solution of the last sweep unless we are doing Runge-Kutta

        Args:
            controller (pySDC.Controller): The controller
            description (dict): The description object used to instantiate the controller

        Returns:
            None
        """
        if RungeKutta not in description["sweeper_class"].__bases__:
            controller.add_convergence_controller(StoreUOld, description=description)
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


class EstimateEmbeddedErrorNonMPI(EstimateEmbeddedError):
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


class EstimateEmbeddedErrorMPI(EstimateEmbeddedError):
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
