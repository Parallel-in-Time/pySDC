import numpy as np
from pySDC.core.ConvergenceController import ConvergenceController, Status
from pySDC.implementations.convergence_controller_classes.step_size_limiter import (
    StepSizeLimiter,
)
from pySDC.implementations.convergence_controller_classes.basic_restarting import (
    BasicRestartingNonMPI,
)
from pySDC.implementations.hooks.log_step_size import LogStepSize


class AdaptivityBase(ConvergenceController):
    """
    Abstract base class for convergence controllers that implement adaptivity based on arbitrary local error estimates
    and update rules.
    """

    def setup(self, controller, params, description, **kwargs):
        """
        Define default parameters here.

        Default parameters are:
         - control_order (int): The order relative to other convergence controllers
         - beta (float): The safety factor

        Args:
            controller (pySDC.Controller): The controller
            params (dict): The params passed for this specific convergence controller
            description (dict): The description object used to instantiate the controller

        Returns:
            (dict): The updated params dictionary
        """
        defaults = {
            "control_order": -50,
            "beta": 0.9,
        }
        controller.add_hook(LogStepSize)
        return {**defaults, **super().setup(controller, params, description, **kwargs)}

    def dependencies(self, controller, description, **kwargs):
        """
        Load step size limiters here, if they are desired.

        Args:
            controller (pySDC.Controller): The controller
            description (dict): The description object used to instantiate the controller

        Returns:
            None
        """

        if "dt_min" in self.params.__dict__.keys() or "dt_max" in self.params.__dict__.keys():
            step_limiter_params = dict()
            step_limiter_params["dt_min"] = self.params.__dict__.get("dt_min", 0)
            step_limiter_params["dt_max"] = self.params.__dict__.get("dt_max", np.inf)
            controller.add_convergence_controller(StepSizeLimiter, params=step_limiter_params, description=description)

        return None

    def get_new_step_size(self, controller, S, **kwargs):
        """
        Determine a step size for the next step from an estimate of the local error of the current step.

        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.Step): The current step

        Returns:
            None
        """
        raise NotImplementedError("Please implement a rule for updating the step size!")

    def compute_optimal_step_size(self, beta, dt, e_tol, e_est, order):
        """
        Compute the optimal step size for the current step based on the order of the scheme.
        This function can be called from `get_new_step_size` for various implementations of adaptivity, but notably not
        all! We require to know the order of the error estimate and if we do adaptivity based on the residual, for
        instance, we do not know that and we can't use this function.

        Args:
            beta (float): Safety factor
            dt (float): Current step size
            e_tol (float): The desired tolerance
            e_est (float): The estimated local error
            order (int): The order of the local error estimate

        Returns:
            float: The optimal step size
        """
        return beta * dt * (e_tol / e_est) ** (1.0 / order)

    def get_local_error_estimate(self, controller, S, **kwargs):
        """
        Get the local error estimate for updating the step size.
        It does not have to be an error estimate, but could be the residual or something else.

        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.Step): The current step

        Returns:
            float: The error estimate
        """
        raise NotImplementedError("Please implement a way to get the local error")

    def determine_restart(self, controller, S, **kwargs):
        """
        Check if the step wants to be restarted by comparing the estimate of the local error to a preset tolerance

        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.Step): The current step

        Returns:
            None
        """
        if S.status.iter >= S.params.maxiter:
            e_est = self.get_local_error_estimate(controller, S)
            if e_est >= self.params.e_tol:
                # see if we try to avoid restarts
                if self.params.get('avoid_restarts'):
                    more_iter_needed = max([L.status.iter_to_convergence for L in S.levels])
                    rho = max([L.status.contraction_factor for L in S.levels])

                    if rho > 1:
                        S.status.restart = True
                        self.log(f"Convergence factor = {rho:.2e} > 1 -> restarting", S)
                    elif S.status.iter + more_iter_needed > 2 * S.params.maxiter:
                        S.status.restart = True
                        self.log(f"{more_iter_needed} more iterations needed for convergence -> restart", S)
                    else:
                        S.status.force_continue = True
                        self.log(f"{more_iter_needed} more iterations needed for convergence -> no restart", S)
                else:
                    S.status.restart = True
                    self.log(f"Restarting: e={e_est:.2e} >= e_tol={self.params.e_tol:.2e}", S)

        return None


class Adaptivity(AdaptivityBase):
    """
    Class to compute time step size adaptively based on embedded error estimate.

    We have a version working in non-MPI pipelined SDC, but Adaptivity requires you to know the order of the scheme,
    which you can also know for block-Jacobi, but it works differently and it is only implemented for block
    Gauss-Seidel so far.

    There is an option to reduce restarts if continued iterations could yield convergence in fewer iterations than
    restarting based on an estimate of the contraction factor.
    Since often only one or two more iterations suffice, this can boost efficiency of adaptivity significantly.
    Notice that the computed step size is not effected.
    Be aware that this does not work when Hot Rod is enabled, since that requires us to know the order of the scheme in
    more detail. Since we reset to the second to last sweep before moving on, we cannot continue to iterate.
    Set the reduced restart up by setting a boolean value for "avoid_restarts" in the parameters for the convergence
    controller.
    The behaviour in multi-step SDC is not well studied and it is unclear if anything useful happens there.
    """

    def dependencies(self, controller, description, **kwargs):
        """
        Load the embedded error estimator.

        Args:
            controller (pySDC.Controller): The controller
            description (dict): The description object used to instantiate the controller

        Returns:
            None
        """
        from pySDC.implementations.convergence_controller_classes.estimate_embedded_error import EstimateEmbeddedError

        super(Adaptivity, self).dependencies(controller, description)

        controller.add_convergence_controller(
            EstimateEmbeddedError.get_implementation("nonMPI" if not self.params.useMPI else "MPI"),
            description=description,
        )

        # load contraction factor estimator if necessary
        if self.params.get('avoid_restarts'):
            from pySDC.implementations.convergence_controller_classes.estimate_contraction_factor import (
                EstimateContractionFactor,
            )

            params = {'e_tol': self.params.e_tol}
            controller.add_convergence_controller(EstimateContractionFactor, description=description, params=params)
        return None

    def check_parameters(self, controller, params, description, **kwargs):
        """
        Check whether parameters are compatible with whatever assumptions went into the step size functions etc.
        For adaptivity, we need to know the order of the scheme.

        Args:
            controller (pySDC.Controller): The controller
            params (dict): The params passed for this specific convergence controller
            description (dict): The description object used to instantiate the controller

        Returns:
            bool: Whether the parameters are compatible
            str: The error message
        """
        if description["level_params"].get("restol", -1.0) >= 0:
            return (
                False,
                "Adaptivity needs constant order in time and hence restol in the step parameters has to be \
smaller than 0!",
            )

        if controller.params.mssdc_jac:
            return (
                False,
                "Adaptivity needs the same order on all steps, please activate Gauss-Seidel multistep mode!",
            )

        if "e_tol" not in params.keys():
            return (
                False,
                "Adaptivity needs a local tolerance! Please pass `e_tol` to the parameters for this convergence controller!",
            )

        return True, ""

    def get_new_step_size(self, controller, S, **kwargs):
        """
        Determine a step size for the next step from an embedded estimate of the local error of the current step.

        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.Step): The current step

        Returns:
            None
        """
        # check if we performed the desired amount of sweeps
        if S.status.iter == S.params.maxiter:
            L = S.levels[0]

            # compute next step size
            order = S.status.iter  # embedded error estimate is same order as time marching

            e_est = self.get_local_error_estimate(controller, S)
            L.status.dt_new = self.compute_optimal_step_size(
                self.params.beta, L.params.dt, self.params.e_tol, e_est, order
            )
            self.log(f'Adjusting step size from {L.params.dt:.2e} to {L.status.dt_new:.2e}', S)

        return None

    def get_local_error_estimate(self, controller, S, **kwargs):
        """
        Get the embedded error estimate of the finest level of the step.

        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.Step): The current step

        Returns:
            float: Embedded error estimate
        """
        return S.levels[0].status.error_embedded_estimate


class AdaptivityRK(Adaptivity):
    """
    Adaptivity for Runge-Kutta methods. Basically, we need to change the order in the step size update
    """

    def check_parameters(self, controller, params, description, **kwargs):
        """
        Check whether parameters are compatible with whatever assumptions went into the step size functions etc.
        For adaptivity, we need to know the order of the scheme.

        Args:
            controller (pySDC.Controller): The controller
            params (dict): The params passed for this specific convergence controller
            description (dict): The description object used to instantiate the controller

        Returns:
            bool: Whether the parameters are compatible
            str: The error message
        """
        if "update_order" not in params.keys():
            return (
                False,
                "Adaptivity needs an order for the update rule! Please set some up in \
description['convergence_control_params']['update_order']!",
            )

        return super(AdaptivityRK, self).check_parameters(controller, params, description)

    def get_new_step_size(self, controller, S, **kwargs):
        """
        Determine a step size for the next step from an embedded estimate of the local error of the current step.

        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.Step): The current step

        Returns:
            None
        """
        # check if we performed the desired amount of sweeps
        if S.status.iter == S.params.maxiter:
            L = S.levels[0]

            # compute next step size
            order = self.params.update_order
            e_est = self.get_local_error_estimate(controller, S)
            L.status.dt_new = self.compute_optimal_step_size(
                self.params.beta, L.params.dt, self.params.e_tol, e_est, order
            )
            self.log(f'Adjusting step size from {L.params.dt:.2e} to {L.status.dt_new:.2e}', S)

        return None


class AdaptivityResidual(AdaptivityBase):
    """
    Do adaptivity based on residual.

    Since we don't know a correlation between the residual and the error (for nonlinear problems), we employ a simpler
    rule to update the step size. Instead of giving a local tolerance that we try to hit as closely as possible, we set
    two thresholds for the residual. When we exceed the upper one, we reduce the step size by a factor of 2 and if the
    residual falls below the lower threshold, we double the step size.
    Please setup these parameters as "e_tol" and "e_tol_low".
    """

    def setup(self, controller, params, description, **kwargs):
        """
        Define default parameters here.

        Default parameters are:
         - control_order (int): The order relative to other convergence controllers
         - e_tol_low (float): Lower absolute threshold for the residual
         - e_tol (float): Upper absolute threshold for the residual
         - max_restarts: Override maximum number of restarts

        Args:
            controller (pySDC.Controller): The controller
            params (dict): The params passed for this specific convergence controller
            description (dict): The description object used to instantiate the controller

        Returns:
            (dict): The updated params dictionary
        """
        defaults = {
            "control_order": -45,
            "e_tol_low": 0,
            "e_tol": np.inf,
            "max_restarts": 99 if "e_tol_low" in params else None,
        }
        return {**defaults, **params}

    def setup_status_variables(self, controller, **kwargs):
        """
        Change maximum number of allowed restarts here.

        Args:
            controller (pySDC.Controller): The controller

        Reutrns:
            None
        """
        if self.params.max_restarts is not None:
            conv_controllers = controller.convergence_controllers
            restart_cont = [me for me in conv_controllers if type(me) == BasicRestartingNonMPI]

            if len(restart_cont) == 0:
                raise NotImplementedError("Please implement override of maximum number of restarts!")

            restart_cont[0].params.max_restarts = self.params.max_restarts
        return None

    def check_parameters(self, controller, params, description, **kwargs):
        """
        Check whether parameters are compatible with whatever assumptions went into the step size functions etc.
        For adaptivity, we want a fixed order of the scheme.

        Args:
            controller (pySDC.Controller): The controller
            params (dict): The params passed for this specific convergence controller
            description (dict): The description object used to instantiate the controller

        Returns:
            bool: Whether the parameters are compatible
            str: The error message
        """
        if description["level_params"].get("restol", -1.0) >= 0:
            return (
                False,
                "Adaptivity needs constant order in time and hence restol in the step parameters has to be \
smaller than 0!",
            )

        if controller.params.mssdc_jac:
            return (
                False,
                "Adaptivity needs the same order on all steps, please activate Gauss-Seidel multistep mode!",
            )

        return True, ""

    def get_new_step_size(self, controller, S, **kwargs):
        """
        Determine a step size for the next step.
        If we exceed the absolute tolerance of the residual in either direction, we either double or halve the step
        size.

        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.Step): The current step

        Returns:
            None
        """
        # check if we performed the desired amount of sweeps
        if S.status.iter == S.params.maxiter:
            L = S.levels[0]

            res = self.get_local_error_estimate(controller, S)

            dt_planned = L.status.dt_new if L.status.dt_new is not None else L.params.dt

            if res > self.params.e_tol:
                L.status.dt_new = min([dt_planned, L.params.dt / 2.0])
                self.log(f'Adjusting step size from {L.params.dt:.2e} to {L.status.dt_new:.2e}', S)
            elif res < self.params.e_tol_low:
                L.status.dt_new = max([dt_planned, L.params.dt * 2.0])
                self.log(f'Adjusting step size from {L.params.dt:.2e} to {L.status.dt_new:.2e}', S)

        return None

    def get_local_error_estimate(self, controller, S, **kwargs):
        """
        Get the residual of the finest level of the step.

        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.Step): The current step

        Returns:
            float: Embedded error estimate
        """
        return S.levels[0].status.residual


class AdaptivityCollocation(AdaptivityBase):
    """
    Control the step size via a collocation based estimate of the local error.
    The error estimate works by subtracting two solutions to collocation problems with different order. You can
    interpolate between collocation methods as much as you want but the adaptive step size selection will always be
    based on the last switch of quadrature.
    """

    def setup(self, controller, params, description, **kwargs):
        """
        Add a default value for control order to the parameters.

        Args:
            controller (pySDC.Controller): The controller
            params (dict): Parameters for the convergence controller
            description (dict): The description object used to instantiate the controller

        Returns:
            dict: Updated parameters
        """
        defaults = {
            "adaptive_coll_params": {},
            "num_colls": 0,
            **super().setup(controller, params, description, **kwargs),
            "control_order": 220,
        }

        for key in defaults['adaptive_coll_params'].keys():
            if type(defaults['adaptive_coll_params'][key]) == list:
                defaults['num_colls'] = max([defaults['num_colls'], len(defaults['adaptive_coll_params'][key])])

        return defaults

    def setup_status_variables(self, controller, **kwargs):
        self.status = Status(['error', 'order'])
        self.status.error = []
        self.status.order = []

    def reset_status_variables(self, controller, **kwargs):
        self.setup_status_variables(controller, **kwargs)

    def dependencies(self, controller, description, **kwargs):
        """
        Load the `EstimateEmbeddedErrorCollocation` convergence controller to estimate the local error by switching
        between collocation problems between iterations.

        Args:
            controller (pySDC.Controller): The controller
            description (dict): The description object used to instantiate the controller
        """
        from pySDC.implementations.convergence_controller_classes.estimate_embedded_error import (
            EstimateEmbeddedErrorCollocation,
        )

        super().dependencies(controller, description)

        params = {'adaptive_coll_params': self.params.adaptive_coll_params}
        controller.add_convergence_controller(
            EstimateEmbeddedErrorCollocation,
            params=params,
            description=description,
        )

    def get_local_error_estimate(self, controller, S, **kwargs):
        """
        Get the collocation based embedded error estimate.

        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.Step): The current step

        Returns:
            float: Embedded error estimate
        """
        if len(self.status.error) > 1:
            return self.status.error[-1][1]
        else:
            return 0.0

    def post_iteration_processing(self, controller, step, **kwargs):
        """
        Get the error estimate and its order if available.

        Args:
            controller (pySDC.Controller.controller): The controller
            step (pySDC.Step.step): The current step
        """
        if step.status.done:
            lvl = step.levels[0]
            self.status.error += [lvl.status.error_embedded_estimate_collocation]
            self.status.order += [lvl.sweep.coll.order]

    def get_new_step_size(self, controller, S, **kwargs):
        if len(self.status.order) == self.params.num_colls:
            lvl = S.levels[0]

            # compute next step size
            order = (
                min(self.status.order[-2::]) + 1
            )  # local order of less accurate of the last two collocation problems
            e_est = self.get_local_error_estimate(controller, S)

            lvl.status.dt_new = self.compute_optimal_step_size(
                self.params.beta, lvl.params.dt, self.params.e_tol, e_est, order
            )
            self.log(f'Adjusting step size from {lvl.params.dt:.2e} to {lvl.status.dt_new:.2e}', S)

    def check_parameters(self, controller, params, description, **kwargs):
        """
        Check whether parameters are compatible with whatever assumptions went into the step size functions etc.
        For adaptivity, we need to know the order of the scheme.

        Args:
            controller (pySDC.Controller): The controller
            params (dict): The params passed for this specific convergence controller
            description (dict): The description object used to instantiate the controller

        Returns:
            bool: Whether the parameters are compatible
            str: The error message
        """
        if controller.params.mssdc_jac:
            return (
                False,
                "Adaptivity needs the same order on all steps, please activate Gauss-Seidel multistep mode!",
            )

        if "e_tol" not in params.keys():
            return (
                False,
                "Adaptivity needs a local tolerance! Please pass `e_tol` to the parameters for this convergence controller!",
            )

        return True, ""

    def determine_restart(self, controller, S, **kwargs):
        """
        Check if the step wants to be restarted by comparing the estimate of the local error to a preset tolerance

        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.Step): The current step

        Returns:
            None
        """
        if len(self.status.order) == self.params.num_colls:
            e_est = self.get_local_error_estimate(controller, S)
            if e_est >= self.params.e_tol:
                S.status.restart = True
                self.log(f"Restarting: e={e_est:.2e} >= e_tol={self.params.e_tol:.2e}", S)
