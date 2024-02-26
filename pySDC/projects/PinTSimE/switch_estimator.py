import numpy as np
import scipy as sp

from pySDC.core.Errors import ParameterError
from pySDC.core.Collocation import CollBase
from pySDC.core.ConvergenceController import ConvergenceController, Status
from pySDC.implementations.convergence_controller_classes.check_convergence import CheckConvergence


class SwitchEstimator(ConvergenceController):
    """
    Class to predict the time point of the event and setting a new step size. For the first time, this is a nonMPI version,
    because a MPI version is not yet developed.
    """

    def setup(self, controller, params, description):
        r"""
        Function sets default variables to handle with the event at the beginning. The default params are:

        - control_order : controls the order of the SE's call of convergence controllers.
        - coll.nodes : defines the collocation nodes for interpolation.
        - tol_zero : inner tolerance for SE; state function has to satisfy it to terminate.
        - t_interp : interpolation axis with time points.
        - state_function : List of values from state function.

        Parameters
        ----------
        controller : pySDC.Controller
            The controller doing all the stuff in a computation.
        params : dict
            The parameters passed for this specific convergence controller.
        description : dict
            The description object used to instantiate the controller.

        Returns
        -------
        convergence_controller_params : dict
            The updated params dictionary.
        """

        # for RK4 sweeper, sweep.coll.nodes now consists of values of ButcherTableau
        # for this reason, collocation nodes will be generated here
        coll = CollBase(
            num_nodes=description['sweeper_params']['num_nodes'],
            quad_type=description['sweeper_params']['quad_type'],
        )

        defaults = {
            'control_order': 0,
            'nodes': coll.nodes,
            'tol_zero': 1e-13,
            't_interp': [],
            'state_function': [],
        }
        return {**defaults, **params}

    def setup_status_variables(self, controller, **kwargs):
        """
        Adds switching specific variables to status variables.

        Parameters
        ----------
        controller : pySDC.Controller
            The controller doing all the stuff in a computation.
        """

        self.status = Status(['is_zero', 'switch_detected', 't_switch'])

    def reset_status_variables(self, controller, **kwargs):
        """
        Resets status variables.

        Parameters
        ----------
        controller : pySDC.Controller
            The controller doing all the stuff in a computation.
        """

        self.setup_status_variables(controller, **kwargs)

    def get_new_step_size(self, controller, S, **kwargs):
        """
        Determine a new step size when an event is found such that the event occurs at the time step.

        Parameters
        ----------
        controller : pySDC.Controller
            The controller doing all the stuff in a computation.
        S : pySDC.Step
            The current step.
        """

        L = S.levels[0]

        if CheckConvergence.check_convergence(S):
            self.status.switch_detected, m_guess, self.params.state_function = L.prob.get_switching_info(L.u, L.time)

            if self.status.switch_detected:
                self.params.t_interp = [L.time + L.dt * self.params.nodes[m] for m in range(len(self.params.nodes))]
                self.params.t_interp, self.params.state_function = self.adapt_interpolation_info(
                    L.time, L.sweep.coll.left_is_node, self.params.t_interp, self.params.state_function
                )

                # when the state function is already close to zero the event is already resolved well
                if (
                    abs(self.params.state_function[-1]) <= self.params.tol_zero
                    or abs(self.params.state_function[0]) <= self.params.tol_zero
                ):
                    if abs(self.params.state_function[0]) <= self.params.tol_zero:
                        t_switch = self.params.t_interp[0]
                        boundary = 'left'
                    elif abs(self.params.state_function[-1]) <= self.params.tol_zero:
                        boundary = 'right'
                        t_switch = self.params.t_interp[-1]

                    msg = f"The value of state function is close to zero, thus event time is already close enough to the {boundary} end point!"
                    self.log(msg, S)
                    self.log_event_time(
                        controller.hooks[0], S.status.slot, L.time, L.level_index, L.status.sweep, t_switch
                    )

                    L.prob.count_switches()
                    self.status.is_zero = True

                # intermediate value theorem states that a root is contained in current step
                if self.params.state_function[0] * self.params.state_function[-1] < 0 and self.status.is_zero is None:
                    self.status.t_switch = self.get_switch(
                        self.params.t_interp, self.params.state_function, m_guess, self.params.typeFD
                    )

                    self.logging_during_estimation(
                        controller.hooks[0],
                        S.status.slot,
                        L.time,
                        L.level_index,
                        L.status.sweep,
                        self.status.t_switch,
                        self.params.state_function,
                    )

                    if L.time < self.status.t_switch < L.time + L.dt:
                        dt_switch = (self.status.t_switch - L.time) * self.params.alpha

                        if (
                            abs(self.status.t_switch - L.time) <= self.params.tol
                            or abs((L.time + L.dt) - self.status.t_switch) <= self.params.tol
                        ):
                            self.log(f"Switch located at time {self.status.t_switch:.15f}", S)
                            L.prob.t_switch = self.status.t_switch
                            self.log_event_time(
                                controller.hooks[0],
                                S.status.slot,
                                L.time,
                                L.level_index,
                                L.status.sweep,
                                self.status.t_switch,
                            )

                            L.prob.count_switches()

                        else:
                            self.log(f"Located Switch at time {self.status.t_switch:.15f} is outside the range", S)

                        # when an event is found, step size matching with this event should be preferred
                        dt_planned = L.status.dt_new if L.status.dt_new is not None else L.params.dt
                        if self.status.switch_detected:
                            L.status.dt_new = dt_switch
                        else:
                            L.status.dt_new = min([dt_planned, dt_switch])

                    else:
                        # event occurs on L.time or L.time + L.dt; no restart necessary
                        boundary = 'left boundary' if self.status.t_switch == L.time else 'right boundary'
                        self.log(f"Estimated switch {self.status.t_switch:.15f} occurs at {boundary}", S)
                        self.log_event_time(
                            controller.hooks[0],
                            S.status.slot,
                            L.time,
                            L.level_index,
                            L.status.sweep,
                            self.status.t_switch,
                        )
                        L.prob.count_switches()
                        self.status.switch_detected = False

                else:  # intermediate value theorem is not satisfied
                    self.status.switch_detected = False

    def determine_restart(self, controller, S, **kwargs):
        """
        Check if the step needs to be restarted due to a predicting switch.

        Parameters
        ----------
        controller : pySDC.Controller
            The controller doing all the stuff in a computation.
        S : pySDC.Step
            The current step.
        """

        if self.status.switch_detected:
            S.status.restart = True
            S.status.force_done = True

        super().determine_restart(controller, S, **kwargs)

    def post_step_processing(self, controller, S, **kwargs):
        """
        After a step is done, some variables will be prepared for predicting a possibly new switch.
        If no Adaptivity is used, the next time step will be set as the default one from the front end.

        Parameters
        ----------
        controller : pySDC.Controller
            The controller doing all the stuff in a computation.
        S : pySDC.Step
            The current step.
        """

        L = S.levels[0]

        if self.status.t_switch is None:
            L.status.dt_new = L.status.dt_new if L.status.dt_new is not None else L.params.dt_initial

        super().post_step_processing(controller, S, **kwargs)

    @staticmethod
    def log_event_time(controller_hooks, process, time, level, sweep, t_switch):
        """
        Logs the event time of an event satisfying an appropriate criterion, e.g., event is already resolved well,
        event time satisfies tolerance.

        Parameters
        ----------
        controller_hooks : pySDC.Controller.hooks
            Controller with access to the hooks.
        process : int
            Process for logging.
        time : float
            Time at which the event time is logged (denotes the current step).
        level : int
            Level at which event is found.
        sweep : int
            Denotes the number of sweep.
        t_switch : float
            Event time founded by switch estimation.
        """

        controller_hooks.add_to_stats(
            process=process,
            time=time,
            level=level,
            iter=0,
            sweep=sweep,
            type='switch',
            value=t_switch,
        )

    @staticmethod
    def logging_during_estimation(controller_hooks, process, time, level, sweep, t_switch, state_function):
        controller_hooks.add_to_stats(
            process=process,
            time=time,
            level=level,
            iter=0,
            sweep=sweep,
            type='switch_all',
            value=t_switch,
        )
        controller_hooks.add_to_stats(
            process=process,
            time=time,
            level=level,
            iter=0,
            sweep=sweep,
            type='h_all',
            value=max([abs(item) for item in state_function]),
        )

    @staticmethod
    def get_switch(t_interp, state_function, m_guess, choiceFD='backward'):
        r"""
        Routine to do the interpolation and root finding stuff.

        Parameters
        ----------
        t_interp : list
            Collocation nodes in a step.
        state_function : list
            Contains values of state function at these collocation nodes.
        m_guess : float
            Index at which the difference drops below zero.
        choiceFD : str
            Finite difference to be used to approximate derivative of
            state function. Can be ``'forward'``, ``'centered'``, or
            ``'backward'``. Default is ``'backward'``

        Returns
        -------
        t_switch : float
           Time point of found event.
        """
        LinearInterpolator = LagrangeInterpolation(t_interp, state_function)
        p = lambda t: LinearInterpolator.eval(t)

        def fprime(t):
            r"""
            Computes the derivative of the scalar interpolant using finite difference.
            Here, different finite differences can be used. The type of FD can be set by
            setting ``typeFD`` in switch estimator parameters. There are three choices possible:

            - ``typeFD='backward'`` for :math:`h=10^{-10}`:

                .. math::
                \frac{dp}{dt} \approx \frac{p(t) - p(t - h)}{h}

            - ``typeFD='centered'`` for :math:`h=10^{-12}`:

                .. math::
                \frac{dp}{dt} \approx \frac{p(t + h) - p(t - h)}{2h}

            - ``typeFD='forward'`` for :math:`h=10^{-10}`:

                .. math::
                \frac{dp}{dt} \approx \frac{p(t + h) - p(t)}{h}

            Parameters
            ----------
            t : float
                Time where the derivatives is computed.

            Returns
            -------
            dp : float
                Derivative of interpolation p at time t.
            """

            if choiceFD == 'forward':
                dt_FD = 1e-10
                dp = (p(t + dt_FD) - p(t)) / (dt_FD)
            elif choiceFD == 'centered':
                dt_FD = 1e-12
                dp = (p(t + dt_FD) - p(t - dt_FD)) / (2 * dt_FD)
            elif choiceFD == 'backward':
                dt_FD = 1e-12
                # dp = (p(t) - p(t - dt_FD)) / (dt_FD)
                # dp = (11 * p(t) - 18 * p(t - dt_FD) + 9 * p(t - 2 * dt_FD) - 2 * p(t - 3 * dt_FD)) / (6 * dt_FD)
                dp = (25 * p(t) - 48 * p(t - dt_FD) + 36 * p(t - 2 * dt_FD) - 16 * p(t - 3 * dt_FD) + 3 * p(t - 4 * dt_FD)) / (12 * dt_FD)
            else:
                raise NotImplementedError
            return dp

        newton_tol, newton_maxiter = 1e-15, 100
        t_switch = newton(t_interp[m_guess], p, fprime, newton_tol, newton_maxiter)
        return t_switch

    @staticmethod
    def adapt_interpolation_info(t, left_is_node, t_interp, state_function):
        """
        Adapts the x- and y-axis for interpolation. For SDC, it is proven whether the left boundary is a
        collocation node or not. In case it is, the first entry of the state function has to be removed,
        because it would otherwise contain double values on starting time and the first node. Otherwise,
        starting time L.time has to be added to t_interp to also take this value in the interpolation
        into account.

        Parameters
        ----------
        t : float
            Starting time of the step.
        left_is_node : bool
            Indicates whether the left boundary is a collocation node or not.
        t_interp : list
            x-values for interpolation containing collocation nodes.
        state_function : list
            y-values for interpolation containing values of state function.

        Returns
        -------
        t_interp : list
            Adapted x-values for interpolation containing collocation nodes.
        state_function : list
            Adapted y-values for interpolation containing values of state function.
        """

        if not left_is_node:
            t_interp.insert(0, t)
        else:
            del state_function[0]

        return t_interp, state_function


def newton(x0, p, fprime, newton_tol, newton_maxiter):
    """
    Newton's method fo find the root of interpolant p.

    Parameters
    ----------
    x0 : float
        Initial guess.
    p : callable
        Interpolated function where Newton's method is applied at.
    fprime : callable
        Approximated derivative of p using finite differences.
    newton_tol : float
        Tolerance for termination.
    newton_maxiter : int
        Maximum of iterations the method should execute.

    Returns
    -------
    root : float
        Root of function p.
    """

    n = 0
    while n < newton_maxiter:
        res = abs(p(x0))
        if res < newton_tol or np.isnan(p(x0)) and np.isnan(fprime(x0)) or np.isclose(fprime(x0), 0.0):
            break

        x0 -= 1.0 / fprime(x0) * p(x0)

        n += 1

    if n == newton_maxiter:
        msg = f'Newton did not converge after {n} iterations, error is {res}'
    else:
        msg = f'Newton did converge after {n} iterations, error for root {x0} is {res}'
    # print(msg)

    root = x0
    return root


class LagrangeInterpolation(object):
    def __init__(self, ti, yi):
        """Initialization routine"""
        self.ti = np.asarray(ti)
        self.yi = np.asarray(yi)
        self.n = len(ti)

    def get_Lagrange_polynomial(self, t, i):
        """
        Computes the basis of the i-th Lagrange polynomial.

        Parameters
        ----------
        t : float
            Time where the polynomial is computed at.
        i : int
            Index of the Lagrange polynomial

        Returns
        -------
        product : float
            The product of the bases.
        """
        product = np.prod([(t - self.ti[k]) / (self.ti[i] - self.ti[k]) for k in range(self.n) if k != i])
        return product

    def eval(self, t):
        """
        Evaluates the Lagrange interpolation at time t.

        Parameters
        ----------
        t : float
            Time where interpolation is computed.

        Returns
        -------
        p : float
            Value of interpolant at time t.
        """
        p = np.sum([self.yi[i] * self.get_Lagrange_polynomial(t, i) for i in range(self.n)])
        return p
