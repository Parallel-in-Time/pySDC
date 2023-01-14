import numpy as np

from pySDC.core.Errors import ParameterError, ProblemError
from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh


class battery(ptype):
    """
    Example implementing the battery drain model as in the description in the PinTSimE project
    Attributes:
        A: system matrix, representing the 2 ODEs
    """

    def __init__(self, problem_params, dtype_u=mesh, dtype_f=imex_mesh):
        """
        Initialization routine
        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: mesh data type for solution
            dtype_f: mesh data type for RHS
        """

        problem_params['nvars'] = 2

        # these parameters will be used later, so assert their existence
        essential_keys = ['Vs', 'Rs', 'C', 'R', 'L', 'alpha', 'V_ref']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(battery, self).__init__(
            init=(problem_params['nvars'], None, np.dtype('float64')),
            dtype_u=dtype_u,
            dtype_f=dtype_f,
            params=problem_params,
        )

        self.A = np.zeros((2, 2))
        self.t_switch = None
        self.count_switches = 0

    def eval_f(self, u, t):
        """
        Routine to evaluate the RHS
        Args:
            u (dtype_u): current values
            t (float): current time
        Returns:
            dtype_f: the RHS
        """

        f = self.dtype_f(self.init, val=0.0)
        f.impl[:] = self.A.dot(u)

        if self.t_switch is not None:
            if t >= self.t_switch:
                f.expl[0] = self.params.Vs / self.params.L

            else:
                f.expl[0] = 0

        else:
            if u[1] <= self.params.V_ref:
                f.expl[0] = self.params.Vs / self.params.L

            else:
                f.expl[0] = 0

        return f

    def solve_system(self, rhs, factor, u0, t):
        """
        Simple linear solver for (I-factor*A)u = rhs
        Args:
            rhs (dtype_f): right-hand side for the linear system
            factor (float): abbrev. for the local stepsize (or any other factor required)
            u0 (dtype_u): initial guess for the iterative solver
            t (float): current time (e.g. for time-dependent BCs)
        Returns:
            dtype_u: solution as mesh
        """
        self.A = np.zeros((2, 2))

        if self.t_switch is not None:
            if t >= self.t_switch:
                self.A[0, 0] = -(self.params.Rs + self.params.R) / self.params.L
            else:
                self.A[1, 1] = -1 / (self.params.C * self.params.R)

        else:
            if rhs[1] <= self.params.V_ref:
                self.A[0, 0] = -(self.params.Rs + self.params.R) / self.params.L
            else:
                self.A[1, 1] = -1 / (self.params.C * self.params.R)

        me = self.dtype_u(self.init)
        me[:] = np.linalg.solve(np.eye(self.params.nvars) - factor * self.A, rhs)
        return me

    def u_exact(self, t):
        """
        Routine to compute the exact solution at time t
        Args:
            t (float): current time
        Returns:
            dtype_u: exact solution
        """
        assert t == 0, 'ERROR: u_exact only valid for t=0'

        me = self.dtype_u(self.init)

        me[0] = 0.0  # cL
        me[1] = self.params.alpha * self.params.V_ref  # vC

        return me

    def get_switching_info(self, u, t):
        """
        Provides information about a discrete event for one subinterval.
        Args:
            u (dtype_u): current values
            t (float): current time
        Returns:
            switch_detected (bool): Indicates if a switch is found or not
            m_guess (np.int): Index of where the discrete event would found
            vC_switch (list): Contains function values of switching condition (for interpolation)
        """

        switch_detected = False
        m_guess = -100

        for m in range(len(u)):
            if u[m][1] - self.params.V_ref <= 0:
                switch_detected = True
                m_guess = m - 1
                break

        vC_switch = []
        if switch_detected:
            for m in range(1, len(u)):
                vC_switch.append(u[m][1] - self.params.V_ref)

        return switch_detected, m_guess, vC_switch

    def set_counter(self):
        """
        Counts the number of switches found.
        """

        self.count_switches += 1


class battery_implicit(ptype):
    """
    Example implementing the battery drain model as in the description in the PinTSimE project
    Attributes:
        A: system matrix, representing the 2 ODEs
    """

    def __init__(self, problem_params, dtype_u=mesh, dtype_f=mesh):
        """
        Initialization routine
        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: mesh data type for solution
            dtype_f: mesh data type for RHS
        """

        problem_params['nvars'] = 2

        # these parameters will be used later, so assert their existence
        essential_keys = [
            'newton_maxiter',
            'newton_tol',
            'Vs',
            'Rs',
            'C',
            'R',
            'L',
            'alpha',
            'V_ref',
        ]
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(battery_implicit, self).__init__(
            init=(problem_params['nvars'], None, np.dtype('float64')),
            dtype_u=dtype_u,
            dtype_f=dtype_f,
            params=problem_params,
        )

        self.A = np.zeros((2, 2))
        self.t_switch = None
        self.count_switches = 0
        self.newton_itercount = 0
        self.lin_itercount = 0
        self.newton_ncalls = 0
        self.lin_ncalls = 0

    def eval_f(self, u, t):
        """
        Routine to evaluate the RHS
        Args:
            u (dtype_u): current values
            t (float): current time
        Returns:
            dtype_f: the RHS
        """

        f = self.dtype_f(self.init, val=0.0)
        non_f = np.zeros(2)

        if self.t_switch is not None:
            if t >= self.t_switch:
                self.A[0, 0] = -(self.params.Rs + self.params.R) / self.params.L
                non_f[0] = self.params.Vs / self.params.L
            else:
                self.A[1, 1] = -1 / (self.params.C * self.params.R)
                non_f[0] = 0

        else:
            if u[1] <= self.params.V_ref:
                self.A[0, 0] = -(self.params.Rs + self.params.R) / self.params.L
                non_f[0] = self.params.Vs / self.params.L
            else:
                self.A[1, 1] = -1 / (self.params.C * self.params.R)
                non_f[0] = 0

        f[:] = self.A.dot(u) + non_f

        return f

    def solve_system(self, rhs, factor, u0, t):
        """
        Simple Newton solver
        Args:
            rhs (dtype_f): right-hand side for the linear system
            factor (float): abbrev. for the local stepsize (or any other factor required)
            u0 (dtype_u): initial guess for the iterative solver
            t (float): current time (e.g. for time-dependent BCs)
        Returns:
            dtype_u: solution as mesh
        """

        u = self.dtype_u(u0)
        non_f = np.zeros(2)
        self.A = np.zeros((2, 2))

        if self.t_switch is not None:
            if t >= self.t_switch:
                self.A[0, 0] = -(self.params.Rs + self.params.R) / self.params.L
                non_f[0] = self.params.Vs / self.params.L
            else:
                self.A[1, 1] = -1 / (self.params.C * self.params.R)
                non_f[0] = 0

        else:
            if rhs[1] <= self.params.V_ref:
                self.A[0, 0] = -(self.params.Rs + self.params.R) / self.params.L
                non_f[0] = self.params.Vs / self.params.L
            else:
                self.A[1, 1] = -1 / (self.params.C * self.params.R)
                non_f[0] = 0

        # start newton iteration
        n = 0
        res = 99
        while n < self.params.newton_maxiter:
            # form function g with g(u) = 0
            g = u - rhs - factor * (self.A.dot(u) + non_f)

            # if g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)

            if res < self.params.newton_tol:
                break

            # assemble dg
            dg = np.eye(self.params.nvars) - factor * self.A

            # newton update: u1 = u0 - g/dg
            u -= np.linalg.solve(dg, g)

            # increase iteration count
            n += 1

        if np.isnan(res) and self.params.stop_at_nan:
            raise ProblemError('Newton got nan after %i iterations, aborting...' % n)
        elif np.isnan(res):
            self.logger.warning('Newton got nan after %i iterations...' % n)

        if n == self.params.newton_maxiter:
            self.logger.warning('Newton did not converge after %i iterations, error is %s' % (n, res))

        self.newton_ncalls += 1
        self.newton_itercount += n

        me = self.dtype_u(self.init)
        me[:] = u[:]

        return me

    def u_exact(self, t):
        """
        Routine to compute the exact solution at time t
        Args:
            t (float): current time
        Returns:
            dtype_u: exact solution
        """
        assert t == 0, 'ERROR: u_exact only valid for t=0'

        me = self.dtype_u(self.init)

        me[0] = 0.0  # cL
        me[1] = self.params.alpha * self.params.V_ref  # vC

        return me

    def get_switching_info(self, u, t):
        """
        Provides information about a discrete event for one subinterval.
        Args:
            u (dtype_u): current values
            t (float): current time
        Returns:
            switch_detected (bool): Indicates if a switch is found or not
            m_guess (np.int): Index of where the discrete event would found
            vC_switch (list): Contains function values of switching condition (for interpolation)
        """

        switch_detected = False
        m_guess = -100

        for m in range(len(u)):
            if u[m][1] - self.params.V_ref <= 0:
                switch_detected = True
                m_guess = m - 1
                break

        vC_switch = []
        if switch_detected:
            for m in range(1, len(u)):
                vC_switch.append(u[m][1] - self.params.V_ref)

        return switch_detected, m_guess, vC_switch

    def set_counter(self):
        """
        Counts the number of switches found.
        """

        self.count_switches += 1
