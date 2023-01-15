import numpy as np

from pySDC.core.Errors import ParameterError
from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh


class battery_2condensators(ptype):
    """
    Example implementing the battery drain model using two capacitors as in the description in the PinTSimE
    project
    Attributes:
        A: system matrix, representing the 3 ODEs
        t_switch: time point of the switch
        SV, SC1, SC2: states of switching (important for switch estimator)
    """

    def __init__(self, problem_params, dtype_u=mesh, dtype_f=imex_mesh):
        """
        Initialization routine
        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: mesh data type for solution
            dtype_f: mesh data type for RHS
        """

        problem_params['nvars'] = 3

        # these parameters will be used later, so assert their existence
        essential_keys = ['Vs', 'Rs', 'C1', 'C2', 'R', 'L', 'alpha', 'V_ref']

        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(battery_2condensators, self).__init__(
            init=(problem_params['nvars'], None, np.dtype('float64')),
            dtype_u=dtype_u,
            dtype_f=dtype_f,
            params=problem_params,
        )

        self.A = np.zeros((3, 3))
        self.t_switch = None
        self.SV = 0
        self.SC1 = 1
        self.SC2 = 0

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
            if self.SV == 0 and self.SC1 == 0 and self.SC2 == 1:
                if t >= self.t_switch:
                    f.expl[0] = 0
                else:
                    f.expl[0] = 0
            elif self.SV == 1 and self.SC1 == 0 and self.SC2 == 0:
                if t >= self.t_switch:
                    f.expl[0] = self.params.Vs / self.params.L
                else:
                    f.expl[0] = 0

        else:
            if u[1] > self.params.V_ref[0] and u[2] > self.params.V_ref[1]:
                f.expl[0] = 0
            elif u[1] <= self.params.V_ref[0] and u[2] > self.params.V_ref[1]:
                f.expl[0] = 0
            elif u[1] <= self.params.V_ref[0] and u[2] <= self.params.V_ref[1]:
                f.expl[0] = self.params.Vs / self.params.L

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
        self.A = np.zeros((3, 3))

        if self.t_switch is not None:
            if self.SV == 0 and self.SC1 == 0 and self.SC2 == 1:
                if t >= self.t_switch:
                    self.A[2, 2] = -1 / (self.params.C2 * self.params.R)
                else:
                    self.A[1, 1] = -1 / (self.params.C1 * self.params.R)
            elif self.SV == 1 and self.SC1 == 0 and self.SC2 == 0:
                if t >= self.t_switch:
                    self.A[0, 0] = -(self.params.Rs + self.params.R) / self.params.L
                else:
                    self.A[2, 2] = -1 / (self.params.C2 * self.params.R)

        else:
            if rhs[1] > self.params.V_ref[0] and rhs[2] > self.params.V_ref[1]:
                self.A[1, 1] = -1 / (self.params.C1 * self.params.R)
            elif rhs[1] <= self.params.V_ref[0] and rhs[2] > self.params.V_ref[1]:
                self.A[2, 2] = -1 / (self.params.C2 * self.params.R)
            elif rhs[1] <= self.params.V_ref[0] and rhs[2] <= self.params.V_ref[1]:
                self.A[0, 0] = -(self.params.Rs + self.params.R) / self.params.L

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
        me[1] = self.params.alpha * self.params.V_ref[0]  # vC1
        me[2] = self.params.alpha * self.params.V_ref[1]  # vC2
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
            for k in range(1, self.params.nvars):
                if u[m][k] - self.params.V_ref[k - 1] <= 0:
                    switch_detected = True
                    m_guess = m - 1
                    k_detected = k
                    break

                if k == self.params.nvars and switch_detected and u[m][k] - self.params.V_ref[k - 1] <= 0:
                    msg = 'A discrete event is already found! Multiple switching handling in the same interval is not yet implemented!'
                    raise AssertionError(msg)

        vC_switch = []
        if switch_detected:
            for m in range(1, len(u)):
                vC_switch.append(u[m][k_detected] - self.params.V_ref[k_detected - 1])

        return switch_detected, m_guess, vC_switch

    def flip_switches(self):
        """
        Flips the switches of the circuit to its new state
        """

        if self.SV == 0 and self.SC1 == 1 and self.SC2 == 0:
            self.SV, self.SC1, self.SC2 = 0, 0, 1
        elif self.SV == 0 and self.SC1 == 0 and self.SC2 == 1:
            self.SV, self.SC1, self.SC2 = 1, 0, 0
