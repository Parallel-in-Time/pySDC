import numpy as np

from pySDC.core.Errors import ProblemError
from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh


class battery_n_capacitors(ptype):
    r"""
    Example implementing the battery drain model with :math:`N` capacitors, where :math:`N` is an arbitrary integer greater than zero.
    First, the capacitor :math:`C` serves as a battery and provides energy. When the voltage of the capacitor :math:`u_{C_n}` for
    :math:`n=1,..,N` drops below their reference value :math:`V_{ref,n-1}`, the circuit switches to the next capacitor. If all capacitors
    has dropped below their reference value, the voltage source :math:`V_s` provides further energy. The problem of simulating the
    battery draining has :math:`N + 1` different states. Each of this state can be expressed as a nonhomogeneous linear system of
    ordinary differential equations (ODEs)

    .. math::
        \frac{d u(t)}{dt} = A_k u(t) + f_k (t)

    for :math:`k=1,..,N+1` using an initial condition.

    Parameters
    ----------
    ncapacitors : int
        Number of capacitors :math:`n_{capacitors}` in the circuit.
    Vs : float
        Voltage at the voltage source :math:`V_s`.
    Rs : float
        Resistance of the resistor :math:`R_s` at the voltage source.
    C : np.ndarray
        Capacitances of the capacitors.
    R : float
        Resistance for the load.
    L : float
        Inductance of inductor.
    alpha : float
        Factor greater than zero to describe the storage of the capacitor(s).
    V_ref : np.ndarray
        Array contains the reference values greater than zero for each capacitor to switch to the next energy source.

    Attributes
    ----------
    A: matrix
        Coefficients matrix of the linear system of ordinary differential equations (ODEs).
    switch_A: dict
        Dictionary that contains the coefficients for the coefficient matrix A.
    switch_f: dict
        Dictionary that contains the coefficients of the right-hand side f of the ODE system.
    t_switch: float
        Time point of the discrete event found by switch estimation.
    nswitches: int
        Number of switches found by switch estimation.

    Note
    ----
    The array containing the capacitances :math:`C_n` and the array containing the reference values :math:`V_{ref, n-1}`
    for each capacitor must be equal to the number of capacitors :math:`n_{capacitors}`.
    """

    dtype_u = mesh
    dtype_f = imex_mesh

    def __init__(
        self, ncapacitors=2, Vs=5.0, Rs=0.5, C=None, R=1.0, L=1.0, alpha=1.2, V_ref=None
    ):
        """Initialization routine"""
        n = ncapacitors
        nvars = n + 1

        if C is None:
            C = np.array([1.0, 1.0])

        if V_ref is None:
            V_ref = np.array([1.0, 1.0])

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super().__init__(init=(nvars, None, np.dtype('float64')))
        self._makeAttributeAndRegister(
            'nvars', 'ncapacitors', 'Vs', 'Rs', 'C', 'R', 'L', 'alpha', 'V_ref', localVars=locals(), readOnly=True
        )

        self.A = np.zeros((n + 1, n + 1))
        self.switch_A, self.switch_f = self.get_problem_dict()
        self.t_switch = None
        self.nswitches = 0

    def eval_f(self, u, t):
        r"""
        Routine to evaluate the right-hand side of the problem. Let :math:`v_k:=v_{C_k}` be the voltage of capacitor :math:`C_k` for :math:`k=1,..,N`
        with reference value :math:`V_{ref, k-1}`. No switch estimator is used: For :math:`N = 3` there are :math:`N + 1 = 4` different states of the battery:

        :math:`C_1` supplies energy if:

        .. math::
            v_1 > V_{ref,0}, v_2 > V_{ref,1}, v_3 > V_{ref,2},

        :math:`C_2` supplies energy if:

        .. math::
            v_1 \leq V_{ref,0}, v_2 > V_{ref,1}, v_3 > V_{ref,2},

        :math:`C_3` supplies energy if:

        .. math::
            v_1 \leq V_{ref,0}, v_2 \leq V_{ref,1}, v_3 > V_{ref,2},

        :math:`V_s` supplies energy if:

        .. math::
            v_1 \leq V_{ref,0}, v_2 \leq V_{ref,1}, v_3 \leq V_{ref,2}.

        :math:`max_{index}` is initialized to :math:`-1`. List "switch" contains a True if :math:`u_k \leq V_{ref,k-1}` is satisfied.
            - Is no True there (i.e., :math:`max_{index}=-1`), we are in the first case.
            - :math:`max_{index}=k\geq 0` means we are in the :math:`(k+1)`-th case.
              So, the actual RHS has key :math:`max_{index}`-1 in the dictionary self.switch_f.
        In case of using the switch estimator, we count the number of switches which illustrates in which case of voltage source we are.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution.
        t : float
            Current time of the numerical solution is computed.

        Returns
        -------
        f : dtype_f
            The right-hand side of the problem.
        """

        f = self.dtype_f(self.init, val=0.0)
        f.impl[:] = self.A.dot(u)

        if self.t_switch is not None:
            f.expl[:] = self.switch_f[self.nswitches]

        else:
            # proof all switching conditions and find largest index where it drops below V_ref
            switch = [True if u[k] <= self.V_ref[k - 1] else False for k in range(1, len(u))]
            max_index = max([k if switch[k] == True else -1 for k in range(len(switch))])

            if max_index == -1:
                f.expl[:] = self.switch_f[0]

            else:
                f.expl[:] = self.switch_f[max_index + 1]

        return f

    def solve_system(self, rhs, factor, u0, t):
        r"""
        Simple linear solver for :math:`(I-factor\cdot A)\vec{u}=\vec{rhs}`.

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side for the linear system.
        factor : float
            Abbrev. for the local stepsize (or any other factor required).
        u0 : dtype_u
            Initial guess for the iterative solver.
        t : float
            Current time (e.g. for time-dependent BCs).

        Returns
        -------
        me : dtype_u
            The solution as mesh.
        """

        if self.t_switch is not None:
            self.A = self.switch_A[self.nswitches]

        else:
            # proof all switching conditions and find largest index where it drops below V_ref
            switch = [True if rhs[k] <= self.V_ref[k - 1] else False for k in range(1, len(rhs))]
            max_index = max([k if switch[k] == True else -1 for k in range(len(switch))])
            if max_index == -1:
                self.A = self.switch_A[0]

            else:
                self.A = self.switch_A[max_index + 1]

        me = self.dtype_u(self.init)
        me[:] = np.linalg.solve(np.eye(self.nvars) - factor * self.A, rhs)
        return me

    def u_exact(self, t):
        """
        Routine to compute the exact solution at time t.

        Parameters
        ----------
        t : float
            Time of the exact solution.

        Returns
        -------
        me : dtype_u
            The exact solution.
        """
        assert t == 0, 'ERROR: u_exact only valid for t=0'

        me = self.dtype_u(self.init)

        me[0] = 0.0  # cL
        me[1:] = self.alpha * self.V_ref  # vC's
        return me

    def get_switching_info(self, u, t):
        """
        Provides information about a discrete event for one subinterval.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution.
        t : float
            Current time of the numerical solution is computed.

        Returns
        -------
        switch_detected : bool
            Indicates if a switch is found or not.
        m_guess : int
            Index of collocation node inside one subinterval of where the discrete event was found.
        vC_switch : list
            Contains function values of switching condition (for interpolation).
        """

        switch_detected = False
        m_guess = -100
        break_flag = False

        for m in range(1, len(u)):
            for k in range(1, self.nvars):
                if u[m][k] - self.V_ref[k - 1] <= 0:
                    switch_detected = True
                    m_guess = m - 1
                    k_detected = k
                    break_flag = True
                    break

            if break_flag:
                break

        vC_switch = [u[m][k_detected] - self.V_ref[k_detected - 1] for m in range(1, len(u))] if switch_detected else []

        return switch_detected, m_guess, vC_switch

    def count_switches(self):
        """
        Counts the number of switches. This function is called when a switch is found inside the range of tolerance
        (in pySDC/projects/PinTSimE/switch_estimator.py)
        """

        self.nswitches += 1

    def get_problem_dict(self):
        """
        Helper to create dictionaries for both the coefficent matrix of the ODE system and the nonhomogeneous part.
        """

        n = self.ncapacitors
        v = np.zeros(n + 1)
        v[0] = 1

        A, f = dict(), dict()
        A = {k: np.diag(-1 / (self.C[k] * self.R) * np.roll(v, k + 1)) for k in range(n)}
        A.update({n: np.diag(-(self.Rs + self.R) / self.L * v)})
        f = {k: np.zeros(n + 1) for k in range(n)}
        f.update({n: self.Vs / self.L * v})
        return A, f


class battery(battery_n_capacitors):
    r"""
    Example implementing the battery drain model with :math:`N=1` capacitor, inherits from battery_n_capacitors. The ODE system
    of this model is given by the following equations. If :math:`v_1 > V_{ref, 0}:`

    .. math::
        \frac{d i_L (t)}{dt} = 0,

    .. math::
        \frac{d v_1 (t)}{dt} = -\frac{1}{CR}v_1 (t),

    where :math:`i_L` denotes the function of the current over time :math:`t`.
    If :math:`v_1 \leq V_{ref, 0}:`

    .. math::
        \frac{d i_L(t)}{dt} = -\frac{R_s + R}{L}i_L (t) + \frac{1}{L} V_s,

    .. math::
        \frac{d v_1(t)}{dt} = 0.

    Note
    ----
    This class has the same attributes as the class it inherits from.
    """

    dtype_f = imex_mesh

    def eval_f(self, u, t):
        """
        Routine to evaluate the right-hand side of the problem.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution.
        t : float
            Current time of the numerical solution is computed.

        Returns
        -------
        f : dtype_f
            The right-hand side of the problem.
        """

        f = self.dtype_f(self.init, val=0.0)
        f.impl[:] = self.A.dot(u)

        t_switch = np.inf if self.t_switch is None else self.t_switch

        if u[1] <= self.V_ref[0] or t >= t_switch:
            f.expl[0] = self.Vs / self.L

        else:
            f.expl[0] = 0

        return f

    def solve_system(self, rhs, factor, u0, t):
        r"""
        Simple linear solver for :math:`(I-factor\cdot A)\vec{u}=\vec{rhs}`.

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side for the linear system.
        factor : float
            Abbrev. for the local stepsize (or any other factor required).
        u0 : dtype_u
            Initial guess for the iterative solver.
        t : float
            Current time (e.g. for time-dependent BCs).

        Returns
        -------
        me : dtype_u
            The solution as mesh.
        """
        self.A = np.zeros((2, 2))

        t_switch = np.inf if self.t_switch is None else self.t_switch

        if rhs[1] <= self.V_ref[0] or t >= t_switch:
            self.A[0, 0] = -(self.Rs + self.R) / self.L

        else:
            self.A[1, 1] = -1 / (self.C[0] * self.R)

        me = self.dtype_u(self.init)
        me[:] = np.linalg.solve(np.eye(self.nvars) - factor * self.A, rhs)
        return me

    def u_exact(self, t):
        """
        Routine to compute the exact solution at time t.

        Parameters
        ----------
        t : float
            Time of the exact solution.

        Returns
        -------
        me : dtype_u
            The exact solution.
        """
        assert t == 0, 'ERROR: u_exact only valid for t=0'

        me = self.dtype_u(self.init)

        me[0] = 0.0  # cL
        me[1] = self.alpha * self.V_ref[0]  # vC

        return me


class battery_implicit(battery):
    r"""
    Example implementing the battery drain model as above. The method solve_system uses a fully-implicit computation.

    Parameters
    ----------
    ncapacitors : int
        Number of capacitors in the circuit.
    Vs : float
        Voltage at the voltage source :math:`V_s`.
    Rs : float
        Resistance of the resistor :math:`R_s` at the voltage source.
    C : np.ndarray
        Capacitances of the capacitors. Length of array must equal to number of capacitors.
    R : float
        Resistance for the load.
    L : float
        Inductance of inductor.
    alpha : float
        Factor greater than zero to describe the storage of the capacitor(s).
    V_ref : float
        Reference value greater than zero for the battery to switch to the voltage source.
    newton_maxiter : int
        Number of maximum iterations for the Newton solver.
    newton_tol : float
        Tolerance for determination of the Newton solver.

    Attributes
    ----------
    newton_itercount: int
        Counts the number of Newton iterations.
    newton_ncalls: int
        Counts the number of how often Newton is called in the simulation of the problem.
    """
    dtype_f = mesh

    def __init__(
        self,
        ncapacitors=1,
        Vs=5.0,
        Rs=0.5,
        C=None,
        R=1.0,
        L=1.0,
        alpha=1.2,
        V_ref=None,
        newton_maxiter=200,
        newton_tol=1e-8,
    ):

        if C is None:
            C = np.array([1.0])

        if V_ref is None:
            V_ref = np.array([1.0])

        super().__init__(ncapacitors, Vs, Rs, C, R, L, alpha, V_ref)
        self._makeAttributeAndRegister('newton_maxiter', 'newton_tol', localVars=locals(), readOnly=True)

        self.newton_itercount = 0
        self.newton_ncalls = 0

    def eval_f(self, u, t):
        """
        Routine to evaluate the right-hand side of the problem.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution.
        t : float
            Current time of the numerical solution is computed.

        Returns
        -------
        f : dtype_f
            The right-hand side of the problem.
        """

        f = self.dtype_f(self.init, val=0.0)
        non_f = np.zeros(2)

        t_switch = np.inf if self.t_switch is None else self.t_switch

        if u[1] <= self.V_ref[0] or t >= t_switch:
            self.A[0, 0] = -(self.Rs + self.R) / self.L
            non_f[0] = self.Vs

        else:
            self.A[1, 1] = -1 / (self.C[0] * self.R)
            non_f[0] = 0

        f[:] = self.A.dot(u) + non_f
        return f

    def solve_system(self, rhs, factor, u0, t):
        """
        Simple Newton solver.

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side for the linear system.
        factor : float
            Abbrev. for the local stepsize (or any other factor required).
        u0 : dtype_u
            Initial guess for the iterative solver
        t : float
            Current time (e.g. for time-dependent BCs).

        Returns
        -------
        me : dtype_u
            The solution as mesh.
        """

        u = self.dtype_u(u0)
        non_f = np.zeros(2)
        self.A = np.zeros((2, 2))

        t_switch = np.inf if self.t_switch is None else self.t_switch

        if rhs[1] <= self.V_ref[0] or t >= t_switch:
            self.A[0, 0] = -(self.Rs + self.R) / self.L
            non_f[0] = self.Vs

        else:
            self.A[1, 1] = -1 / (self.C[0] * self.R)
            non_f[0] = 0

        # start newton iteration
        n = 0
        res = 99
        while n < self.newton_maxiter:
            # form function g with g(u) = 0
            g = u - rhs - factor * (self.A.dot(u) + non_f)

            # if g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)

            if res < self.newton_tol:
                break

            # assemble dg
            dg = np.eye(self.nvars) - factor * self.A

            # newton update: u1 = u0 - g/dg
            u -= np.linalg.solve(dg, g)

            # increase iteration count
            n += 1

        if np.isnan(res) and self.stop_at_nan:
            raise ProblemError('Newton got nan after %i iterations, aborting...' % n)
        elif np.isnan(res):
            self.logger.warning('Newton got nan after %i iterations...' % n)

        if n == self.newton_maxiter:
            self.logger.warning('Newton did not converge after %i iterations, error is %s' % (n, res))

        self.newton_ncalls += 1
        self.newton_itercount += n

        me = self.dtype_u(self.init)
        me[:] = u[:]

        return me
