import numpy as np

from pySDC.core.Errors import ParameterError
from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh


class state_space_inverter(ptype):
    """
    Example implementing the state space inverter model as in the description in the PinTSimE project
    Attributes:
        A: system matrix, representing the 8 ODEs
    """

    def __init__(self, problem_params, dtype_u=mesh, dtype_f=imex_mesh):
        """
        Initialization routine
        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: mesh data type for solution
            dtype_f: mesh data type for RHS
        """

        problem_params['nvars'] = 8

        # these parameters will be used later, so assert their existence
        essential_keys = ['fsw', 'CDC1', 'CDC2', 'C1', 'C2', 'C3', 'L1',
                          'L2', 'L3', 'Rs1', 'Rs2', 'Rl1', 'Rl2', 'Rl3', 'V1', 'V2']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(state_space_inverter, self).__init__(init=(problem_params['nvars'], None, np.dtype('float64')),
                                                   dtype_u=dtype_u, dtype_f=dtype_f, params=problem_params)

        self.A = np.zeros((8, 8))

    @staticmethod
    def get_PWM(t):
        """
            Computes the PWM signal as control signal for switching
            Args:
                t (float): time point at which PWM is computed
            Returns:
                The duty cycle for all three phases
        """

        f = 50
        duty1 = 0.5 + 0.5 * np.sin(2 * np.pi * f * t)
        duty2 = 0.5 + 0.5 * np.sin(2 * np.pi * f * t + 2 * np.pi / 3)
        duty3 = 0.5 + 0.5 * np.sin(2 * np.pi * f * t - 2 * np.pi / 3)

        return duty1, duty2, duty3

    @staticmethod
    def get_A(fsw, CDC1, CDC2, C1, C2, C3, L1, L2, L3, Rs1, Rs2, Rl1, Rl2, Rl3, V1, V2, t, duty1, duty2, duty3):
        """
            Helper function to compute the coefficient matrix A
            Args:
                fsw (int): switching frequency
                t (float): time point to evaluate
                duty1 (float): Duty cycle 1 of the PWM signal (results from sinus signal)
                duty2 (float): Duty cycle 2 of the PWM signal (results from sinus signal + 2*pi/3)
                duty3 (float): Duty cycle 3 of the PWM signal (results from sinus signal - 2*pi/3)
        """

        Tsw = 1 / fsw
        A = np.zeros((8, 8))

        if 0 <= ((t / Tsw) % 1) <= duty1 and 0 <= ((t / Tsw) % 1) <= duty2 and 0 <= ((t / Tsw) % 1) <= duty3:
            # state 1
            A[0, 0] = -1 / (CDC1 * Rs1)
            A[0, 2] = -1 / CDC1

            A[1, 1] = -1 / (CDC2 * Rs2)

            A[2, 0] = 1 / L1
            A[2, 5] = -1 / L1

            A[3, 0] = 1 / L2
            A[3, 5] = -2 / L2
            A[3, 6] = 1 / L2

            A[4, 0] = 1 / L3
            A[4, 5] = -2 / L3
            A[4, 7] = 1 / L3

            A[5, 2] = 1 / C1
            A[5, 5] = -1 / (C1 * Rl1)

            A[6, 3] = 1 / C2
            A[6, 6] = -1 / (C2 * Rl2)

            A[7, 4] = 1 / C3
            A[7, 7] = -1 / (C3 * Rl3)

        elif 0 <= ((t / Tsw) % 1) <= duty1 and 0 <= ((t / Tsw) % 1) <= duty2:
            # state 2
            A[0, 0] = -1 / (CDC1 * Rs1)

            A[1, 1] = -1 / (CDC2 * Rs2)
            A[1, 4] = -1 / CDC2

            A[2, 0] = 1 / L1
            A[2, 5] = -1 / L1

            A[3, 0] = 1 / L2
            A[3, 7] = 1 / L2

            A[4, 1] = 1 / L3
            A[4, 5] = -1 / L3

            A[5, 2] = 1 / C1
            A[5, 5] = -1 / (C1 * Rl1)

            A[6, 3] = 1 / C2
            A[6, 6] = -1 / (C2 * Rl2)

            A[7, 4] = 1 / C3
            A[7, 7] = -1 / (C3 * Rl3)

        elif 0 <= ((t / Tsw) % 1) <= duty2 and 0 <= ((t / Tsw) % 1) <= duty3:
            # state 5
            A[0, 0] = -1 / (CDC1 * Rs1)

            A[1, 1] = -1 / (CDC2 * Rs2)
            A[1, 2] = -1 / CDC2

            A[2, 0] = 1 / L1
            A[2, 5] = -1 / L1

            A[3, 0] = 1 / L2
            A[3, 6] = -1 / L2

            A[4, 0] = 1 / L3
            A[4, 7] = -1 / L3

            A[5, 2] = 1 / C1
            A[5, 5] = -1 / (C1 * Rl1)

            A[6, 3] = 1 / C2
            A[6, 6] = -1 / (C2 * Rl2)

            A[7, 4] = 1 / C3
            A[7, 7] = -1 / (C3 * Rl3)

        elif 0 <= ((t / Tsw) % 1) <= duty1 and 0 <= ((t / Tsw) % 1) <= duty3:
            # state 3
            A[0, 0] = -1 / (CDC1 * Rs1)

            A[1, 1] = -1 / (CDC2 * Rs2)
            A[1, 4] = -1 / CDC2

            A[2, 0] = 1 / L1
            A[2, 5] = -1 / L1

            A[3, 0] = 1 / L2
            A[3, 6] = -1 / L2
            A[3, 7] = -1 / L2

            A[4, 0] = 1 / L3
            A[4, 7] = -2 / L3

            A[5, 2] = 1 / C1
            A[5, 5] = -1 / (C1 * Rl1)

            A[6, 3] = 1 / C2
            A[6, 6] = -1 / (C2 * Rl2)

            A[7, 4] = 1 / C3
            A[7, 7] = -1 / (C3 * Rl3)

        elif 0 <= ((t / Tsw) % 1) <= duty1:
            # state 7
            A[0, 0] = -1 / (CDC1 * Rs1)
            A[0, 4] = -1 / CDC1

            A[1, 1] = -1 / (CDC2 * Rs2)

            A[2, 0] = 1 / L1
            A[2, 5] = -1 / L1

            A[3, 1] = 1 / L2
            A[3, 6] = -1 / L2

            A[4, 0] = 1 / L3
            A[4, 7] = -1 / L3

            A[5, 2] = 1 / C1
            A[5, 5] = -1 / (C1 * Rl1)

            A[6, 3] = 1 / C2
            A[6, 6] = -1 / (C2 * Rl2)

            A[7, 4] = 1 / C3
            A[7, 7] = -1 / (C3 * Rl3)

        elif 0 <= ((t / Tsw) % 1) <= duty2:
            # state 6
            A[0, 0] = -1 / (CDC1 * Rs1)
            A[0, 3] = -1 / CDC1

            A[1, 1] = -1 / (CDC2 * Rs2)

            A[2, 0] = 1 / L1
            A[2, 5] = -1 / L1

            A[3, 0] = 1 / L2
            A[3, 6] = -1 / L2

            A[4, 1] = 1 / L3
            A[4, 7] = -1 / L3

            A[5, 2] = 1 / C1
            A[5, 5] = -1 / (C1 * Rl1)

            A[6, 3] = 1 / C2
            A[6, 6] = -1 / (C2 * Rl2)

            A[7, 4] = 1 / C3
            A[7, 7] = -1 / (C3 * Rl3)

        elif 0 <= ((t / Tsw) % 1) <= duty3:
            # state 4
            A[0, 0] = -1 / (CDC1 * Rs1)
            A[0, 2] = -1 / CDC1

            A[1, 1] = -1 / (CDC2 * Rs2)

            A[2, 0] = 1 / L1
            A[2, 5] = -1 / L1

            A[3, 1] = 1 / L2
            A[3, 7] = -1 / L2

            A[4, 1] = 1 / L3
            A[4, 6] = 1 / L3
            A[4, 7] = -2 / L3

            A[5, 2] = 1 / C1
            A[5, 5] = -1 / (C1 * Rl1)

            A[6, 3] = 1 / C2
            A[6, 6] = -1 / (C2 * Rl2)

            A[7, 4] = 1 / C3
            A[7, 7] = -1 / (C3 * Rl3)

        else:
            # state 8
            A[0, 0] = -1 / (CDC1 * Rs1)

            A[1, 1] = -1 / (CDC2 * Rs2)
            A[1, 4] = -1 / CDC2

            A[2, 1] = 1 / L1
            A[2, 5] = -1 / L1

            A[3, 1] = 1 / L2
            A[3, 6] = -1 / L2

            A[4, 1] = 1 / L3
            A[4, 7] = -1 / L3

            A[5, 2] = 1 / C1
            A[5, 5] = -1 / (C1 * Rl1)

            A[6, 3] = 1 / C2
            A[6, 6] = -1 / (C2 * Rl2)

            A[7, 4] = 1 / C3
            A[7, 7] = -1 / (C3 * Rl3)

        return A

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

        f.expl[0] = self.params.V1 / (self.params.CDC1 * self.params.Rs1)
        f.expl[1] = -self.params.V2 / (self.params.CDC2 * self.params.Rs2)

        return f

    def solve_system(self, rhs, factor, u0, t):
        """
        Simple linear solver for (I-factor*A)u = rhs. For the state-space inverter there are 8 different states:
            1. state -> S_A1 = 1, S_A2 = 0, S_B1 = 1, S_B2 = 0, S_C1 = 1, S_C1 = 0
            2. state -> S_A1 = 0, S_A2 = 1, S_B1 = 1, S_B2 = 0, S_C1 = 1, S_C1 = 0
            3. state -> S_A1 = 1, S_A2 = 0, S_B1 = 0, S_B2 = 1, S_C1 = 1, S_C1 = 0
            4. state -> S_A1 = 0, S_A2 = 1, S_B1 = 0, S_B2 = 1, S_C1 = 1, S_C1 = 0
            5. state -> S_A1 = 1, S_A2 = 0, S_B1 = 1, S_B2 = 0, S_C1 = 0, S_C1 = 1
            6. state -> S_A1 = 0, S_A2 = 1, S_B1 = 1, S_B2 = 0, S_C1 = 0, S_C1 = 1
            7. state -> S_A1 = 1, S_A2 = 0, S_B1 = 0, S_B2 = 1, S_C1 = 0, S_C1 = 1
            8. state -> S_A1 = 0, S_A2 = 1, S_B1 = 0, S_B2 = 1, S_C1 = 0, S_C1 = 1
        Args:
            rhs (dtype_f): right-hand side for the linear system
            factor (float): abbrev. for the local stepsize (or any other factor required)
            u0 (dtype_u): initial guess for the iterative solver
            t (float): current time (e.g. for time-dependent BCs)
        Returns:
            dtype_u: solution as mesh
        """

        self.A = np.zeros((8, 8))

        duty1, duty2, duty3 = self.get_PWM(t)
        self.A = self.get_A(self.params.fsw, self.params.CDC1, self.params.CDC2, self.params.C1,
                            self.params.C2, self.params.C3, self.params.L1, self.params.L2, self.params.L3,
                            self.params.Rs1, self.params.Rs2, self.params.Rl1, self.params.Rl2, self.params.Rl3,
                            self.params.V1, self.params.V2, t, duty1, duty2, duty3)

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

        me = self.dtype_u(self.init)

        me[0] = 0.0  # vCDC1
        me[1] = 0.0  # vCDC2
        me[2] = 0.0  # iL1
        me[3] = 0.0  # iL2
        me[4] = 0.0  # iL3
        me[5] = 0.0  # vC1
        me[6] = 0.0  # vC2
        me[7] = 0.0  # vC3
        return me
