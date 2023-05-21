import numpy as np
import warnings
from scipy.interpolate import interp1d

from pySDC.projects.DAE.misc.ProblemDAE import ptype_dae
from pySDC.implementations.datatype_classes.mesh import mesh


class synchronous_machine_infinite_bus(ptype_dae):
    r"""
    Synchronous machine model from Kundur (equiv. circuits fig. 3.18 in [1]_) attached to infinite bus. The machine can be
    represented as two different circuits at the direct-axis and the quadrature-axis. Detailed information can be found in
    [1]_. The system of differential-algebraic equations (DAEs) consists of the equations for

        - the stator voltage equations

        .. math::
            \frac{d \Psi_d (t)}{dt} = \omega_b (v_d + R_a i_d (t) + \omega_r \Psi_q (t)),

        .. math::
            \frac{d \Psi_q (t)}{dt} = \omega_b (v_q + R_a i_q (t) - \omega_r \Psi_d (t)),

        .. math::
            \frac{d \Psi_0 (t)}{dt} = \omega_b (v_0 + R_a i_0 (t)),

        - the rotor voltage equations

        .. math::
            \frac{d \Psi_F (t)}{dt} = \omega_b (v_F - R_F i_F (t)),

        .. math::
            \frac{d \Psi_D (t)}{dt} = -\omega_b (R_D i_D (t)),

        .. math::
            \frac{d \Psi_{Q1} (t)}{dt} = -\omega_b (R_{Q1} i_{Q1} (t)),

        .. math::
            \frac{d \Psi_{Q2} (t)}{dt} = -\omega_b (R_{Q2} i_{Q2} (t)),

        - the stator flux linkage equations

        .. math::
            \Psi_d (t) = L_d i_d (t)  + L_{md} i_F (t) + L_{md} i_D (t),

        .. math::
            \Psi_q (t) = L_q i_q (t) + L_{mq} i_{Q1} (t) + L_{mq} i_{Q2} (t),

        .. math::
            \Psi_0 (t) = L_0 i_0 (t)

        - the rotor flux linkage equations

        .. math::
            \Psi_F = L_F i_F (t) + L_D i_D + L_{md} i_d (t),

        .. math::
            \Psi_D = L_F i_F (t) + L_D i_D + L_{md} i_d (t),

        .. math::
            \Psi_{Q1} = L_{Q1} i_{Q1} (t) + L_{mq} i_{Q2} + L_{mq} i_q (t),

        .. math::
            \Psi_{Q2} = L_{mq} i_{Q1} (t) + L_{Q2} i_{Q2} + L_{mq} i_q (t),

        - the swing equations

        .. math::
            \frac{d \delta (t)}{dt} = \omega_b (\omega_r (t) - 1),

        .. math::
            \frac{d \omega_r (t)}{dt} = \frac{1}{2 H}(T_m - T_e - K_D \omega_b (\omega_r (t) - 1)).

    The voltages :math:`v_d`, :math:`v_q` can be updated via the following procedure. The stator's currents are mapped
    to the comlex-valued external reference frame current :math:`I` with

    .. math::
        \Re(I) = i_d (t) \sin(\delta (t)) + i_q (t) \cos(\delta (t)),

    .. math::
        \Im(I) = -i_d (t) \cos(\delta (t)) + i_q (t) \sin(\delta (t)).

    The voltage V across the stator terminals can then be computed as complex-value via

    .. math::
        V_{comp} = E_B + Z_{line} (\Re(I) + i \Im(I))

    with impedance :math:`Z_{line}\in\mathbb{C}`. Then, :math:`v_d`, :math:`v_q` can be computed via the network equations

    .. math::
        v_d = \Re(V_{comp}) \sin(\delta (t)) - \Im(V_{comp}) \cos(\delta (t)),

    .. math::
       v_q = \Re(V_{comp}) \cos(\delta (t)) + \Im(V_{comp}) \sin(\delta (t)),

    which describes the connection between the machine and the infinite bus.

    Parameters
    ----------
    nvars : int
        Number of unknowns of the system of DAEs.
    newton_tol : float
        Tolerance for Newton solver.

    Attributes
    ----------
    L_d: float
        Inductance of inductor :math:'L_d', see [1]_.
    L_q: float
        Inductance of inductor :math:'L_q', see [1]_.
    L_F: float
        Inductance of inductor :math:'L_F', see [1]_.
    L_D: float
        Inductance of inductor :math:'L_D', see [1]_.
    L_Q1: float
        Inductance of inductor :math:'L_{Q1}', see [1]_.
    L_Q2: float
        Inductance of inductor :math:'L_{Q2}', see [1]_.
    L_md: float
        Inductance of inductor :math:'L_{md}', see [1]_.
    L_mq: float
        Inductance of inductor :math:'L_{mq}', see [1]_.
    R_s: float
        Resistance of resistor :math:`R_s`, see [1]_.
    R_F: float
        Resistance of resistor :math:`R_F`, see [1]_.
    R_D: float
        Resistance of resistor :math:`R_D`, see [1]_.
    R_Q1: float
        Resistance of resistor :math:`R_{Q1}`, see [1]_.
    R_Q2: float
        Resistance of resistor :math:`R_{Q2}`, see [1]_.
    omega_b: float
        Base frequency of the rotor in mechanical :math:`rad/s`.
    H_: float
        Defines the per unit inertia constant.
    K_D: float
        Factor that accounts for damping losses.
    Z_line: complex
        Impedance of the transmission line that connects the infinite bus to the generator.
    E_B: float
        Voltage of infinite bus.
    v_F: float
        Voltage at the field winding.
    T_m: float
        Defines the mechanical torque applied to the rotor shaft.

    References
    ----------
    .. [1] P. Kundur, N. J. Balu, M. G. Lauby. Power system stability and control. The EPRI power system series (1994).
    """

    def __init__(self, nvars, newton_tol):
        super(synchronous_machine_infinite_bus, self).__init__(nvars, newton_tol)
        # load reference solution
        # data file must be generated and stored under misc/data and self.t_end = t[-1]
        # data = np.load(r'pySDC/projects/DAE/misc/data/synch_gen.npy')
        # x = data[:, 0]
        # y = data[:, 1:]
        # self.u_ref = interp1d(x, y, kind='cubic', axis=0, fill_value='extrapolate')
        self.t_end = 0.0

        self.L_d = 1.8099
        self.L_q = 1.76
        self.L_F = 1.8247
        self.L_D = 1.8312
        self.L_Q1 = 2.3352
        self.L_Q2 = 1.735
        self.L_md = 1.6599
        self.L_mq = 1.61
        self.R_s = 0.003
        self.R_F = 0.0006
        self.R_D = 0.0284
        self.R_Q1 = 0.0062
        self.R_Q2 = 0.0237
        self.omega_b = 376.9911184307752
        self.H_ = 3.525
        self.K_D = 0.0
        # Line impedance
        self.Z_line = -0.2688022164909709 - 0.15007173591230372j
        # Infinite bus voltage
        self.E_B = 0.7
        # Rotor (field) operating voltages
        # These are modelled as constants. Intuition: permanent magnet as rotor
        self.v_F = 8.736809687330562e-4
        self.T_m = 0.854

    def eval_f(self, u, du, t):
        r"""
        Routine to evaluate the implicit representation of the problem, i.e., :math:`F(u, u', t)`.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution at time t.
        du : dtype_u
            Current values of the derivative of the numerical solution at time t.
        t : float
            Current time of the numerical solution.

        Returns
        -------
        f : dtype_f
            Current value of the right-hand side of f (which includes 14 components).
        """

        # simulate torque change at t = 0.05
        if t >= 0.05:
            self.T_m = 0.354

        f = self.dtype_f(self.init)

        # u = [psi_d, psi_q, psi_F, psi_D, psi_Q1, psi_Q2,
        #       i_d, i_q, i_F, i_D, i_Q1, i_Q2
        #       omega_m,
        #       v_d, v_q,
        #       iz_d, iz_q, il_d, il_q, vl_d, vl_q]

        # extract variables for readability
        # algebraic components
        psi_d, psi_q, psi_F, psi_D, psi_Q1, psi_Q2 = u[0], u[1], u[2], u[3], u[4], u[5]
        i_d, i_q, i_F, i_D, i_Q1, i_Q2 = u[6], u[7], u[8], u[9], u[10], u[11]
        delta_r = u[12]
        omega_m = u[13]

        # differential components
        # these result directly from the voltage equations, introduced e.g. pg. 145 Krause
        dpsi_d, dpsi_q, dpsi_F, dpsi_D, dpsi_Q1, dpsi_Q2 = du[0], du[1], du[2], du[3], du[4], du[5]
        ddelta_r = du[12]
        domega_m = du[13]
        # Network current
        I_Re = i_d * np.sin(delta_r) + i_q * np.cos(delta_r)
        I_Im = -i_d * np.cos(delta_r) + i_q * np.sin(delta_r)
        # Machine terminal voltages in network coordinates
        # Need to transform like this to subtract infinite bus voltage
        V_comp = self.E_B - self.Z_line * (-1) * (I_Re + 1j * I_Im)
        # Terminal voltages in dq0 coordinates
        v_d = np.real(V_comp) * np.sin(delta_r) - np.imag(V_comp) * np.cos(delta_r)
        v_q = np.real(V_comp) * np.cos(delta_r) + np.imag(V_comp) * np.sin(delta_r)

        # algebraic variables are i_d, i_q, i_F, i_D, i_Q1, i_Q2, il_d, il_q
        f[:] = (
            # differential generator
            -dpsi_d + self.omega_b * (v_d - self.R_s * i_d + omega_m * psi_q),
            -dpsi_q + self.omega_b * (v_q - self.R_s * i_q - omega_m * psi_d),
            -dpsi_F + self.omega_b * (self.v_F - self.R_F * i_F),
            -dpsi_D - self.omega_b * self.R_D * i_D,
            -dpsi_Q1 - self.omega_b * self.R_Q1 * i_Q1,
            -dpsi_Q2 - self.omega_b * self.R_Q2 * i_Q2,
            -ddelta_r + self.omega_b * (omega_m - 1),
            -domega_m
            + 1 / (2 * self.H_) * (self.T_m - (psi_q * i_d - psi_d * i_q) - self.K_D * self.omega_b * (omega_m - 1)),
            # algebraic generator
            -psi_d + self.L_d * i_d + self.L_md * i_F + self.L_md * i_D,
            -psi_q + self.L_q * i_q + self.L_mq * i_Q1 + self.L_mq * i_Q2,
            -psi_F + self.L_md * i_d + self.L_F * i_F + self.L_md * i_D,
            -psi_D + self.L_md * i_d + self.L_md * i_F + self.L_D * i_D,
            -psi_Q1 + self.L_mq * i_q + self.L_Q1 * i_Q1 + self.L_mq * i_Q2,
            -psi_Q2 + self.L_mq * i_q + self.L_mq * i_Q1 + self.L_Q2 * i_Q2,
        )
        return f

    def u_exact(self, t):
        """
        Approximation of the exact solution generated by spline interpolation of an extremely accurate numerical reference solution.

        Parameters
        ----------
        t : float
            The time of the reference solution.

        Returns
        -------
        me : dtype_u
            The reference solution as mesh object. It contains fixed initial conditions at initial time (which includes
            14 components).
        """
        me = self.dtype_u(self.init)

        if t == 0:
            psi_d = 0.7770802016688648
            psi_q = -0.6337183129426077
            psi_F = 1.152966888216155
            psi_D = 0.9129958488040036
            psi_Q1 = -0.5797082294536264
            psi_Q2 = -0.579708229453273
            i_d = -0.9061043142342473
            i_q = -0.36006722326230495
            i_F = 1.45613494788927
            i_D = 0.0
            i_Q1 = 0.0
            i_Q2 = 0.0

            delta_r = 39.1 * np.pi / 180
            omega_0 = 2 * np.pi * 60
            omega_b = 2 * np.pi * 60
            omega_m = omega_0 / omega_b  # = omega_r since pf = 2 i.e. two pole machine

            me[:] = (psi_d, psi_q, psi_F, psi_D, psi_Q1, psi_Q2, i_d, i_q, i_F, i_D, i_Q1, i_Q2, delta_r, omega_m)
        elif t < self.t_end:
            me[:] = self.u_ref(t)
        else:
            warnings.warn("Requested time exceeds domain of the reference solution. Returning zero.")
            me[:] = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        return me


# class synchronous_machine_pi_line(ptype_dae):
#     """
#     Synchronous machine model from Kundur (equiv. circuits fig. 3.18)
#     attached to pi line with resistive load

#     This model does not work yet but is included as a starting point for developing similar models and in the hope that somebody will figure out why it does not work
#     """

#     def __init__(self, problem_params, dtype_u=mesh, dtype_f=mesh):
#         super(synchronous_machine_pi_line, self).__init__(problem_params, dtype_u, dtype_f)
#         # load reference solution
#         # data file must be generated and stored under misc/data and self.t_end = t[-1]
#         # data = np.load(r'pySDC/projects/DAE/misc/data/synch_gen.npy')
#         # x = data[:, 0]
#         # y = data[:, 1:]
#         # self.u_ref = interp1d(x, y, kind='cubic', axis=0, fill_value='extrapolate')
#         self.t_end = 0.0

#         self.L_d = 1.8099
#         self.L_q = 1.76
#         self.L_F = 1.8247
#         self.L_D = 1.8312
#         self.L_Q1 = 2.3352
#         self.L_Q2 = 1.735
#         self.L_md = 1.6599
#         self.L_mq = 1.61
#         self.R_s = 0.003
#         self.R_F = 0.0006
#         self.R_D = 0.0284
#         self.R_Q1 = 0.0062
#         self.R_Q2 = 0.0237
#         self.omega_b = 376.9911184307752
#         self.H_ = 3.525
#         self.K_D = 0.0
#         # pi line
#         self.C_pi = 0.000002
#         self.R_pi = 0.02
#         self.L_pi = 0.00003
#         # load
#         self.R_L = 0.75
#         self.v_F = 8.736809687330562e-4
#         self.v_D = 0
#         self.v_Q1 = 0
#         self.v_Q2 = 0
#         self.T_m = 0.854

#     def eval_f(self, u, du, t):
#         """
#         Routine to evaluate the implicit representation of the problem i.e. F(u', u, t)
#         Args:
#             u (dtype_u): the current values. This parameter has been "hijacked" to contain [u', u] in this case to enable evaluation of the implicit representation
#             t (float): current time
#         Returns:
#             Current value of F(), 21 components
#         """

#         # simulate torque change at t = 0.05
#         if t >= 0.05:
#             self.T_m = 0.354

#         f = self.dtype_f(self.init)

#         # u = [psi_d, psi_q, psi_F, psi_D, psi_Q1, psi_Q2,
#         #       i_d, i_q, i_F, i_D, i_Q1, i_Q2
#         #       omega_m,
#         #       v_d, v_q,
#         #       iz_d, iz_q, il_d, il_q, vl_d, vl_q]

#         # extract variables for readability
#         # algebraic components
#         psi_d, psi_q, psi_F, psi_D, psi_Q1, psi_Q2 = u[0], u[1], u[2], u[3], u[4], u[5]
#         i_d, i_q, i_F, i_D, i_Q1, i_Q2 = u[6], u[7], u[8], u[9], u[10], u[11]
#         # delta_r = u[12]
#         omega_m = u[12]
#         v_d, v_q = u[13], u[14]
#         iz_d, iz_q, il_d, il_q, vl_d, vl_q = u[15], u[16], u[17], u[18], u[19], u[20]

#         # differential components
#         # these result directly from the voltage equations, introduced e.g. pg. 145 Krause
#         dpsi_d, dpsi_q, dpsi_F, dpsi_D, dpsi_Q1, dpsi_Q2 = du[0], du[1], du[2], du[3], du[4], du[5]
#         # ddelta_r = du[12]
#         domega_m = du[12]
#         dv_d, dv_q = du[13], du[14]
#         diz_d, diz_q, dvl_d, dvl_q = du[15], du[16],du[19], du[20]

#         # algebraic variables are i_d, i_q, i_F, i_D, i_Q1, i_Q2, il_d, il_q

#         f[:] = (
#             # differential generator
#             dpsi_d + self.omega_b * (v_d - self.R_s * i_d + omega_m * psi_q),
#             dpsi_q + self.omega_b * (v_q - self.R_s * i_q - omega_m * psi_d),
#             dpsi_F + self.omega_b * (self.v_F - self.R_F * i_F),
#             dpsi_D + self.omega_b * (self.v_D - self.R_D * i_D),
#             dpsi_Q1 + self.omega_b * (self.v_Q1 - self.R_Q1 * i_Q1),
#             dpsi_Q2 + self.omega_b * (self.v_Q2 - self.R_Q2 * i_Q2),
#             -domega_m + 1 / (2 * self.H_) * (self.T_m - (psi_q * i_d - psi_d * i_q) - self.K_D * self.omega_b * (omega_m-1)),
#             # differential pi line
#             -dv_d + omega_m * v_q + 2/self.C_pi * (i_d - iz_d),
#             -dv_q - omega_m * v_d + 2/self.C_pi * (i_q - iz_q),
#             -dvl_d + omega_m * vl_q + 2/self.C_pi * (iz_d - il_d),
#             -dvl_q - omega_m * vl_d + 2/self.C_pi * (iz_q - il_q),
#             -diz_d - self.R_pi/self.L_pi * iz_d + omega_m * iz_q + (v_d - vl_d) / self.L_pi,
#             -diz_q - self.R_pi/self.L_pi * iz_q - omega_m * iz_d + (v_q - vl_q) / self.L_pi,
#             # algebraic generator
#             psi_d + self.L_d * i_d + self.L_md * i_F + self.L_md * i_D,
#             psi_q + self.L_q * i_q + self.L_mq * i_Q1 + self.L_mq * i_Q2,
#             psi_F + self.L_md * i_d + self.L_F * i_F + self.L_md * i_D,
#             psi_D + self.L_md * i_d + self.L_md * i_F + self.L_D * i_D,
#             psi_Q1 + self.L_mq * i_q + self.L_Q1 * i_Q1 + self.L_mq * i_Q2,
#             psi_Q2 + self.L_mq * i_q + self.L_mq * i_Q1 + self.L_Q2 * i_Q2,
#             # algebraic pi line
#             -il_d + vl_d/self.R_L,
#             -il_q + vl_q/self.R_L,
#         )
#         return f

#     def u_exact(self, t):
#         """
#         Approximation of the exact solution generated by spline interpolation of an extremely accurate numerical reference solution.
#         Args:
#             t (float): current time
#         Returns:
#             Mesh containing fixed initial value, 5 components
#         """
#         me = self.dtype_u(self.init)

#         if t == 0:
#             psi_d = 0.3971299
#             psi_q = 0.9219154
#             psi_F = 0.8374232
#             psi_D = 0.5795112
#             psi_Q1 = 0.8433430
#             psi_Q2 = 0.8433430
#             i_d = -1.215876
#             i_q = 0.5238156
#             i_F = 1.565
#             i_D = 0
#             i_Q1 = 0
#             i_Q2 = 0
#             v_d = -0.9362397
#             v_q = 0.4033005
#             omega_m = 1.0
#             # pi line
#             iz_d = -1.215875
#             iz_q = 0.5238151
#             il_d = -1.215875
#             il_q = 0.5238147
#             vl_d = -0.9119063
#             vl_q = 0.3928611
#             me[:] = (psi_d, psi_q, psi_F, psi_D, psi_Q1, psi_Q2,
#                 i_d, i_q, i_F, i_D, i_Q1, i_Q2,
#                 omega_m,
#                 v_d, v_q,
#                 iz_d, iz_q, il_d, il_q, vl_d, vl_q)
#         elif t < self.t_end:
#             me[:] = self.u_ref(t)
#         else:
#             warnings.warn("Requested time exceeds domain of the reference solution. Returning zero.")
#             me[:] = (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)

#         return me
