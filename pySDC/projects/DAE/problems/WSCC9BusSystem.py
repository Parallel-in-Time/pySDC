import numpy as np
from pySDC.projects.DAE.misc.ProblemDAE import ptype_dae
from pySDC.implementations.datatype_classes.mesh import mesh
from pySDC.core.Errors import ParameterError


def WSCC9Bus():
    """
    Returns the Ybus for the power system.

    Returns
    -------
    ppc_res : dict
        The data with buses, branches, generators (with power flow result) and the Ybus to define the power system.
    """
    ppc_res = {}

    ppc_res['baseMVA'] = 100
    ppc_res['Ybus'] = get_initial_Ybus()
    ppc_res['bus'] = np.array(
        [
            [
                1.0,
                3.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                0.0,
                345.0,
                1.0,
                1.100000000000000089e00,
                9.000000000000000222e-01,
            ],
            [
                2.0,
                2.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                9.999999999999998890e-01,
                9.668741126628123794e00,
                345.0,
                1.0,
                1.100000000000000089e00,
                9.000000000000000222e-01,
            ],
            [
                3.0,
                2.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                9.999999999999998890e-01,
                4.771073237177319015e00,
                345.0,
                1.0,
                1.100000000000000089e00,
                9.000000000000000222e-01,
            ],
            [
                4.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                9.870068523919054426e-01,
                -2.406643919519410257e00,
                345.0,
                1.0,
                1.100000000000000089e00,
                9.000000000000000222e-01,
            ],
            [
                5.0,
                1.0,
                90.0,
                30.0,
                0.0,
                0.0,
                1.0,
                9.754721770850530715e-01,
                -4.017264326707549849e00,
                345.0,
                1.0,
                1.100000000000000089e00,
                9.000000000000000222e-01,
            ],
            [
                6.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.003375436452800251e00,
                1.925601686828564363e00,
                345.0,
                1.0,
                1.100000000000000089e00,
                9.000000000000000222e-01,
            ],
            [
                7.0,
                1.0,
                100.0,
                35.0,
                0.0,
                0.0,
                1.0,
                9.856448817249467975e-01,
                6.215445553889322738e-01,
                345.0,
                1.0,
                1.100000000000000089e00,
                9.000000000000000222e-01,
            ],
            [
                8.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                9.961852458090698637e-01,
                3.799120192692319264e00,
                345.0,
                1.0,
                1.100000000000000089e00,
                9.000000000000000222e-01,
            ],
            [
                9.0,
                1.0,
                125.0,
                50.0,
                0.0,
                0.0,
                1.0,
                9.576210404299042578e-01,
                -4.349933576561006987e00,
                345.0,
                1.0,
                1.100000000000000089e00,
                9.000000000000000222e-01,
            ],
        ]
    )
    ppc_res['branch'] = np.array(
        [
            [
                1.0,
                4.0,
                0.0,
                5.759999999999999842e-02,
                0.0,
                250.0,
                250.0,
                250.0,
                0.0,
                0.0,
                1.0,
                -360.0,
                360.0,
                7.195470158922189796e01,
                2.406895777275899206e01,
                -7.195470158922189796e01,
                -2.075304453873995314e01,
            ],
            [
                4.0,
                5.0,
                1.700000000000000122e-02,
                9.199999999999999845e-02,
                1.580000000000000016e-01,
                250.0,
                250.0,
                250.0,
                0.0,
                0.0,
                1.0,
                -360.0,
                360.0,
                3.072828027973678999e01,
                -5.858508226424568033e-01,
                -3.055468555805444453e01,
                -1.368795049942141873e01,
            ],
            [
                5.0,
                6.0,
                3.899999999999999994e-02,
                1.700000000000000122e-01,
                3.579999999999999849e-01,
                150.0,
                150.0,
                150.0,
                0.0,
                0.0,
                1.0,
                -360.0,
                360.0,
                -5.944531444194475966e01,
                -1.631204950057851022e01,
                6.089386583276659337e01,
                -1.242746953108854591e01,
            ],
            [
                3.0,
                6.0,
                0.0,
                5.859999999999999931e-02,
                0.0,
                300.0,
                300.0,
                300.0,
                0.0,
                0.0,
                1.0,
                -360.0,
                360.0,
                8.499999999999997158e01,
                -3.649025534209526800e00,
                -8.499999999999997158e01,
                7.890678351196221740e00,
            ],
            [
                6.0,
                7.0,
                1.190000000000000085e-02,
                1.008000000000000007e-01,
                2.089999999999999913e-01,
                150.0,
                150.0,
                150.0,
                0.0,
                0.0,
                1.0,
                -360.0,
                360.0,
                2.410613416723325741e01,
                4.536791179891427994e00,
                -2.401064777894146474e01,
                -2.440076244077697609e01,
            ],
            [
                7.0,
                8.0,
                8.500000000000000611e-03,
                7.199999999999999456e-02,
                1.489999999999999936e-01,
                250.0,
                250.0,
                250.0,
                0.0,
                0.0,
                1.0,
                -360.0,
                360.0,
                -7.598935222105758669e01,
                -1.059923755922268285e01,
                7.649556434279409700e01,
                2.562394697223899231e-01,
            ],
            [
                8.0,
                2.0,
                0.0,
                6.250000000000000000e-02,
                0.0,
                250.0,
                250.0,
                250.0,
                0.0,
                0.0,
                1.0,
                -360.0,
                360.0,
                -1.629999999999997158e02,
                2.276189879408803574e00,
                1.629999999999997442e02,
                1.446011953112515869e01,
            ],
            [
                8.0,
                9.0,
                3.200000000000000067e-02,
                1.610000000000000042e-01,
                3.059999999999999942e-01,
                250.0,
                250.0,
                250.0,
                0.0,
                0.0,
                1.0,
                -360.0,
                360.0,
                8.650443565720313188e01,
                -2.532429349130207452e00,
                -8.403988686535042518e01,
                -1.428198298779915731e01,
            ],
            [
                9.0,
                4.0,
                1.000000000000000021e-02,
                8.500000000000000611e-02,
                1.759999999999999898e-01,
                250.0,
                250.0,
                250.0,
                0.0,
                0.0,
                1.0,
                -360.0,
                360.0,
                -4.096011313464404680e01,
                -3.571801701219811775e01,
                4.122642130948177197e01,
                2.133889536138378062e01,
            ],
        ]
    )
    ppc_res['gen'] = np.array(
        [
            [
                1.0,
                71.0,
                24.0,
                300.0,
                -300.0,
                1.0,
                100.0,
                1.0,
                250.0,
                10.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                2.0,
                163.0,
                14.0,
                300.0,
                -300.0,
                1.0,
                100.0,
                1.0,
                300.0,
                10.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                3.0,
                85.0,
                -3.0,
                300.0,
                -300.0,
                1.0,
                100.0,
                1.0,
                270.0,
                10.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
        ]
    )

    return ppc_res


def get_initial_Ybus():
    ybus = np.array(
        [
            [0 - 17.36111111111111j, 0 + 0j, 0 + 0j, 0 + 17.36111111111111j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
            [0 + 0j, 0 - 16j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 16j, 0 + 0j],
            [0 + 0j, 0 + 0j, 0 - 17.06484641638225j, 0 + 0j, 0 + 0j, 0 + 17.06484641638225j, 0 + 0j, 0 + 0j, 0 + 0j],
            [
                0 + 17.36111111111111j,
                0 + 0j,
                0 + 0j,
                3.307378962025306 - 39.30888872611897j,
                -1.942191248714727 + 10.51068205186793j,
                0 + 0j,
                0 + 0j,
                0 + 0j,
                -1.36518771331058 + 11.60409556313993j,
            ],
            [
                0 + 0j,
                0 + 0j,
                0 + 0j,
                -1.942191248714727 + 10.51068205186793j,
                3.224200387138842 - 15.84092701422946j,
                -1.282009138424115 + 5.588244962361526j,
                0 + 0j,
                0 + 0j,
                0 + 0j,
            ],
            [
                0 + 0j,
                0 + 0j,
                0 + 17.06484641638225j,
                0 + 0j,
                -1.282009138424115 + 5.588244962361526j,
                2.437096619314212 - 32.15386180510696j,
                -1.155087480890097 + 9.784270426363173j,
                0 + 0j,
                0 + 0j,
            ],
            [
                0 + 0j,
                0 + 0j,
                0 + 0j,
                0 + 0j,
                0 + 0j,
                -1.155087480890097 + 9.784270426363173j,
                2.772209954136233 - 23.30324902327162j,
                -1.617122473246136 + 13.69797859690844j,
                0 + 0j,
            ],
            [
                0 + 0j,
                0 + 16j,
                0 + 0j,
                0 + 0j,
                0 + 0j,
                0 + 0j,
                -1.617122473246136 + 13.69797859690844j,
                2.804726852537284 - 35.44561313021703j,
                -1.187604379291148 + 5.975134533308591j,
            ],
            [
                0 + 0j,
                0 + 0j,
                0 + 0j,
                -1.36518771331058 + 11.60409556313993j,
                0 + 0j,
                0 + 0j,
                0 + 0j,
                -1.187604379291148 + 5.975134533308591j,
                2.552792092601728 - 17.33823009644852j,
            ],
        ],
        dtype=complex,
    )

    return ybus


def get_event_Ybus():
    ybus = np.array(
        [
            [0 - 17.36111111111111j, 0 + 0j, 0 + 0j, 0 + 17.36111111111111j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
            [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
            [0 + 0j, 0 + 0j, 0 - 17.06484641638225j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 17.06484641638225j],
            [
                0 + 17.36111111111111j,
                0 + 0j,
                0 + 0j,
                3.307378962025306 - 39.30888872611897j,
                -1.36518771331058 + 11.60409556313993j,
                -1.942191248714727 + 10.51068205186793j,
                0 + 0j,
                0 + 0j,
                0 + 0j,
            ],
            [
                0 + 0j,
                0 + 0j,
                0 + 0j,
                -1.36518771331058 + 11.60409556313993j,
                2.552792092601728 - 17.33823009644852j,
                0 + 0j,
                -1.187604379291148 + 5.975134533308591j,
                0 + 0j,
                0 + 0j,
            ],
            [
                0 + 0j,
                0 + 0j,
                0 + 0j,
                -1.942191248714727 + 10.51068205186793j,
                0 + 0j,
                3.224200387138842 - 15.84092701422946j,
                0 + 0j,
                0 + 0j,
                -1.282009138424115 + 5.588244962361526j,
            ],
            [
                0 + 0j,
                0 + 0j,
                0 + 0j,
                0 + 0j,
                -1.187604379291148 + 5.975134533308591j,
                0 + 0j,
                2.804726852537284 - 19.44561313021703j,
                -1.617122473246136 + 13.69797859690844j,
                0 + 0j,
            ],
            [
                0 + 0j,
                0 + 0j,
                0 + 0j,
                0 + 0j,
                0 + 0j,
                0 + 0j,
                -1.617122473246136 + 13.69797859690844j,
                2.772209954136233 - 23.30324902327162j,
                -1.155087480890097 + 9.784270426363173j,
            ],
            [
                0 + 0j,
                0 + 0j,
                0 + 17.06484641638225j,
                0 + 0j,
                0 + 0j,
                -1.282009138424115 + 5.588244962361526j,
                0 + 0j,
                -1.155087480890097 + 9.784270426363173j,
                2.437096619314212 - 32.15386180510696j,
            ],
        ],
        dtype=complex,
    )

    return ybus


# def get_YBus(ppc):

#     ppci = ext2int(ppc)
#     Ybus, yf, yt = makeYbus(ppci['baseMVA'],ppci['bus'],ppci['branch'])

#     return ppc['Ybus']()


class WSCC9BusSystem(ptype_dae):
    r"""
    Example implementing the WSCC 9 Bus system [1]_. For this complex model, the equations can be found in [2]_, and
    sub-transient and turbine parameters are taken from [3]_. The network data of the system are taken from MATPOWER [4]_.

    Parameters
    ----------
    nvars : int
        Number of unknowns of the system of DAEs (not used here, since it is set up inside this class).
    newton_tol : float
        Tolerance for Newton solver.

    Attributes
    ----------
    mpc : dict
        Contains the data for the buses, branches, generators, and the Ybus.
    m : int
        Number of machines used in the network.
    n : int
        Number of buses used in the network.
    baseMVA : float
        Base value of the apparent power.
    ws : float
        Generator synchronous speed in rad per second.
    ws_vector : np.1darray
        Vector containing ``ws``.
    MD : np.2darray
        Machine data.
    ED : np.2darray
        Excitation data.
    TD : np.2darray
        Turbine data.
    bus : np.2darray
        Data for the buses.
    branch : np.2darray
        Data for the branches in the power system.
    gen : np.2darray
        Data for generators in the system.
    Ybus : np.2darray
        Ybus.
    YBus_line6_8_outage : np.2darray
        Contains the data for the line outage in the power system, where line at bus6 is outaged.
    psv_max : float
         Maximum valve position.
    IC1 : list
        Contains the :math:`8`-th row of the ``bus`` matrix.
    IC2 : list
        Contains the :math:`9`-th row of the ``bus`` matrix.
    IC3 : list
        Generator values divided by ``baseMVA``.
    IC4 : list
        Generator values divided by ``baseMVA``.
    IC5 : list
        Loads divided by ``baseMVA``.
    IC6 : list
        Loads divided by ``baseMVA``.
    genP : list
        Contains data for one generator from the ``mpc``.
    IC : list
        Contains all the values of `IC1, IC2, IC3, IC4, IC5, IC6`.
    PL : list
        Contains the :math:`5`-th column of ``IC``.
    QL : list
        Contains the :math:`6`-th column of ``IC``.
    PG : np.1darray
        Contains the :math:`3`-rd column of ``IC``.
    QG : np.1darray
        Contains the :math:`4`-th column of ``IC``.
    TH0 : np.1darray
        Initial condition for angle of bus voltage in rad.
    V0 : np.1darray
        Contains the :math:`1`-st column of ``IC``, initial condition for magnitude of bus voltage in per unit.
    VG0 : np.1darray
        Initial condition for complex voltage phasor.
    THG0 : np.1darray
        Initial condition for angle of the bus voltage in rad.
    H : np.1darray
        Shaft inertia constant in second.
    Xd : np.1darray
        d-axis reactance in per unit.
    Xdp : np.1darray
        Transient d-axis reactance in per unit.
    Xdpp : np.1darray
        Sub-transient d-axis reactance in per unit.
    Xq : np.1darray
        q-axis reactance in per unit.
    Xqp : np.1darray
        Transient q-axis reactance in per unit.
    Xqpp : np.1darray
        Sub-transient q-axis reactance in per unit.
    Td0p : np.1darray
        d-axis time constant associated with :math:`E_q'` in second.
    Td0pp : np.1darray
        d-axis time constant associated with :math:`\psi_{1d}` in second.
    Tq0p : np.1darray
        q-axis time constant associated with :math:`E_d'` in second.
    Tq0pp : np.1darray
        q-axis time constant associated with :math:`\psi_{2q}` in second.
    Rs : np.1darray
        Stator resistance in per unit.
    Xls : np.1darray
        Parameter :math:`X_{\ell s}`.
    Dm : np.1darray
        Rotor angle in rad.
    KA : np.1darray
        Amplifier gain.
    TA : np.1darray
        Amplifier time constant in second.
    KE : np.1darray
        Separate or self-excited constant.
    TE : np.1darray
        Parameter :math:`T_E`.
    KF : np.1darray
        Parameter _math:`K_F`.
    TF : np.1darray
        Parameter :math:`T_F`.
    Ax : np.1darray
        Constant :math:`A_x` of the saturation function :math:`S_{E_i}`.
    Bx : np.1darray
        Constant :math:`B_x` of the saturation function :math:`S_{E_i}`.
    TCH : np.1darray
        Incremental steam chest time constant in second.
    TSV : np.1darray
        Steam valve time constant in second.
    RD : np.1darray
        Speed regulation quantity in Hz/per unit.
    MH : float
        Factor :math:`\frac{2 H_i}{w_s}`.
    QG : np.1darray
        Used to compute :math:`I_{phasor}`.
    Vphasor : np.1darray
        Complex voltage phasor.
    Iphasor : np.1darray
        Complex current phasor.
    E0 : np.1darray
        Initial internal voltage of the synchronous generator.
    Em : np.1darray
        Absolute values of ``E0``.
    D0 : np.1darray
        Initial condition for rotor angle in rad.
    Id0 : np.1darray
        Initial condition for d-axis current in per unit.
    Iq0 : np.1darray
        Initial condition for q-axis current in per unit.
    Edp0 : np.1darray
        Initial condition for d-axis transient internal voltages in per unit.
    Si2q0 : np.1darray
        Initial condition for damper winding 2q flux linkages in per unit.
    Eqp0 : np.1darray
        Initial condition for q-axis transient internal voltages in per unit.
    Si1d0 : np.1darray
        Initial condition for damper winding 1d flux linkages in per unit.
    Efd0 : np.1darray
        Initial condition for field winding fd flux linkages in per unit.
    TM0 : np.1darray
        Initial condition for mechanical input torque in per unit.
    VR0 : np.1darray
        Initial condition for exciter input in per unit.
    RF0 : np.1darray
        Initial condition for exciter feedback in per unit.
    Vref : np.1darray
        Reference voltage input in per unit.
    PSV0 : np.1darray
        Initial condition for steam valve position in per unit.
    PC : np.1darray
        Initial condition for control power input in per unit.
    alpha : int
        Active load parameter.
    beta : int
        Reactive load parameter.
    bb1, aa1 : list of ndarrays
        Used to access on specific values of ``TH``.
    bb2, aa2 : list of ndarrays
        Used to access on specific values of ``TH``.
    t_switch : float
        Time the event found by detection.
    nswitches : int
        Number of events found by detection.

    References
    ----------
    .. [1] WSCC 9-Bus System - Illinois Center for a Smarter Electric Grid. https://icseg.iti.illinois.edu/wscc-9-bus-system/
    .. [2] P. W. Sauer, M. A. Pai. Power System Dynamics and Analysis. John Wiley & Sons (2008).
    .. [3] I. Abdulrahman. MATLAB-Based Programs for Power System Dynamics Analysis. IEEE Open Access Journal of Power and Energy.
       Vol. 7, No. 1, pp. 59–69 (2020).
    .. [4] R. D. Zimmerman, C. E. Murillo-Sánchez, R. J. Thomas. MATPOWER: Steady-State Operations, Planning, and Analysis Tools
       for Power Systems Research and Education. IEEE Transactions on Power Systems. Vol. 26, No. 1, pp. 12–19 (2011).
    """

    dtype_u = mesh
    dtype_f = mesh

    def __init__(self, nvars=None, newton_tol=1e-10, m=3, n=9):
        """Initialization routine"""

        nvars = 11 * m + 2 * m + 2 * n
        # invoke super init, passing number of dofs
        super().__init__(nvars, newton_tol)
        self._makeAttributeAndRegister('nvars', 'newton_tol', localVars=locals(), readOnly=True)
        self._makeAttributeAndRegister('m', 'n', localVars=locals())
        self.mpc = WSCC9Bus()

        self.baseMVA = self.mpc['baseMVA']
        self.ws = 2 * np.pi * 60
        self.ws_vector = self.ws * np.ones(self.m)

        # Machine data (MD) as a 2D NumPy array
        self.MD = np.array(
            [
                [23.640, 6.4000, 3.0100],  # 1 - H
                [0.1460, 0.8958, 1.3125],  # 2 - Xd
                [0.0608, 0.1198, 0.1813],  # 3 - Xdp
                [0.0489, 0.0881, 0.1133],  # 4 - Xdpp
                [0.0969, 0.8645, 1.2578],  # 5 - Xq
                [0.0969, 0.1969, 0.2500],  # 6 - Xqp
                [0.0396, 0.0887, 0.0833],  # 7 - Xqpp
                [8.960000000000001, 6.0000, 5.8900],  # 8 - Tdop
                [0.1150, 0.0337, 0.0420],  # 9 - Td0pp
                [0.3100, 0.5350, 0.6000],  # 10 - Tqop
                [0.0330, 0.0780, 0.1875],  # 11 - Tq0pp
                [0.0041, 0.0026, 0.0035],  # 12 - RS
                [0.1200, 0.1020, 0.0750],  # 13 - Xls
                [
                    0.1 * (2 * 23.64) / self.ws,
                    0.2 * (2 * 6.4) / self.ws,
                    0.3 * (2 * 3.01) / self.ws,
                ],  # 14 - Dm (ws should be defined)
            ]
        )

        # Excitation data (ED) as a 2D NumPy array
        self.ED = np.array(
            [
                20.000 * np.ones(self.m),  # 1 - KA
                0.2000 * np.ones(self.m),  # 2 - TA
                1.0000 * np.ones(self.m),  # 3 - KE
                0.3140 * np.ones(self.m),  # 4 - TE
                0.0630 * np.ones(self.m),  # 5 - KF
                0.3500 * np.ones(self.m),  # 6 - TF
                0.0039 * np.ones(self.m),  # 7 - Ax
                1.5550 * np.ones(self.m),  # 8 - Bx
            ]
        )

        # Turbine data (TD) as a 2D NumPy array
        self.TD = np.array(
            [
                0.10 * np.ones(self.m),  # 1 - TCH
                0.05 * np.ones(self.m),  # 2 - TSV
                0.05 * np.ones(self.m),  # 3 - RD
            ]
        )

        self.bus = self.mpc['bus']
        self.branch = self.mpc['branch']
        self.gen = self.mpc['gen']
        self.YBus = get_initial_Ybus()

        temp_mpc = self.mpc
        temp_mpc['branch'] = np.delete(
            temp_mpc['branch'], 6, 0
        )  # note that this is correct but not necessary, because we have the event Ybus already
        self.YBus_line6_8_outage = get_event_Ybus()

        # excitation limiter vmax
        # self.vmax = 2.1
        self.psv_max = 1.0

        self.IC1 = [row[7] for row in self.bus]  # Column 8 in MATLAB is indexed as 7 in Python (0-based index)
        self.IC2 = [row[8] for row in self.bus]  # Column 9 in MATLAB is indexed as 8 in Python

        n_prev, m_prev = self.n, self.m
        self.n = len(self.bus)  # Number of rows in 'bus' list; self.n already defined above?!
        self.m = len(self.gen)  # Number of rows in 'gen' list; self.m already defined above?!
        if n_prev != 9 or m_prev != 3:
            raise ParameterError("Number of rows in bus or gen not equal to initialised n or m!")

        gen0 = [0] * self.n
        for i in range(self.m):
            gen0[i] = self.gen[i][1]
        self.genP = gen0
        self.IC3 = [val / self.baseMVA for val in self.genP]

        gen0 = [0] * self.n
        for i in range(self.m):
            gen0[i] = self.gen[i][2]
        genQ = gen0
        for i in range(self.n):
            genQ[i] += self.bus[i][5]  # Column 6 in MATLAB is indexed as 5 in Python
        self.IC4 = [val / self.baseMVA for val in genQ]

        self.IC5 = [row[2] for row in self.bus]  # Column 3 in MATLAB is indexed as 2 in Python
        self.IC5 = [val / self.baseMVA for val in self.IC5]

        self.IC6 = [row[3] for row in self.bus]  # Column 4 in MATLAB is indexed as 3 in Python
        self.IC6 = [val / self.baseMVA for val in self.IC6]

        self.IC = list(zip(self.IC1, self.IC2, self.IC3, self.IC4, self.IC5, self.IC6))

        self.PL = [row[4] for row in self.IC]  # Column 5 in MATLAB is indexed as 4 in Python
        self.QL = [row[5] for row in self.IC]  # Column 6 in MATLAB is indexed as 5 in Python

        self.PG = np.array([row[2] for row in self.IC])  # Column 3 in MATLAB is indexed as 2 in Python
        self.QG = np.array([row[3] for row in self.IC])  # Column 4 in MATLAB is indexed as 3 in Python

        self.TH0 = np.array([row[1] * np.pi / 180 for row in self.IC])
        self.V0 = np.array([row[0] for row in self.IC])
        self.VG0 = self.V0[: self.m]
        self.THG0 = self.TH0[: self.m]

        # Extracting values from the MD array
        self.H = self.MD[0, :]
        self.Xd = self.MD[1, :]
        self.Xdp = self.MD[2, :]
        self.Xdpp = self.MD[3, :]
        self.Xq = self.MD[4, :]
        self.Xqp = self.MD[5, :]
        self.Xqpp = self.MD[6, :]
        self.Td0p = self.MD[7, :]
        self.Td0pp = self.MD[8, :]
        self.Tq0p = self.MD[9, :]
        self.Tq0pp = self.MD[10, :]
        self.Rs = self.MD[11, :]
        self.Xls = self.MD[12, :]
        self.Dm = self.MD[13, :]

        # Extracting values from the ED array
        self.KA = self.ED[0, :]
        self.TA = self.ED[1, :]
        self.KE = self.ED[2, :]
        self.TE = self.ED[3, :]
        self.KF = self.ED[4, :]
        self.TF = self.ED[5, :]
        self.Ax = self.ED[6, :]
        self.Bx = self.ED[7, :]

        # Extracting values from the TD array
        self.TCH = self.TD[0, :]
        self.TSV = self.TD[1, :]
        self.RD = self.TD[2, :]

        # Calculate MH
        self.MH = 2 * self.H / self.ws

        # Represent QG as complex numbers
        self.QG = self.QG.astype(complex)

        # Convert VG0 and THG0 to complex phasors
        self.Vphasor = self.VG0 * np.exp(1j * self.THG0)

        # Calculate Iphasor
        self.Iphasor = np.conj(np.divide(self.PG[:m] + 1j * self.QG[:m], self.Vphasor))

        # Calculate E0
        self.E0 = self.Vphasor + (self.Rs + 1j * self.Xq) * self.Iphasor

        # Calculate Em, D0, Id0, and Iq0
        self.Em = np.abs(self.E0)
        self.D0 = np.angle(self.E0)
        self.Id0 = np.real(self.Iphasor * np.exp(-1j * (self.D0 - np.pi / 2)))
        self.Iq0 = np.imag(self.Iphasor * np.exp(-1j * (self.D0 - np.pi / 2)))

        # Calculate Edp0, Si2q0, Eqp0, and Si1d0
        self.Edp0 = (self.Xq - self.Xqp) * self.Iq0
        self.Si2q0 = (self.Xls - self.Xq) * self.Iq0
        self.Eqp0 = self.Rs * self.Iq0 + self.Xdp * self.Id0 + self.V0[: self.m] * np.cos(self.D0 - self.TH0[: self.m])
        self.Si1d0 = self.Eqp0 - (self.Xdp - self.Xls) * self.Id0

        # Calculate Efd0 and TM0
        self.Efd0 = self.Eqp0 + (self.Xd - self.Xdp) * self.Id0
        self.TM0 = (
            ((self.Xdpp - self.Xls) / (self.Xdp - self.Xls)) * self.Eqp0 * self.Iq0
            + ((self.Xdp - self.Xdpp) / (self.Xdp - self.Xls)) * self.Si1d0 * self.Iq0
            + ((self.Xqpp - self.Xls) / (self.Xqp - self.Xls)) * self.Edp0 * self.Id0
            - ((self.Xqp - self.Xqpp) / (self.Xqp - self.Xls)) * self.Si2q0 * self.Id0
            + (self.Xqpp - self.Xdpp) * self.Id0 * self.Iq0
        )

        # Calculate VR0 and RF0
        self.VR0 = (self.KE + self.Ax * np.exp(self.Bx * self.Efd0)) * self.Efd0
        self.RF0 = (self.KF / self.TF) * self.Efd0

        # Calculate Vref and PSV0
        self.Vref = self.V0[: self.m] + self.VR0 / self.KA
        self.PSV0 = self.TM0
        self.PC = self.PSV0

        self.alpha = 2
        self.beta = 2

        self.bb1, self.aa1 = np.meshgrid(np.arange(0, self.m), np.arange(0, self.n))
        self.bb1, self.aa1 = self.bb1.astype(int), self.aa1.astype(int)

        # Create matrix grid to eliminate for-loops (load buses)
        self.bb2, self.aa2 = np.meshgrid(np.arange(self.m, self.n), np.arange(0, self.n))
        self.bb2, self.aa2 = self.bb2.astype(int), self.aa2.astype(int)

        self.t_switch = None
        self.nswitches = 0

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
            The right-hand side of f (contains two components).
        """

        dEqp, dSi1d, dEdp = du[0 : self.m], du[self.m : 2 * self.m], du[2 * self.m : 3 * self.m]
        dSi2q, dDelta = du[3 * self.m : 4 * self.m], du[4 * self.m : 5 * self.m]
        dw, dEfd, dRF = du[5 * self.m : 6 * self.m], du[6 * self.m : 7 * self.m], du[7 * self.m : 8 * self.m]
        dVR, dTM, dPSV = du[8 * self.m : 9 * self.m], du[9 * self.m : 10 * self.m], du[10 * self.m : 11 * self.m]

        Eqp, Si1d, Edp = u[0 : self.m], u[self.m : 2 * self.m], u[2 * self.m : 3 * self.m]
        Si2q, Delta = u[3 * self.m : 4 * self.m], u[4 * self.m : 5 * self.m]
        w, Efd, RF = u[5 * self.m : 6 * self.m], u[6 * self.m : 7 * self.m], u[7 * self.m : 8 * self.m]
        VR, TM, PSV = u[8 * self.m : 9 * self.m], u[9 * self.m : 10 * self.m], u[10 * self.m : 11 * self.m]

        Id, Iq = u[11 * self.m : 11 * self.m + self.m], u[11 * self.m + self.m : 11 * self.m + 2 * self.m]
        V = u[11 * self.m + 2 * self.m : 11 * self.m + 2 * self.m + self.n]
        TH = u[11 * self.m + 2 * self.m + self.n : 11 * self.m + 2 * self.m + 2 * self.n]

        # line outage disturbance:
        if t >= 0.05:
            self.YBus = self.YBus_line6_8_outage

        self.Yang = np.angle(self.YBus)
        self.Yabs = np.abs(self.YBus)

        COI = np.sum(w * self.MH) / np.sum(self.MH)

        # Voltage-dependent active loads PL2, and voltage-dependent reactive loads QL2
        PL2 = np.array(self.PL)
        QL2 = np.array(self.QL)

        V = V.T

        # Vectorized calculations
        Vectorized_angle1 = (
            np.array([TH.take(indices) for indices in self.bb1.T])
            - np.array([TH.take(indices) for indices in self.aa1.T])
            - self.Yang[: self.m, : self.n]
        )
        Vectorized_mag1 = (V[: self.m] * V[: self.n].reshape(-1, 1)).T * self.Yabs[: self.m, : self.n]

        sum1 = np.sum(Vectorized_mag1 * np.cos(Vectorized_angle1), axis=1)
        sum2 = np.sum(Vectorized_mag1 * np.sin(Vectorized_angle1), axis=1)

        VG = V[: self.m]
        THG = TH[: self.m]
        Angle_diff = Delta - THG

        Vectorized_angle2 = (
            np.array([TH.take(indices) for indices in self.bb2.T])
            - np.array([TH.take(indices) for indices in self.aa2.T])
            - self.Yang[self.m : self.n, : self.n]
        )
        Vectorized_mag2 = (V[self.m : self.n] * V[: self.n].reshape(-1, 1)).T * self.Yabs[self.m : self.n, : self.n]

        sum3 = np.sum(Vectorized_mag2 * np.cos(Vectorized_angle2), axis=1)
        sum4 = np.sum(Vectorized_mag2 * np.sin(Vectorized_angle2), axis=1)

        # Initialise f
        f = self.dtype_f(self.init)

        t_switch = np.inf if self.t_switch is None else self.t_switch

        # Equations as list
        eqs = []
        eqs.append(
            (1.0 / self.Td0p)
            * (
                -Eqp
                - (self.Xd - self.Xdp)
                * (
                    Id
                    - ((self.Xdp - self.Xdpp) / (self.Xdp - self.Xls) ** 2) * (Si1d + (self.Xdp - self.Xls) * Id - Eqp)
                )
                + Efd
            )
            - dEqp
        )  # (1)
        eqs.append((1.0 / self.Td0pp) * (-Si1d + Eqp - (self.Xdp - self.Xls) * Id) - dSi1d)  # (2)
        eqs.append(
            (1.0 / self.Tq0p)
            * (
                -Edp
                + (self.Xq - self.Xqp)
                * (
                    Iq
                    - ((self.Xqp - self.Xqpp) / (self.Xqp - self.Xls) ** 2) * (Si2q + (self.Xqp - self.Xls) * Iq + Edp)
                )
            )
            - dEdp
        )  # (3)
        eqs.append((1.0 / self.Tq0pp) * (-Si2q - Edp - (self.Xqp - self.Xls) * Iq) - dSi2q)  # (4)
        eqs.append(w - COI - dDelta)  # (5)
        eqs.append(
            (self.ws / (2.0 * self.H))
            * (
                TM
                - ((self.Xdpp - self.Xls) / (self.Xdp - self.Xls)) * Eqp * Iq
                - ((self.Xdp - self.Xdpp) / (self.Xdp - self.Xls)) * Si1d * Iq
                - ((self.Xqpp - self.Xls) / (self.Xqp - self.Xls)) * Edp * Id
                + ((self.Xqp - self.Xqpp) / (self.Xqp - self.Xls)) * Si2q * Id
                - (self.Xqpp - self.Xdpp) * Id * Iq
                - self.Dm * (w - self.ws)
            )
            - dw
        )  # (6)
        eqs.append((1.0 / self.TE) * ((-(self.KE + self.Ax * np.exp(self.Bx * Efd))) * Efd + VR) - dEfd)  # (7)
        eqs.append((1.0 / self.TF) * (-RF + (self.KF / self.TF) * Efd) - dRF)  # (8)
        eqs.append(
            (1.0 / self.TA)
            * (-VR + self.KA * RF - ((self.KA * self.KF) / self.TF) * Efd + self.KA * (self.Vref - V[: self.m]))
            - dVR
        )  # (9)

        # Limitation of valve position Psv with limiter start
        if PSV[0] >= self.psv_max or t >= t_switch:
            _temp_dPSV_g1 = (1.0 / self.TSV[1]) * (
                -PSV[1] + self.PSV0[1] - (1.0 / self.RD[1]) * (w[1] / self.ws - 1)
            ) - dPSV[1]
            _temp_dPSV_g2 = (1.0 / self.TSV[2]) * (
                -PSV[2] + self.PSV0[2] - (1.0 / self.RD[2]) * (w[2] / self.ws - 1)
            ) - dPSV[2]
            eqs.append(np.array([dPSV[0], _temp_dPSV_g1, _temp_dPSV_g2]))
        else:
            eqs.append((1.0 / self.TSV) * (-PSV + self.PSV0 - (1.0 / self.RD) * (w / self.ws - 1)) - dPSV)
        # Limitation of valve position Psv with limiter end

        eqs.append((1.0 / self.TCH) * (-TM + PSV) - dTM)  # (10)
        eqs.append(
            self.Rs * Id
            - self.Xqpp * Iq
            - ((self.Xqpp - self.Xls) / (self.Xqp - self.Xls)) * Edp
            + ((self.Xqp - self.Xqpp) / (self.Xqp - self.Xls)) * Si2q
            + VG * np.sin(Angle_diff)
        )  # (12)
        eqs.append(
            self.Rs * Iq
            + self.Xdpp * Id
            - ((self.Xdpp - self.Xls) / (self.Xdp - self.Xls)) * Eqp
            - ((self.Xdp - self.Xdpp) / (self.Xdp - self.Xls)) * Si1d
            + VG * np.cos(Angle_diff)
        )  # (13)
        eqs.append((Id * VG.T * np.sin(Angle_diff) + Iq * VG.T * np.cos(Angle_diff)) - PL2[0 : self.m] - sum1)  # (14)
        eqs.append((Id * VG.T * np.cos(Angle_diff) - Iq * VG.T * np.sin(Angle_diff)) - QL2[0 : self.m] - sum2)  # (15)
        eqs.append(-PL2[self.m : self.n] - sum3)  # (16)
        eqs.append(-QL2[self.m : self.n] - sum4)  # (17)
        eqs_flatten = [item for sublist in eqs for item in sublist]

        f[:] = eqs_flatten
        return f

    def u_exact(self, t):
        r"""
        Returns the initial conditions at time :math:`t=0`.

        Parameters
        ----------
        t : float
            Time of the initial conditions.

        Returns
        -------
        me : dtype_u
            Initial conditions.
        """
        assert t == 0, 'ERROR: u_exact only valid for t=0'

        me = self.dtype_u(self.init)
        me[0 : self.m] = self.Eqp0
        me[self.m : 2 * self.m] = self.Si1d0
        me[2 * self.m : 3 * self.m] = self.Edp0
        me[3 * self.m : 4 * self.m] = self.Si2q0
        me[4 * self.m : 5 * self.m] = self.D0
        me[5 * self.m : 6 * self.m] = self.ws_vector
        me[6 * self.m : 7 * self.m] = self.Efd0
        me[7 * self.m : 8 * self.m] = self.RF0
        me[8 * self.m : 9 * self.m] = self.VR0
        me[9 * self.m : 10 * self.m] = self.TM0
        me[10 * self.m : 11 * self.m] = self.PSV0
        me[11 * self.m : 11 * self.m + self.m] = self.Id0
        me[11 * self.m + self.m : 11 * self.m + 2 * self.m] = self.Iq0
        me[11 * self.m + 2 * self.m : 11 * self.m + 2 * self.m + self.n] = self.V0
        me[11 * self.m + 2 * self.m + self.n : 11 * self.m + 2 * self.m + 2 * self.n] = self.TH0
        return me

    def get_switching_info(self, u, t, du=None):
        r"""
        Provides information about the state function of the problem. When the state function changes its sign,
        typically an event occurs. So the check for an event should be done in the way that the state function
        is checked for a sign change. If this is the case, the intermediate value theorem states a root in this
        step.

        The state function for this problem is given by

        .. math::
           h(P_{SV,1}(t)) = P_{SV,1}(t) - P_{SV,1, max}

        for :math:`P_{SV,1,max}=1.0`.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution at time :math:`t`.
        t : float
            Current time of the numerical solution.

        Returns
        -------
        switch_detected : bool
            Indicates whether a discrete event is found or not.
        m_guess : int
            The index before the sign changes.
        state_function : list
            Defines the values of the state function at collocation nodes where it changes the sign.
        """

        switch_detected = False
        m_guess = -100
        for m in range(1, len(u)):
            h_prev_node = u[m - 1][10 * self.m] - self.psv_max
            h_curr_node = u[m][10 * self.m] - self.psv_max
            if h_prev_node < 0 and h_curr_node >= 0:
                switch_detected = True
                m_guess = m - 1
                break

        state_function = [u[m][10 * self.m] - self.psv_max for m in range(len(u))]
        return switch_detected, m_guess, state_function

    def count_switches(self):
        """
        Setter to update the number of switches if one is found.
        """
        self.nswitches += 1
