import numpy as np
from pySDC.projects.DAE.misc.ProblemDAE import ptype_dae
from pySDC.implementations.datatype_classes.mesh import mesh
from pySDC.core.Errors import ParameterError




class ThreeInverterSystem(ptype_dae):
    r"""
    Example system from F. Cecati
    """
    dtype_u = mesh
    dtype_f = mesh

    def __init__(self, nvars=None, newton_tol=1e-10):
        """Initialization routine"""

        nvars = 30
        # invoke super init, passing number of dofs
        super().__init__(nvars, newton_tol)
        self._makeAttributeAndRegister('nvars', 'newton_tol', localVars=locals(), readOnly=True)

        ## Parameters declaration
        self.omega = 2*np.pi*50;           # reference omega, f = 50 Hz
        self.jw =np.array([
                            [ 0, -self.omega],     # cross coupiling dq matrix
                            [self.omega,   0] 
                             ])       # cross coupiling dq matrix

        self.e = np.array([563, 0]).T
        self.Power1 = 1e6;  # 1 MW
        self.Power2 = 1e6;  # 1 MW
        self.Power3 = 1e6;  # 1 MW
        self.Power  = self.Power1 + self.Power2 + self.Power3
        self.fs = 2000
        self.Ts = 1/self.fs
        self.v_dc_ref = 1100
        self.v_ac_ref = 563
        self.C_dc = 22e-3

        # Distribution lines %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        self.Ll_12 = 0.01 / (2*np.pi*50)   
        self.Rl_12 = 2*np.pi*50 * self.Ll_12 * 2.5

        self.Ll_23 = 0.02 / (2*np.pi*50)
        self.Rl_23 = 2*np.pi*50 * self.Ll_23 * 1.5

        # PLL %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        self.PLL_D = 1/np.sqrt(2)
        self.PLL_Tsettle = 0.2  # original 0.2
        self.Vamp = 563

        self.Kp_PLL = 2*4.6 /self.PLL_Tsettle
        self.Ti_PLL = self.PLL_Tsettle*(self.PLL_D**2) / 2.3
        self.Ki_PLL = self.Kp_PLL/self.Ti_PLL

        self.Kp_PLL = self.Kp_PLL / self.Vamp
        self.Ki_PLL = self.Ki_PLL / self.Vamp

        # Current Loop %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        self.Lf = 0.1e-3    # L Filter
        self.Rf = 0.33*2*np.pi*50*self.Lf

        self.Kp  = self.Lf*1.8/(3*self.Ts)
        self.Ti  = self.Lf/self.Rf
        self.Ki  = self.Kp/self.Ti
        self.T_c = self.Kp/self.Lf

        # Dc and AC voltage Loop %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        self.Kp_DC = 0.12 * self.C_dc/self.Ts * 1.8   # DC Voltage loop
        self.Ti_DC = 17*self.Ts                  # DC Voltage loop
        self.Ki_DC = self.Kp_DC/self.Ti_DC / 4        # DC Voltage loop

        self.Kp_AC = -25 # AC Voltage droop


        # Grid Impedance

        self.SCR_des = 3.4 
        self.RXratio = 0.3

        self.Z_g = 690**2 / (self.Power * self.SCR_des)
        self.X_g = np.sqrt( self.Z_g**2 / ( self.RXratio**2 + 1 )   )

        self.L_g = self.X_g / self.omega
        self.R_g = self.RXratio * self.X_g


        ## Disturbance Simulation - Voltage sag

        self.phase_jump = -0*np.pi/9
        self.sag = 0.8
        self.v_after = np.dot( np.array([ [np.cos(self.phase_jump), np.sin(self.phase_jump)],
                                    [-np.sin(self.phase_jump), np.cos(self.phase_jump)]
                                    ]), np.array([563*self.sag, 0]).T) # SAG + PHASE JUMP


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

        # dEqp, dSi1d, dEdp = du[0 : self.m], du[self.m : 2 * self.m], du[2 * self.m : 3 * self.m]
        # dSi2q, dDelta = du[3 * self.m : 4 * self.m], du[4 * self.m : 5 * self.m]
        # dw, dEfd, dRF = du[5 * self.m : 6 * self.m], du[6 * self.m : 7 * self.m], du[7 * self.m : 8 * self.m]
        # dVR, dTM, dPSV = du[8 * self.m : 9 * self.m], du[9 * self.m : 10 * self.m], du[10 * self.m : 11 * self.m]

        # Eqp, Si1d, Edp = u[0 : self.m], u[self.m : 2 * self.m], u[2 * self.m : 3 * self.m]
        # Si2q, Delta = u[3 * self.m : 4 * self.m], u[4 * self.m : 5 * self.m]
        # w, Efd, RF = u[5 * self.m : 6 * self.m], u[6 * self.m : 7 * self.m], u[7 * self.m : 8 * self.m]
        # VR, TM, PSV = u[8 * self.m : 9 * self.m], u[9 * self.m : 10 * self.m], u[10 * self.m : 11 * self.m]

        # Id, Iq = u[11 * self.m : 11 * self.m + self.m], u[11 * self.m + self.m : 11 * self.m + 2 * self.m]
        # V = u[11 * self.m + 2 * self.m : 11 * self.m + 2 * self.m + self.n]
        # TH = u[11 * self.m + 2 * self.m + self.n : 11 * self.m + 2 * self.m + 2 * self.n]

        di_c1, dv_dc1, dphi_dc1, ddelta1, dphi_q1, dv_cc1 = du[0:2], du[2], du[3], du[4], du[5], du[6:8]
        di_c2, dv_dc2, dphi_dc2, ddelta2, dphi_q2, dv_cc2 = du[8:10] , du[10], du[11], du[12], du[13], du[14:16]     
        di_c3, dv_dc3, dphi_dc3, ddelta3, dphi_q3, dv_cc3 = du[16:18], du[18], du[19], du[20], du[21], du[22:24]
        di_pccDQ, dil_12DQ, dil_23DQ = du[24:26], du[26:28], du[28:30]         

    ## State definition and prelevation - Converter 1
        i_c1    = u[0:2]  
        v_dc1   = u[2]    
        phi_dc1 = u[3]    
        delta1  = u[4]    
        phi_q1  = u[5]    
        v_cc1   = u[6:8]  

        ## State definition and prelevation - Converter 2
        i_c2    = u[8:10] 
        v_dc2   = u[10]   
        phi_dc2 = u[11]   
        delta2  = u[12]   
        phi_q2  = u[13]   
        v_cc2   = u[14:16]

        ## State definition and prelevation - Converter 3
        i_c3    = u[16:18]
        v_dc3   = u[18]   
        phi_dc3 = u[19]   
        delta3  = u[20]   
        phi_q3  = u[21]   
        v_cc3   = u[22:24]

        ## State definition and prelevation - Power Network
        i_pccDQ  = u[24:26]
        il_12DQ  = u[26:28]
        il_23DQ  = u[28:30]

        # line outage disturbance:
        # if t >= 0.05:
        #     self.YBus = self.YBus_line6_8_outage
        
        # disturbance
        # at t=1 the disturbance occurs
        if(t > 0.2): 
            self.e = self.v_after; # voltage sag
        #       Power3 = 1e6;


        # Initialise f
        f = self.dtype_f(self.init)

        t_switch = np.inf if self.t_switch is None else self.t_switch

        # Equations as list

        eqs = []
        
        ## Power network nodal equations, according to the Kirkhoff laws
        i_g1DQ = i_pccDQ - il_12DQ
        i_g2DQ = il_12DQ - il_23DQ
        i_g3DQ = il_23DQ
        
        ## Equations Converter 1

        iq_ref1 = 0
        Tdelta1 = np.array([
            [np.cos(delta1), np.sin(delta1)],
            [-np.sin(delta1), np.cos(delta1)]
            ]) # from DQ to dq

        v_g1 = self.Kp*(i_c1 - np.dot(Tdelta1, i_g1DQ)) + v_cc1

        i_ref1 =  np.array([  
                    self.Kp_DC*(v_dc1 - self.v_dc_ref)  + self.Ki_DC * phi_dc1,
                    iq_ref1 + self.Kp_AC*(self.v_ac_ref - np.sqrt(np.dot(v_g1.T, v_g1)))
                    ]);   ###  Voltage controller
                
        wcc = self.Kp / self.Lf
        Tdelta1 = np.array([[np.cos(delta1), np.sin(delta1)],
                            [-np.sin(delta1), np.cos(delta1)]]) # from DQ to dq
        
        eqs.append(np.dot(-wcc, i_c1) + np.dot(wcc, i_ref1) - di_c1)
        eqs.append(-3/2 * (1/self.C_dc) * np.dot(np.dot(v_g1.T, Tdelta1), i_g1DQ)  / v_dc1 + 1/self.C_dc * self.Power1 / v_dc1 - dv_dc1)
        eqs.append(v_dc1 - self.v_dc_ref - dphi_dc1)
        eqs.append(np.dot(np.array([0, self.Kp_PLL]), v_g1)  + self.Ki_PLL * phi_q1 - ddelta1)
        eqs.append(np.dot(np.array([0, 1]), v_g1) - dphi_q1)
        eqs.append(-self.Ki * np.dot(Tdelta1, i_g1DQ) + self.Ki * i_c1 - dv_cc1)

        v_g1DQ = np.dot(Tdelta1.T, v_g1)

        ## Equations Converter 2
        iq_ref2 = 0
        Tdelta2 = np.array([
            [np.cos(delta2), np.sin(delta2)],
            [-np.sin(delta2), np.cos(delta2)]
            ]) # from DQ to dq
        
        v_g2 = self.Kp*(i_c2 - np.dot(Tdelta2, i_g2DQ)) + v_cc2

        i_ref2 =  np.array([  
                    self.Kp_DC*(v_dc2 - self.v_dc_ref)  + self.Ki_DC * phi_dc2,
                    iq_ref2 + self.Kp_AC*(self.v_ac_ref - np.sqrt(np.dot(v_g2.T, v_g2)))
                    ]);   ###  Voltage controller
        eqs.append(np.dot(-wcc, i_c2) + np.dot(wcc, i_ref2) - di_c2)
        eqs.append(-3/2 * (1/self.C_dc) * np.dot(np.dot(v_g2.T, Tdelta2), i_g2DQ) / v_dc2 + 1/self.C_dc * self.Power2 / v_dc2 - dv_dc2)
        eqs.append(v_dc2 - self.v_dc_ref - dphi_dc2)
        eqs.append(np.dot(np.array([0, self.Kp_PLL]), v_g2) + self.Ki_PLL * phi_q2 - ddelta2)
        eqs.append(np.dot(np.array([0, 1]), v_g2) - dphi_q2)
        eqs.append(-self.Ki * np.dot(Tdelta2, i_g2DQ) + self.Ki*i_c2 - dv_cc2)
        
        v_g2DQ = np.dot(Tdelta2.T, v_g2)

        ## Equations Converter 3
        iq_ref3 = -500
        Tdelta3 = np.array([
            [np.cos(delta3), np.sin(delta3)],
            [-np.sin(delta3), np.cos(delta3)]
            ]) # from DQ to dq
        
        v_g3 = self.Kp * (i_c3 - np.dot(Tdelta3, i_g3DQ)) + v_cc3

        i_ref3 =  np.array([  
                    self.Kp_DC*(v_dc3 - self.v_dc_ref)  + self.Ki_DC * phi_dc3,
                    iq_ref3 + self.Kp_AC * (self.v_ac_ref - np.sqrt(np.dot(v_g3.T, v_g3)))
                    ]);   ###  Voltage controller
        eqs.append(np.dot(-wcc, i_c3) + np.dot(wcc, i_ref3) - di_c3)
        eqs.append(-3/2 * (1/self.C_dc) * np.dot(np.dot(v_g3.T, Tdelta3), i_g3DQ) / v_dc3 + 1/self.C_dc * self.Power3 / v_dc3 - dv_dc3)
        eqs.append(v_dc3 - self.v_dc_ref - dphi_dc3)
        eqs.append(np.dot(np.array([0, self.Kp_PLL]), v_g3) + self.Ki_PLL * phi_q3 - ddelta3)
        eqs.append(np.dot(np.array([0, 1]), v_g3) - dphi_q3)
        eqs.append(-self.Ki * np.dot(Tdelta3, i_g3DQ) + self.Ki*i_c3 - dv_cc3)

        v_g3DQ = np.dot(Tdelta3.T, v_g3)
        ## Power network and grid equations, according to the Kirkhoff laws

        eqs.append(-(self.R_g)/(self.L_g)*i_pccDQ     - np.dot(self.jw, i_pccDQ) + 1/(self.L_g)   * (v_g1DQ - self.e.T) - di_pccDQ)          # Grid 
        eqs.append(-(self.Rl_12)/(self.Ll_12)*il_12DQ - np.dot(self.jw, il_12DQ) + 1/(self.Ll_12) * (v_g2DQ - v_g1DQ) - dil_12DQ)      # Transmission line 1-2
        eqs.append(-(self.Rl_23)/(self.Ll_23)*il_23DQ - np.dot(self.jw, il_23DQ) + 1/(self.Ll_23) * (v_g3DQ - v_g2DQ) - dil_23DQ)      # Transmission line 2-3

        eqs_flatten = [item for sublist in eqs for item in (sublist if isinstance(sublist,mesh) else [sublist])]
        # eqs_flatten = np.hstack(eqs)
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
        x_init = 1.0e+03 *   np.array([     1.6787,
                                        -0.6657,
                                            1.1000,
                                            0.0023,
                                            0.0010,
                                        -0.0000,
                                            0.4964,
                                            0.0000,
                                            1.5104,
                                        -0.1128,
                                            1.1000,
                                            0.0020,
                                            0.0011,
                                        -0.0000,
                                            0.5517,
                                        -0.0000,
                                            1.4409,
                                            0.1535,
                                            1.1000,
                                            0.0019,
                                            0.0011,
                                        -0.0000,
                                            0.5783,
                                        -0.0000,
                                            2.8237,
                                            3.7043,
                                            1.3960,
                                            2.5985,
                                            0.5559,
                                            1.3381 ,
                                            ]); # initial state with simulation flat start (to be computed by static analysis)
                
        me[0:2]  = x_init[0:2]  
        me[2]    = x_init[2]    
        me[3]    = x_init[3]    
        me[4]    = x_init[4]    
        me[5]    = x_init[5]    
        me[6:8]  = x_init[6:8]  
        me[8:10] = x_init[8:10] 
        me[10]   = x_init[10]   
        me[11]   = x_init[11]   
        me[12]   = x_init[12]   
        me[13]   = x_init[13]   
        me[14:16]= x_init[14:16]
        me[16:18]= x_init[16:18]
        me[18]   = x_init[18]   
        me[19]   = x_init[19]   
        me[20]   = x_init[20]   
        me[21]   = x_init[21]   
        me[22:24]= x_init[22:24]
        me[24:26]= x_init[24:26]
        me[26:28]= x_init[26:28]
        me[28:30]= x_init[28:30]
        return me

