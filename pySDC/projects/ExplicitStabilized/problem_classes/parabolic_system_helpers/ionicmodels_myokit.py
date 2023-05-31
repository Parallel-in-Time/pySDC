import numpy as np
from dolfinx import fem, mesh
from mpi4py import MPI
import ufl
from pySDC.core.Errors import ParameterError
from pySDC.projects.ExplicitStabilized.problem_classes.parabolic_system_helpers.ionicmodels import IonicModel

def NV_Ith_S(y,k):
    return y.sub(k)

# Non stiff with rho_max ~ 40
class HodgkinHuxley(IonicModel):
    def __init__(self):
        super(HodgkinHuxley,self).__init__()

        self.size = 4

        # Set values of constants
        self.AC_g_L = 0.3
        self.AC_Cm = 1.0
        self.AC_E_R = -75.0
        self.AC_E_K = self.AC_E_R - 12.0
        self.AC_g_K = 36.0
        self.AC_E_Na = self.AC_E_R + 115.0
        self.AC_g_Na = 120.0
        self.AC_E_L = self.AC_E_R + 10.613

        self.f_nonstiff_args = [0,1,2,3]
        self.f_stiff_args = [0,1]
        self.f_expl_args = [0,1,2,3]

    def f(self, y):

        ydot = [None]*self.size
            
        AV_alpha_n = (-0.01) * (NV_Ith_S(y, 0) + 65.0) / (ufl.exp((-(NV_Ith_S(y, 0) + 65.0)) / 10.0) - 1.0)
        AV_beta_n = 0.125 * ufl.exp((NV_Ith_S(y, 0) + 75.0) / 80.0)
        ydot[3] = AV_alpha_n * (1.0 - NV_Ith_S(y, 3)) - AV_beta_n * NV_Ith_S(y, 3)
        
        AV_alpha_h = 0.07 * ufl.exp((-(NV_Ith_S(y, 0) + 75.0)) / 20.0)
        AV_beta_h = 1.0 / (ufl.exp((-(NV_Ith_S(y, 0) + 45.0)) / 10.0) + 1.0)
        ydot[2] = AV_alpha_h * (1.0 - NV_Ith_S(y, 2)) - AV_beta_h * NV_Ith_S(y, 2)
        
        AV_alpha_m = (-0.1) * (NV_Ith_S(y, 0) + 50.0) / (ufl.exp((-(NV_Ith_S(y, 0) + 50.0)) / 10.0) - 1.0)
        AV_beta_m = 4.0 * ufl.exp((-(NV_Ith_S(y, 0) + 75.0)) / 18.0)
        ydot[1] = AV_alpha_m * (1.0 - NV_Ith_S(y, 1)) - AV_beta_m * NV_Ith_S(y, 1)
                
        AV_i_K = self.AC_g_K * pow(NV_Ith_S(y, 3), 4.0) * (NV_Ith_S(y, 0) - self.AC_E_K)        
        AV_i_Na = self.AC_g_Na * pow(NV_Ith_S(y, 1), 3.0) * NV_Ith_S(y, 2) * (NV_Ith_S(y, 0) - self.AC_E_Na)        
        AV_i_L = self.AC_g_L * (NV_Ith_S(y, 0) - self.AC_E_L)
        ydot[0] = -( AV_i_Na + AV_i_K + AV_i_L)

        return ydot
    
    def f_expl(self, y):
        ydot = [None]*self.size            
                
        AV_i_K = self.AC_g_K * pow(NV_Ith_S(y, 3), 4.0) * (NV_Ith_S(y, 0) - self.AC_E_K)        
        AV_i_Na = self.AC_g_Na * pow(NV_Ith_S(y, 1), 3.0) * NV_Ith_S(y, 2) * (NV_Ith_S(y, 0) - self.AC_E_Na)        
        AV_i_L = self.AC_g_L * (NV_Ith_S(y, 0) - self.AC_E_L)
        ydot[0] = -( AV_i_Na + AV_i_K + AV_i_L)

        return ydot
    
    def u_exp_coeffs(self, y):
        yinf = [None]*self.size
        tau = [None]*self.size
        
        AV_alpha_n = (-0.01) * (NV_Ith_S(y, 0) + 65.0) / (ufl.exp((-(NV_Ith_S(y, 0) + 65.0)) / 10.0) - 1.0)
        AV_beta_n = 0.125 * ufl.exp((NV_Ith_S(y, 0) + 75.0) / 80.0)
        tau[3] = 1/(AV_alpha_n+AV_beta_n)
        yinf[3] = AV_alpha_n*tau[3]
        
        AV_alpha_h = 0.07 * ufl.exp((-(NV_Ith_S(y, 0) + 75.0)) / 20.0)
        AV_beta_h = 1.0 / (ufl.exp((-(NV_Ith_S(y, 0) + 45.0)) / 10.0) + 1.0)
        tau[2] = 1./(AV_alpha_h+AV_beta_h)
        yinf[2] = AV_alpha_h*tau[2]
        
        AV_alpha_m = (-0.1) * (NV_Ith_S(y, 0) + 50.0) / (ufl.exp((-(NV_Ith_S(y, 0) + 50.0)) / 10.0) - 1.0)
        AV_beta_m = 4.0 * ufl.exp((-(NV_Ith_S(y, 0) + 75.0)) / 18.0)
        tau[1] = 1/(AV_alpha_m+AV_beta_m)
        yinf[1] = AV_alpha_m*tau[1]    

        return yinf,tau
    
    def f_nonstiff(self, y):
        ydot = [None]*self.size
            
        AV_alpha_n = (-0.01) * (NV_Ith_S(y, 0) + 65.0) / (ufl.exp((-(NV_Ith_S(y, 0) + 65.0)) / 10.0) - 1.0)
        AV_beta_n = 0.125 * ufl.exp((NV_Ith_S(y, 0) + 75.0) / 80.0)
        ydot[3] = AV_alpha_n * (1.0 - NV_Ith_S(y, 3)) - AV_beta_n * NV_Ith_S(y, 3)
        
        AV_alpha_h = 0.07 * ufl.exp((-(NV_Ith_S(y, 0) + 75.0)) / 20.0)
        AV_beta_h = 1.0 / (ufl.exp((-(NV_Ith_S(y, 0) + 45.0)) / 10.0) + 1.0)
        ydot[2] = AV_alpha_h * (1.0 - NV_Ith_S(y, 2)) - AV_beta_h * NV_Ith_S(y, 2)    

        # AV_alpha_m = (-0.1) * (NV_Ith_S(y, 0) + 50.0) / (ufl.exp((-(NV_Ith_S(y, 0) + 50.0)) / 10.0) - 1.0)
        # AV_beta_m = 4.0 * ufl.exp((-(NV_Ith_S(y, 0) + 75.0)) / 18.0)
        # ydot[1] = AV_alpha_m * (1.0 - NV_Ith_S(y, 1)) - AV_beta_m * NV_Ith_S(y, 1)            
                
        AV_i_K = self.AC_g_K * pow(NV_Ith_S(y, 3), 4.0) * (NV_Ith_S(y, 0) - self.AC_E_K)        
        AV_i_Na = self.AC_g_Na * pow(NV_Ith_S(y, 1), 3.0) * NV_Ith_S(y, 2) * (NV_Ith_S(y, 0) - self.AC_E_Na)        
        AV_i_L = self.AC_g_L * (NV_Ith_S(y, 0) - self.AC_E_L)
        ydot[0] = -( AV_i_Na + AV_i_K + AV_i_L)

        return ydot
    
    def f_stiff(self, y):
        ydot = [None]*self.size            
        
        # AV_alpha_n = (-0.01) * (NV_Ith_S(y, 0) + 65.0) / (ufl.exp((-(NV_Ith_S(y, 0) + 65.0)) / 10.0) - 1.0)
        # AV_beta_n = 0.125 * ufl.exp((NV_Ith_S(y, 0) + 75.0) / 80.0)
        # ydot[3] = AV_alpha_n * (1.0 - NV_Ith_S(y, 3)) - AV_beta_n * NV_Ith_S(y, 3)
        
        # AV_alpha_h = 0.07 * ufl.exp((-(NV_Ith_S(y, 0) + 75.0)) / 20.0)
        # AV_beta_h = 1.0 / (ufl.exp((-(NV_Ith_S(y, 0) + 45.0)) / 10.0) + 1.0)
        # ydot[2] = AV_alpha_h * (1.0 - NV_Ith_S(y, 2)) - AV_beta_h * NV_Ith_S(y, 2)    

        AV_alpha_m = (-0.1) * (NV_Ith_S(y, 0) + 50.0) / (ufl.exp((-(NV_Ith_S(y, 0) + 50.0)) / 10.0) - 1.0)
        AV_beta_m = 4.0 * ufl.exp((-(NV_Ith_S(y, 0) + 75.0)) / 18.0)
        ydot[1] = AV_alpha_m * (1.0 - NV_Ith_S(y, 1)) - AV_beta_m * NV_Ith_S(y, 1)            
                
        # AV_i_K = self.AC_g_K * pow(NV_Ith_S(y, 3), 4.0) * (NV_Ith_S(y, 0) - self.AC_E_K)        
        # AV_i_Na = self.AC_g_Na * pow(NV_Ith_S(y, 1), 3.0) * NV_Ith_S(y, 2) * (NV_Ith_S(y, 0) - self.AC_E_Na)        
        # AV_i_L = self.AC_g_L * (NV_Ith_S(y, 0) - self.AC_E_L)
        # ydot[0] = -( AV_i_Na + AV_i_K + AV_i_L)  

        return ydot
    
    def u_stiff_coeffs(self, y):
        yinf = [None]*self.size
        tau = [None]*self.size

        # AV_alpha_n = (-0.01) * (NV_Ith_S(y, 0) + 65.0) / (ufl.exp((-(NV_Ith_S(y, 0) + 65.0)) / 10.0) - 1.0)
        # AV_beta_n = 0.125 * ufl.exp((NV_Ith_S(y, 0) + 75.0) / 80.0)
        # tau[3] = 1/(AV_alpha_n+AV_beta_n)
        # yinf[3] = AV_alpha_n*tau[3]    
        
        # AV_alpha_h = 0.07 * ufl.exp((-(NV_Ith_S(y, 0) + 75.0)) / 20.0)
        # AV_beta_h = 1.0 / (ufl.exp((-(NV_Ith_S(y, 0) + 45.0)) / 10.0) + 1.0)
        # tau[2] = 1/(AV_alpha_h+AV_beta_h)
        # yinf[2] = AV_alpha_h*tau[2]    

        AV_alpha_m = (-0.1) * (NV_Ith_S(y, 0) + 50.0) / (ufl.exp((-(NV_Ith_S(y, 0) + 50.0)) / 10.0) - 1.0)
        AV_beta_m = 4.0 * ufl.exp((-(NV_Ith_S(y, 0) + 75.0)) / 18.0)
        tau[1] = 1/(AV_alpha_m+AV_beta_m)
        yinf[1] = AV_alpha_m*tau[1]    

        return yinf,tau
    

    def initial_values(self):

        y0 = [None]*self.size
        y0[0] = -75.0
        y0[1] = 0.05
        y0[2] = 0.595
        y0[3] = 0.317

        return y0
    
    
# Mildly stiff with rho_max ~ 130
class Courtemanche1998(IonicModel):
    def __init__(self):
        super(Courtemanche1998,self).__init__()

        self.size = 21

        self.AC_CMDN_max = 0.05
        self.AC_CSQN_max = 10.0
        self.AC_Km_CMDN = 0.00238
        self.AC_Km_CSQN = 0.8
        self.AC_Km_TRPN = 0.0005
        self.AC_TRPN_max = 0.07
        
        self.AC_I_up_max = 0.005
        self.AC_K_up = 0.00092
        
        self.AC_tau_f_Ca = 2.0
        
        self.AC_Ca_o = 1.8
        self.AC_K_o = 5.4
        self.AC_Na_o = 140.0
        
        self.AC_tau_tr = 180.0
        
        self.AC_Ca_up_max = 15.0
        
        self.AC_K_rel = 30.0
        
        self.AC_tau_u = 8.0
        
        self.AC_g_Ca_L = 0.12375
        
        self.AC_I_NaCa_max = 1600.0
        self.AC_K_mCa = 1.38
        self.AC_K_mNa = 87.5
        self.AC_K_sat = 0.1
        self.AC_Na_Ca_exchanger_current_gamma = 0.35
        
        self.AC_g_B_Ca = 0.001131
        self.AC_g_B_K = 0.0
        self.AC_g_B_Na =  6.74437500000000015e-04
        
        self.AC_g_Na = 7.8
        
        self.AC_V_cell = 20100.0
        self.AC_V_i = self.AC_V_cell * 0.68
        self.AC_V_rel = 0.0048 * self.AC_V_cell
        self.AC_V_up = 0.0552 * self.AC_V_cell
        
        self.AC_Cm = 100.0
        self.AC_F = 96.4867
        self.AC_R = 8.3143
        self.AC_T = 310.0
        
        self.AC_g_Kr =  2.94117649999999994e-02
        
        self.AC_i_CaP_max = 0.275
        
        self.AC_g_Ks =  1.29411759999999987e-01
        
        self.AC_Km_K_o = 1.5
        self.AC_Km_Na_i = 10.0
        self.AC_i_NaK_max =  5.99338739999999981e-01
        self.AC_sigma = 1.0 / 7.0 * (np.exp(self.AC_Na_o / 67.3) - 1.0)
        
        self.AC_g_K1 = 0.09
        
        self.AC_K_Q10 = 3.0
        self.AC_g_to = 0.1652

        #overall stiffness of f is 130 circa
        # sitff indeces:
        # 0 : no, 0.3
        # 1 : yes, 130
        # 2: no, 8
        # 3: no, 0.3
        # 4: no, 3
        # 5: no, 0.1
        # 6: no, 2
        # 7: no, almost 0
        # 8: no, almost 0
        # 9: no, almost 0
        # 10: no, 3
        # 11: no, almost 0
        # 12: no, 0.5
        # 13: no, 0.1
        # 14: no, 0.5
        # 15: no, 5
        # 16: no, 4
        # 17: unknown, spectral radius did not converge. Seems not
        # 18: no, 8
        # 19: no, almost 0
        # 20: no, almost 0
        self.f_nonstiff_args = list(range(self.size))
        self.f_stiff_args = [0,1]
        self.f_expl_args = list(range(self.size))

    def f(self, y):

        ydot = [None]*self.size

        # Linear (in the gating variables) terms
        
        #/* Ca_release_current_from_JSR_w_gate */
        AV_tau_w = ufl.conditional(ufl.lt(  ufl.algebra.Abs(NV_Ith_S(y, 0) - 7.9), 1e-10), \
                                    6.0 * 0.2 / 1.3, \
                                    6.0 * (1.0 - ufl.exp((-(NV_Ith_S(y, 0) - 7.9)) / 5.0)) / ((1.0 + 0.3 * ufl.exp((-(NV_Ith_S(y, 0) - 7.9)) / 5.0)) * 1.0 * (NV_Ith_S(y, 0) - 7.9)))
        AV_w_infinity = 1.0 - pow(1.0 + ufl.exp((-(NV_Ith_S(y, 0) - 40.0)) / 17.0), (-1.0))
        ydot[15] = (AV_w_infinity - NV_Ith_S(y, 15)) / AV_tau_w                
        
        #/* L_type_Ca_channel_d_gate */
        AV_d_infinity = pow(1.0 + ufl.exp((NV_Ith_S(y, 0) + 10.0) / (-8.0)), (-1.0))
        AV_tau_d = ufl.conditional(ufl.lt(  ufl.algebra.Abs(NV_Ith_S(y, 0) + 10.0), 1e-10), 4.579 / (1.0 + ufl.exp((NV_Ith_S(y, 0) + 10.0) / (-6.24))), (1.0 - ufl.exp((NV_Ith_S(y, 0) + 10.0) / (-6.24))) / (0.035 * (NV_Ith_S(y, 0) + 10.0) * (1.0 + ufl.exp((NV_Ith_S(y, 0) + 10.0) / (-6.24)))))
        ydot[10] = (AV_d_infinity - NV_Ith_S(y, 10)) / AV_tau_d
                
        #/* L_type_Ca_channel_f_gate */
        AV_f_infinity = ufl.exp((-(NV_Ith_S(y, 0) + 28.0)) / 6.9) / (1.0 + ufl.exp((-(NV_Ith_S(y, 0) + 28.0)) / 6.9))
        AV_tau_f = 9.0 * pow(0.0197 * ufl.exp((-pow(0.0337, 2.0)) * pow(NV_Ith_S(y, 0) + 10.0, 2.0)) + 0.02, (-1.0))
        ydot[11] = (AV_f_infinity - NV_Ith_S(y, 11)) / AV_tau_f        
        
        #/* fast_sodium_current_h_gate */
        AV_alpha_h = ufl.conditional(ufl.lt(NV_Ith_S(y, 0), (-40.0)), 0.135 * ufl.exp((NV_Ith_S(y, 0) + 80.0) / (-6.8)), 0.0)
        AV_beta_h = ufl.conditional(ufl.lt(NV_Ith_S(y, 0) , (-40.0)),3.56 * ufl.exp(0.079 * NV_Ith_S(y, 0)) + 310000.0 * ufl.exp(0.35 * NV_Ith_S(y, 0)), 1.0 / (0.13 * (1.0 + ufl.exp((NV_Ith_S(y, 0) + 10.66) / (-11.1)))))
        AV_h_inf = AV_alpha_h / (AV_alpha_h + AV_beta_h)
        AV_tau_h = 1.0 / (AV_alpha_h + AV_beta_h)
        ydot[2] = (AV_h_inf - NV_Ith_S(y, 2)) / AV_tau_h
        
        #/* fast_sodium_current_j_gate */
        AV_alpha_j = ufl.conditional(ufl.lt(NV_Ith_S(y, 0) , (-40.0)),((-127140.0) * ufl.exp(0.2444 * NV_Ith_S(y, 0)) - 3.474e-05 * ufl.exp((-0.04391) * NV_Ith_S(y, 0))) * (NV_Ith_S(y, 0) + 37.78) / (1.0 + ufl.exp(0.311 * (NV_Ith_S(y, 0) + 79.23))), 0.0)
        AV_beta_j = ufl.conditional(ufl.lt(NV_Ith_S(y, 0), (-40.0)),0.1212 * ufl.exp((-0.01052) * NV_Ith_S(y, 0)) / (1.0 + ufl.exp((-0.1378) * (NV_Ith_S(y, 0) + 40.14))), 0.3 * ufl.exp((-2.535e-07) * NV_Ith_S(y, 0)) / (1.0 + ufl.exp((-0.1) * (NV_Ith_S(y, 0) + 32.0))))
        AV_j_inf = AV_alpha_j / (AV_alpha_j + AV_beta_j)
        AV_tau_j = 1.0 / (AV_alpha_j + AV_beta_j)
        ydot[3] = (AV_j_inf - NV_Ith_S(y, 3)) / AV_tau_j
        
        #/* fast_sodium_current_m_gate */
        AV_alpha_m = ufl.conditional(ufl.eq(NV_Ith_S(y, 0), (-47.13)),3.2, 0.32 * (NV_Ith_S(y, 0) + 47.13) / (1.0 - ufl.exp((-0.1) * (NV_Ith_S(y, 0) + 47.13))))
        AV_beta_m = 0.08 * ufl.exp((-NV_Ith_S(y, 0)) / 11.0)
        AV_m_inf = AV_alpha_m / (AV_alpha_m + AV_beta_m)
        AV_tau_m = 1.0 / (AV_alpha_m + AV_beta_m)
        ydot[1] = (AV_m_inf - NV_Ith_S(y, 1)) / AV_tau_m
        
        #/* rapid_delayed_rectifier_K_current_xr_gate */
        AV_alpha_xr = ufl.conditional(ufl.lt(  ufl.algebra.Abs(NV_Ith_S(y, 0) + 14.1) , 1e-10),0.0015, 0.0003 * (NV_Ith_S(y, 0) + 14.1) / (1.0 - ufl.exp((NV_Ith_S(y, 0) + 14.1) / (-5.0))))
        AV_beta_xr = ufl.conditional(ufl.lt(  ufl.algebra.Abs(NV_Ith_S(y, 0) - 3.3328), 1e-10),3.78361180000000004e-04,  7.38980000000000030e-05 * (NV_Ith_S(y, 0) - 3.3328) / (ufl.exp((NV_Ith_S(y, 0) - 3.3328) / 5.1237) - 1.0))
        AV_xr_infinity = pow(1.0 + ufl.exp((NV_Ith_S(y, 0) + 14.1) / (-6.5)), (-1.0))
        AV_tau_xr = pow(AV_alpha_xr + AV_beta_xr, (-1.0))
        ydot[8] = (AV_xr_infinity - NV_Ith_S(y, 8)) / AV_tau_xr
        
        #/* slow_delayed_rectifier_K_current_xs_gate */
        AV_alpha_xs = ufl.conditional(ufl.lt(  ufl.algebra.Abs(NV_Ith_S(y, 0) - 19.9) , 1e-10),0.00068, 4e-05 * (NV_Ith_S(y, 0) - 19.9) / (1.0 - ufl.exp((NV_Ith_S(y, 0) - 19.9) / (-17.0))))
        AV_beta_xs = ufl.conditional(ufl.lt(  ufl.algebra.Abs(NV_Ith_S(y, 0) - 19.9) , 1e-10),0.000315, 3.5e-05 * (NV_Ith_S(y, 0) - 19.9) / (ufl.exp((NV_Ith_S(y, 0) - 19.9) / 9.0) - 1.0))
        AV_xs_infinity = pow(1.0 + ufl.exp((NV_Ith_S(y, 0) - 19.9) / (-12.7)), (-0.5))
        AV_tau_xs = 0.5 * pow(AV_alpha_xs + AV_beta_xs, (-1.0))
        ydot[9] = (AV_xs_infinity - NV_Ith_S(y, 9)) / AV_tau_xs

        #/* transient_outward_K_current_oa_gate */
        AV_alpha_oa = 0.65 * pow(ufl.exp((NV_Ith_S(y, 0) - (-10.0)) / (-8.5)) + ufl.exp((NV_Ith_S(y, 0) - (-10.0) - 40.0) / (-59.0)), (-1.0))
        AV_beta_oa = 0.65 * pow(2.5 + ufl.exp((NV_Ith_S(y, 0) - (-10.0) + 72.0) / 17.0), (-1.0))
        AV_oa_infinity = pow(1.0 + ufl.exp((NV_Ith_S(y, 0) - (-10.0) + 10.47) / (-17.54)), (-1.0))
        AV_tau_oa = pow(AV_alpha_oa + AV_beta_oa, (-1.0)) / self.AC_K_Q10
        ydot[4] = (AV_oa_infinity - NV_Ith_S(y, 4)) / AV_tau_oa        

        #/* transient_outward_K_current_oi_gate */
        AV_alpha_oi = pow(18.53 + 1.0 * ufl.exp((NV_Ith_S(y, 0) - (-10.0) + 103.7) / 10.95), (-1.0))
        AV_beta_oi = pow(35.56 + 1.0 * ufl.exp((NV_Ith_S(y, 0) - (-10.0) - 8.74) / (-7.44)), (-1.0))
        AV_oi_infinity = pow(1.0 + ufl.exp((NV_Ith_S(y, 0) - (-10.0) + 33.1) / 5.3), (-1.0))
        AV_tau_oi = pow(AV_alpha_oi + AV_beta_oi, (-1.0)) / self.AC_K_Q10
        ydot[5] = (AV_oi_infinity - NV_Ith_S(y, 5)) / AV_tau_oi

        #/* ultrarapid_delayed_rectifier_K_current_ua_gate */
        AV_alpha_ua = 0.65 * pow(ufl.exp((NV_Ith_S(y, 0) - (-10.0)) / (-8.5)) + ufl.exp((NV_Ith_S(y, 0) - (-10.0) - 40.0) / (-59.0)), (-1.0))
        AV_beta_ua = 0.65 * pow(2.5 + ufl.exp((NV_Ith_S(y, 0) - (-10.0) + 72.0) / 17.0), (-1.0))
        AV_ua_infinity = pow(1.0 + ufl.exp((NV_Ith_S(y, 0) - (-10.0) + 20.3) / (-9.6)), (-1.0))
        AV_tau_ua = pow(AV_alpha_ua + AV_beta_ua, (-1.0)) / self.AC_K_Q10
        ydot[6] = (AV_ua_infinity - NV_Ith_S(y, 6)) / AV_tau_ua

        #/* ultrarapid_delayed_rectifier_K_current_ui_gate */
        AV_alpha_ui = pow(21.0 + 1.0 * ufl.exp((NV_Ith_S(y, 0) - (-10.0) - 195.0) / (-28.0)), (-1.0))
        AV_beta_ui = 1.0 / ufl.exp((NV_Ith_S(y, 0) - (-10.0) - 168.0) / (-16.0))
        AV_ui_infinity = pow(1.0 + ufl.exp((NV_Ith_S(y, 0) - (-10.0) - 109.45) / 27.48), (-1.0))
        AV_tau_ui = pow(AV_alpha_ui + AV_beta_ui, (-1.0)) / self.AC_K_Q10
        ydot[7] = (AV_ui_infinity - NV_Ith_S(y, 7)) / AV_tau_ui


        # Non Linear (in the gating variables) terms

        #/* L_type_Ca_channel_f_Ca_gate */
        AV_f_Ca_infinity = pow(1.0 + NV_Ith_S(y, 17) / 0.00035, (-1.0))
        ydot[12] = (AV_f_Ca_infinity - NV_Ith_S(y, 12)) / self.AC_tau_f_Ca

        #/* transfer_current_from_NSR_to_JSR */
        AV_i_tr = (NV_Ith_S(y, 20) - NV_Ith_S(y, 19)) / self.AC_tau_tr
        
        #/* Ca_leak_current_by_the_NSR */
        AV_i_up_leak = self.AC_I_up_max * NV_Ith_S(y, 20) / self.AC_Ca_up_max
        
        #/* Ca_release_current_from_JSR */
        AV_i_rel = self.AC_K_rel * pow(NV_Ith_S(y, 13), 2.0) * NV_Ith_S(y, 14) * NV_Ith_S(y, 15) * (NV_Ith_S(y, 19) - NV_Ith_S(y, 17))
        
        #/* intracellular_ion_concentrations */
        ydot[19] = (AV_i_tr - AV_i_rel) * pow(1.0 + self.AC_CSQN_max * self.AC_Km_CSQN / pow(NV_Ith_S(y, 19) + self.AC_Km_CSQN, 2.0), (-1.0))

        #/* Ca_uptake_current_by_the_NSR */
        AV_i_up = self.AC_I_up_max / (1.0 + self.AC_K_up / NV_Ith_S(y, 17))
        ydot[20] = AV_i_up - (AV_i_up_leak + AV_i_tr * self.AC_V_rel / self.AC_V_up)
        
        #/* sarcolemmal_calcium_pump_current */
        AV_i_CaP = self.AC_Cm * self.AC_i_CaP_max * NV_Ith_S(y, 17) / (0.0005 + NV_Ith_S(y, 17))
        
        #/* sodium_potassium_pump */
        AV_f_NaK = pow(1.0 + 0.1245 * ufl.exp((-0.1) * self.AC_F * NV_Ith_S(y, 0) / (self.AC_R * self.AC_T)) + 0.0365 * self.AC_sigma * ufl.exp((-self.AC_F) * NV_Ith_S(y, 0) / (self.AC_R * self.AC_T)), (-1.0))
        AV_i_NaK = self.AC_Cm * self.AC_i_NaK_max * AV_f_NaK * 1.0 / (1.0 + pow(self.AC_Km_Na_i / NV_Ith_S(y, 16), 1.5)) * self.AC_K_o / (self.AC_K_o + self.AC_Km_K_o)
        
        #/* time_independent_potassium_current */
        AV_E_K = self.AC_R * self.AC_T / self.AC_F * ufl.ln(self.AC_K_o / NV_Ith_S(y, 18))
        AV_i_K1 = self.AC_Cm * self.AC_g_K1 * (NV_Ith_S(y, 0) - AV_E_K) / (1.0 + ufl.exp(0.07 * (NV_Ith_S(y, 0) + 80.0)))
        
        #/* transient_outward_K_current */
        AV_i_to = self.AC_Cm * self.AC_g_to * pow(NV_Ith_S(y, 4), 3.0) * NV_Ith_S(y, 5) * (NV_Ith_S(y, 0) - AV_E_K)                                
        
        #/* ultrarapid_delayed_rectifier_K_current */
        AV_g_Kur = 0.005 + 0.05 / (1.0 + ufl.exp((NV_Ith_S(y, 0) - 15.0) / (-13.0)))
        AV_i_Kur = self.AC_Cm * AV_g_Kur * pow(NV_Ith_S(y, 6), 3.0) * NV_Ith_S(y, 7) * (NV_Ith_S(y, 0) - AV_E_K)                                
        
        #/* *remaining* */
        AV_i_Ca_L = self.AC_Cm * self.AC_g_Ca_L * NV_Ith_S(y, 10) * NV_Ith_S(y, 11) * NV_Ith_S(y, 12) * (NV_Ith_S(y, 0) - 65.0)
        AV_i_NaCa = self.AC_Cm * self.AC_I_NaCa_max * (ufl.exp(self.AC_Na_Ca_exchanger_current_gamma * self.AC_F * NV_Ith_S(y, 0) / (self.AC_R * self.AC_T)) * pow(NV_Ith_S(y, 16), 3.0) * self.AC_Ca_o - ufl.exp((self.AC_Na_Ca_exchanger_current_gamma - 1.0) * self.AC_F * NV_Ith_S(y, 0) / (self.AC_R * self.AC_T)) * pow(self.AC_Na_o, 3.0) * NV_Ith_S(y, 17)) / ((pow(self.AC_K_mNa, 3.0) + pow(self.AC_Na_o, 3.0)) * (self.AC_K_mCa + self.AC_Ca_o) * (1.0 + self.AC_K_sat * ufl.exp((self.AC_Na_Ca_exchanger_current_gamma - 1.0) * NV_Ith_S(y, 0) * self.AC_F / (self.AC_R * self.AC_T))))
        AV_E_Ca = self.AC_R * self.AC_T / (2.0 * self.AC_F) * ufl.ln(self.AC_Ca_o / NV_Ith_S(y, 17))
        AV_i_B_K = self.AC_Cm * self.AC_g_B_K * (NV_Ith_S(y, 0) - AV_E_K)
        AV_E_Na = self.AC_R * self.AC_T / self.AC_F * ufl.ln(self.AC_Na_o / NV_Ith_S(y, 16))
        AV_i_Kr = self.AC_Cm * self.AC_g_Kr * NV_Ith_S(y, 8) * (NV_Ith_S(y, 0) - AV_E_K) / (1.0 + ufl.exp((NV_Ith_S(y, 0) + 15.0) / 22.4))
        AV_i_Ks = self.AC_Cm * self.AC_g_Ks * pow(NV_Ith_S(y, 9), 2.0) * (NV_Ith_S(y, 0) - AV_E_K)
        AV_Fn = 1000.0 * (1e-15 * self.AC_V_rel * AV_i_rel - 1e-15 / (2.0 * self.AC_F) * (0.5 * AV_i_Ca_L - 0.2 * AV_i_NaCa))
        AV_i_B_Ca = self.AC_Cm * self.AC_g_B_Ca * (NV_Ith_S(y, 0) - AV_E_Ca)
        AV_i_B_Na = self.AC_Cm * self.AC_g_B_Na * (NV_Ith_S(y, 0) - AV_E_Na)
        AV_i_Na = self.AC_Cm * self.AC_g_Na * pow(NV_Ith_S(y, 1), 3.0) * NV_Ith_S(y, 2) * NV_Ith_S(y, 3) * (NV_Ith_S(y, 0) - AV_E_Na)
        ydot[18] = (2.0 * AV_i_NaK - (AV_i_K1 + AV_i_to + AV_i_Kur + AV_i_Kr + AV_i_Ks + AV_i_B_K)) / (self.AC_V_i * self.AC_F)
        AV_u_infinity = pow(1.0 + ufl.exp((-(AV_Fn -  3.41749999999999983e-13)) / 1.367e-15), (-1.0))
        AV_tau_v = 1.91 + 2.09 * pow(1.0 + ufl.exp((-(AV_Fn -  3.41749999999999983e-13)) / 1.367e-15), (-1.0))
        AV_v_infinity = 1.0 - pow(1.0 + ufl.exp((-(AV_Fn - 6.835e-14)) / 1.367e-15), (-1.0))
        ydot[16] = ((-3.0) * AV_i_NaK - (3.0 * AV_i_NaCa + AV_i_B_Na + AV_i_Na)) / (self.AC_V_i * self.AC_F)
        
        ydot[0] = (-(AV_i_Na + AV_i_K1 + AV_i_to + AV_i_Kur + AV_i_Kr + AV_i_Ks + AV_i_B_Na + AV_i_B_Ca + AV_i_NaK + AV_i_CaP + AV_i_NaCa + AV_i_Ca_L)) / self.AC_Cm
        ydot[13] = (AV_u_infinity - NV_Ith_S(y, 13)) / self.AC_tau_u
        ydot[14] = (AV_v_infinity - NV_Ith_S(y, 14)) / AV_tau_v
        
        AV_B1 = (2.0 * AV_i_NaCa - (AV_i_CaP + AV_i_Ca_L + AV_i_B_Ca)) / (2.0 * self.AC_V_i * self.AC_F) + (self.AC_V_up * (AV_i_up_leak - AV_i_up) + AV_i_rel * self.AC_V_rel) / self.AC_V_i
        AV_B2 = 1.0 + self.AC_TRPN_max * self.AC_Km_TRPN / pow(NV_Ith_S(y, 17) + self.AC_Km_TRPN, 2.0) + self.AC_CMDN_max * self.AC_Km_CMDN / pow(NV_Ith_S(y, 17) + self.AC_Km_CMDN, 2.0)
        ydot[17] = AV_B1 / AV_B2

        return ydot
    

    def f_stiff(self, y):

        ydot = [None]*self.size

        #/* fast_sodium_current_m_gate */
        AV_alpha_m = ufl.conditional(ufl.eq(NV_Ith_S(y, 0), (-47.13)),3.2, 0.32 * (NV_Ith_S(y, 0) + 47.13) / (1.0 - ufl.exp((-0.1) * (NV_Ith_S(y, 0) + 47.13))))
        AV_beta_m = 0.08 * ufl.exp((-NV_Ith_S(y, 0)) / 11.0)
        AV_m_inf = AV_alpha_m / (AV_alpha_m + AV_beta_m)
        AV_tau_m = 1.0 / (AV_alpha_m + AV_beta_m)
        ydot[1] = (AV_m_inf - NV_Ith_S(y, 1)) / AV_tau_m

        return ydot
    
    def u_stiff_coeffs(self, y):
        yinf = [None]*self.size
        tau = [None]*self.size
        
         #/* fast_sodium_current_m_gate */
        AV_alpha_m = ufl.conditional(ufl.eq(NV_Ith_S(y, 0), (-47.13)),3.2, 0.32 * (NV_Ith_S(y, 0) + 47.13) / (1.0 - ufl.exp((-0.1) * (NV_Ith_S(y, 0) + 47.13))))
        AV_beta_m = 0.08 * ufl.exp((-NV_Ith_S(y, 0)) / 11.0)
        yinf[1] = AV_alpha_m / (AV_alpha_m + AV_beta_m)
        tau[1] = 1.0 / (AV_alpha_m + AV_beta_m)

        return yinf,tau
    
    def f_nonstiff(self, y):

        ydot = [None]*self.size

        # Linear (in the gating variables) terms
        
        #/* Ca_release_current_from_JSR_w_gate */
        AV_tau_w = ufl.conditional(ufl.lt(  ufl.algebra.Abs(NV_Ith_S(y, 0) - 7.9), 1e-10), \
                                    6.0 * 0.2 / 1.3, \
                                    6.0 * (1.0 - ufl.exp((-(NV_Ith_S(y, 0) - 7.9)) / 5.0)) / ((1.0 + 0.3 * ufl.exp((-(NV_Ith_S(y, 0) - 7.9)) / 5.0)) * 1.0 * (NV_Ith_S(y, 0) - 7.9)))
        AV_w_infinity = 1.0 - pow(1.0 + ufl.exp((-(NV_Ith_S(y, 0) - 40.0)) / 17.0), (-1.0))
        ydot[15] = (AV_w_infinity - NV_Ith_S(y, 15)) / AV_tau_w                
        
        #/* L_type_Ca_channel_d_gate */
        AV_d_infinity = pow(1.0 + ufl.exp((NV_Ith_S(y, 0) + 10.0) / (-8.0)), (-1.0))
        AV_tau_d = ufl.conditional(ufl.lt(  ufl.algebra.Abs(NV_Ith_S(y, 0) + 10.0), 1e-10), 4.579 / (1.0 + ufl.exp((NV_Ith_S(y, 0) + 10.0) / (-6.24))), (1.0 - ufl.exp((NV_Ith_S(y, 0) + 10.0) / (-6.24))) / (0.035 * (NV_Ith_S(y, 0) + 10.0) * (1.0 + ufl.exp((NV_Ith_S(y, 0) + 10.0) / (-6.24)))))
        ydot[10] = (AV_d_infinity - NV_Ith_S(y, 10)) / AV_tau_d
                
        #/* L_type_Ca_channel_f_gate */
        AV_f_infinity = ufl.exp((-(NV_Ith_S(y, 0) + 28.0)) / 6.9) / (1.0 + ufl.exp((-(NV_Ith_S(y, 0) + 28.0)) / 6.9))
        AV_tau_f = 9.0 * pow(0.0197 * ufl.exp((-pow(0.0337, 2.0)) * pow(NV_Ith_S(y, 0) + 10.0, 2.0)) + 0.02, (-1.0))
        ydot[11] = (AV_f_infinity - NV_Ith_S(y, 11)) / AV_tau_f        
        
        #/* fast_sodium_current_h_gate */
        AV_alpha_h = ufl.conditional(ufl.lt(NV_Ith_S(y, 0), (-40.0)), 0.135 * ufl.exp((NV_Ith_S(y, 0) + 80.0) / (-6.8)), 0.0)
        AV_beta_h = ufl.conditional(ufl.lt(NV_Ith_S(y, 0) , (-40.0)),3.56 * ufl.exp(0.079 * NV_Ith_S(y, 0)) + 310000.0 * ufl.exp(0.35 * NV_Ith_S(y, 0)), 1.0 / (0.13 * (1.0 + ufl.exp((NV_Ith_S(y, 0) + 10.66) / (-11.1)))))
        AV_h_inf = AV_alpha_h / (AV_alpha_h + AV_beta_h)
        AV_tau_h = 1.0 / (AV_alpha_h + AV_beta_h)
        ydot[2] = (AV_h_inf - NV_Ith_S(y, 2)) / AV_tau_h
        
        #/* fast_sodium_current_j_gate */
        AV_alpha_j = ufl.conditional(ufl.lt(NV_Ith_S(y, 0) , (-40.0)),((-127140.0) * ufl.exp(0.2444 * NV_Ith_S(y, 0)) - 3.474e-05 * ufl.exp((-0.04391) * NV_Ith_S(y, 0))) * (NV_Ith_S(y, 0) + 37.78) / (1.0 + ufl.exp(0.311 * (NV_Ith_S(y, 0) + 79.23))), 0.0)
        AV_beta_j = ufl.conditional(ufl.lt(NV_Ith_S(y, 0), (-40.0)),0.1212 * ufl.exp((-0.01052) * NV_Ith_S(y, 0)) / (1.0 + ufl.exp((-0.1378) * (NV_Ith_S(y, 0) + 40.14))), 0.3 * ufl.exp((-2.535e-07) * NV_Ith_S(y, 0)) / (1.0 + ufl.exp((-0.1) * (NV_Ith_S(y, 0) + 32.0))))
        AV_j_inf = AV_alpha_j / (AV_alpha_j + AV_beta_j)
        AV_tau_j = 1.0 / (AV_alpha_j + AV_beta_j)
        ydot[3] = (AV_j_inf - NV_Ith_S(y, 3)) / AV_tau_j
        
        #/* fast_sodium_current_m_gate */        
        # this is computed in self.f_stiff        
        
        #/* rapid_delayed_rectifier_K_current_xr_gate */
        AV_alpha_xr = ufl.conditional(ufl.lt(  ufl.algebra.Abs(NV_Ith_S(y, 0) + 14.1) , 1e-10),0.0015, 0.0003 * (NV_Ith_S(y, 0) + 14.1) / (1.0 - ufl.exp((NV_Ith_S(y, 0) + 14.1) / (-5.0))))
        AV_beta_xr = ufl.conditional(ufl.lt(  ufl.algebra.Abs(NV_Ith_S(y, 0) - 3.3328), 1e-10),3.78361180000000004e-04,  7.38980000000000030e-05 * (NV_Ith_S(y, 0) - 3.3328) / (ufl.exp((NV_Ith_S(y, 0) - 3.3328) / 5.1237) - 1.0))
        AV_xr_infinity = pow(1.0 + ufl.exp((NV_Ith_S(y, 0) + 14.1) / (-6.5)), (-1.0))
        AV_tau_xr = pow(AV_alpha_xr + AV_beta_xr, (-1.0))
        ydot[8] = (AV_xr_infinity - NV_Ith_S(y, 8)) / AV_tau_xr
        
        #/* slow_delayed_rectifier_K_current_xs_gate */
        AV_alpha_xs = ufl.conditional(ufl.lt(  ufl.algebra.Abs(NV_Ith_S(y, 0) - 19.9) , 1e-10),0.00068, 4e-05 * (NV_Ith_S(y, 0) - 19.9) / (1.0 - ufl.exp((NV_Ith_S(y, 0) - 19.9) / (-17.0))))
        AV_beta_xs = ufl.conditional(ufl.lt(  ufl.algebra.Abs(NV_Ith_S(y, 0) - 19.9) , 1e-10),0.000315, 3.5e-05 * (NV_Ith_S(y, 0) - 19.9) / (ufl.exp((NV_Ith_S(y, 0) - 19.9) / 9.0) - 1.0))
        AV_xs_infinity = pow(1.0 + ufl.exp((NV_Ith_S(y, 0) - 19.9) / (-12.7)), (-0.5))
        AV_tau_xs = 0.5 * pow(AV_alpha_xs + AV_beta_xs, (-1.0))
        ydot[9] = (AV_xs_infinity - NV_Ith_S(y, 9)) / AV_tau_xs

        #/* transient_outward_K_current_oa_gate */
        AV_alpha_oa = 0.65 * pow(ufl.exp((NV_Ith_S(y, 0) - (-10.0)) / (-8.5)) + ufl.exp((NV_Ith_S(y, 0) - (-10.0) - 40.0) / (-59.0)), (-1.0))
        AV_beta_oa = 0.65 * pow(2.5 + ufl.exp((NV_Ith_S(y, 0) - (-10.0) + 72.0) / 17.0), (-1.0))
        AV_oa_infinity = pow(1.0 + ufl.exp((NV_Ith_S(y, 0) - (-10.0) + 10.47) / (-17.54)), (-1.0))
        AV_tau_oa = pow(AV_alpha_oa + AV_beta_oa, (-1.0)) / self.AC_K_Q10
        ydot[4] = (AV_oa_infinity - NV_Ith_S(y, 4)) / AV_tau_oa        

        #/* transient_outward_K_current_oi_gate */
        AV_alpha_oi = pow(18.53 + 1.0 * ufl.exp((NV_Ith_S(y, 0) - (-10.0) + 103.7) / 10.95), (-1.0))
        AV_beta_oi = pow(35.56 + 1.0 * ufl.exp((NV_Ith_S(y, 0) - (-10.0) - 8.74) / (-7.44)), (-1.0))
        AV_oi_infinity = pow(1.0 + ufl.exp((NV_Ith_S(y, 0) - (-10.0) + 33.1) / 5.3), (-1.0))
        AV_tau_oi = pow(AV_alpha_oi + AV_beta_oi, (-1.0)) / self.AC_K_Q10
        ydot[5] = (AV_oi_infinity - NV_Ith_S(y, 5)) / AV_tau_oi

        #/* ultrarapid_delayed_rectifier_K_current_ua_gate */
        AV_alpha_ua = 0.65 * pow(ufl.exp((NV_Ith_S(y, 0) - (-10.0)) / (-8.5)) + ufl.exp((NV_Ith_S(y, 0) - (-10.0) - 40.0) / (-59.0)), (-1.0))
        AV_beta_ua = 0.65 * pow(2.5 + ufl.exp((NV_Ith_S(y, 0) - (-10.0) + 72.0) / 17.0), (-1.0))
        AV_ua_infinity = pow(1.0 + ufl.exp((NV_Ith_S(y, 0) - (-10.0) + 20.3) / (-9.6)), (-1.0))
        AV_tau_ua = pow(AV_alpha_ua + AV_beta_ua, (-1.0)) / self.AC_K_Q10
        ydot[6] = (AV_ua_infinity - NV_Ith_S(y, 6)) / AV_tau_ua

        #/* ultrarapid_delayed_rectifier_K_current_ui_gate */
        AV_alpha_ui = pow(21.0 + 1.0 * ufl.exp((NV_Ith_S(y, 0) - (-10.0) - 195.0) / (-28.0)), (-1.0))
        AV_beta_ui = 1.0 / ufl.exp((NV_Ith_S(y, 0) - (-10.0) - 168.0) / (-16.0))
        AV_ui_infinity = pow(1.0 + ufl.exp((NV_Ith_S(y, 0) - (-10.0) - 109.45) / 27.48), (-1.0))
        AV_tau_ui = pow(AV_alpha_ui + AV_beta_ui, (-1.0)) / self.AC_K_Q10
        ydot[7] = (AV_ui_infinity - NV_Ith_S(y, 7)) / AV_tau_ui


        # Non Linear (in the gating variables) terms

        #/* L_type_Ca_channel_f_Ca_gate */
        AV_f_Ca_infinity = pow(1.0 + NV_Ith_S(y, 17) / 0.00035, (-1.0))
        ydot[12] = (AV_f_Ca_infinity - NV_Ith_S(y, 12)) / self.AC_tau_f_Ca

        #/* transfer_current_from_NSR_to_JSR */
        AV_i_tr = (NV_Ith_S(y, 20) - NV_Ith_S(y, 19)) / self.AC_tau_tr
        
        #/* Ca_leak_current_by_the_NSR */
        AV_i_up_leak = self.AC_I_up_max * NV_Ith_S(y, 20) / self.AC_Ca_up_max
        
        #/* Ca_release_current_from_JSR */
        AV_i_rel = self.AC_K_rel * pow(NV_Ith_S(y, 13), 2.0) * NV_Ith_S(y, 14) * NV_Ith_S(y, 15) * (NV_Ith_S(y, 19) - NV_Ith_S(y, 17))
        
        #/* intracellular_ion_concentrations */
        ydot[19] = (AV_i_tr - AV_i_rel) * pow(1.0 + self.AC_CSQN_max * self.AC_Km_CSQN / pow(NV_Ith_S(y, 19) + self.AC_Km_CSQN, 2.0), (-1.0))

        #/* Ca_uptake_current_by_the_NSR */
        AV_i_up = self.AC_I_up_max / (1.0 + self.AC_K_up / NV_Ith_S(y, 17))
        ydot[20] = AV_i_up - (AV_i_up_leak + AV_i_tr * self.AC_V_rel / self.AC_V_up)
        
        #/* sarcolemmal_calcium_pump_current */
        AV_i_CaP = self.AC_Cm * self.AC_i_CaP_max * NV_Ith_S(y, 17) / (0.0005 + NV_Ith_S(y, 17))
        
        #/* sodium_potassium_pump */
        AV_f_NaK = pow(1.0 + 0.1245 * ufl.exp((-0.1) * self.AC_F * NV_Ith_S(y, 0) / (self.AC_R * self.AC_T)) + 0.0365 * self.AC_sigma * ufl.exp((-self.AC_F) * NV_Ith_S(y, 0) / (self.AC_R * self.AC_T)), (-1.0))
        AV_i_NaK = self.AC_Cm * self.AC_i_NaK_max * AV_f_NaK * 1.0 / (1.0 + pow(self.AC_Km_Na_i / NV_Ith_S(y, 16), 1.5)) * self.AC_K_o / (self.AC_K_o + self.AC_Km_K_o)
        
        #/* time_independent_potassium_current */
        AV_E_K = self.AC_R * self.AC_T / self.AC_F * ufl.ln(self.AC_K_o / NV_Ith_S(y, 18))
        AV_i_K1 = self.AC_Cm * self.AC_g_K1 * (NV_Ith_S(y, 0) - AV_E_K) / (1.0 + ufl.exp(0.07 * (NV_Ith_S(y, 0) + 80.0)))
        
        #/* transient_outward_K_current */
        AV_i_to = self.AC_Cm * self.AC_g_to * pow(NV_Ith_S(y, 4), 3.0) * NV_Ith_S(y, 5) * (NV_Ith_S(y, 0) - AV_E_K)                                
        
        #/* ultrarapid_delayed_rectifier_K_current */
        AV_g_Kur = 0.005 + 0.05 / (1.0 + ufl.exp((NV_Ith_S(y, 0) - 15.0) / (-13.0)))
        AV_i_Kur = self.AC_Cm * AV_g_Kur * pow(NV_Ith_S(y, 6), 3.0) * NV_Ith_S(y, 7) * (NV_Ith_S(y, 0) - AV_E_K)                                
        
        #/* *remaining* */
        AV_i_Ca_L = self.AC_Cm * self.AC_g_Ca_L * NV_Ith_S(y, 10) * NV_Ith_S(y, 11) * NV_Ith_S(y, 12) * (NV_Ith_S(y, 0) - 65.0)
        AV_i_NaCa = self.AC_Cm * self.AC_I_NaCa_max * (ufl.exp(self.AC_Na_Ca_exchanger_current_gamma * self.AC_F * NV_Ith_S(y, 0) / (self.AC_R * self.AC_T)) * pow(NV_Ith_S(y, 16), 3.0) * self.AC_Ca_o - ufl.exp((self.AC_Na_Ca_exchanger_current_gamma - 1.0) * self.AC_F * NV_Ith_S(y, 0) / (self.AC_R * self.AC_T)) * pow(self.AC_Na_o, 3.0) * NV_Ith_S(y, 17)) / ((pow(self.AC_K_mNa, 3.0) + pow(self.AC_Na_o, 3.0)) * (self.AC_K_mCa + self.AC_Ca_o) * (1.0 + self.AC_K_sat * ufl.exp((self.AC_Na_Ca_exchanger_current_gamma - 1.0) * NV_Ith_S(y, 0) * self.AC_F / (self.AC_R * self.AC_T))))
        AV_E_Ca = self.AC_R * self.AC_T / (2.0 * self.AC_F) * ufl.ln(self.AC_Ca_o / NV_Ith_S(y, 17))
        AV_i_B_K = self.AC_Cm * self.AC_g_B_K * (NV_Ith_S(y, 0) - AV_E_K)
        AV_E_Na = self.AC_R * self.AC_T / self.AC_F * ufl.ln(self.AC_Na_o / NV_Ith_S(y, 16))
        AV_i_Kr = self.AC_Cm * self.AC_g_Kr * NV_Ith_S(y, 8) * (NV_Ith_S(y, 0) - AV_E_K) / (1.0 + ufl.exp((NV_Ith_S(y, 0) + 15.0) / 22.4))
        AV_i_Ks = self.AC_Cm * self.AC_g_Ks * pow(NV_Ith_S(y, 9), 2.0) * (NV_Ith_S(y, 0) - AV_E_K)
        AV_Fn = 1000.0 * (1e-15 * self.AC_V_rel * AV_i_rel - 1e-15 / (2.0 * self.AC_F) * (0.5 * AV_i_Ca_L - 0.2 * AV_i_NaCa))
        AV_i_B_Ca = self.AC_Cm * self.AC_g_B_Ca * (NV_Ith_S(y, 0) - AV_E_Ca)
        AV_i_B_Na = self.AC_Cm * self.AC_g_B_Na * (NV_Ith_S(y, 0) - AV_E_Na)
        AV_i_Na = self.AC_Cm * self.AC_g_Na * pow(NV_Ith_S(y, 1), 3.0) * NV_Ith_S(y, 2) * NV_Ith_S(y, 3) * (NV_Ith_S(y, 0) - AV_E_Na)
        ydot[18] = (2.0 * AV_i_NaK - (AV_i_K1 + AV_i_to + AV_i_Kur + AV_i_Kr + AV_i_Ks + AV_i_B_K)) / (self.AC_V_i * self.AC_F)
        AV_u_infinity = pow(1.0 + ufl.exp((-(AV_Fn -  3.41749999999999983e-13)) / 1.367e-15), (-1.0))
        AV_tau_v = 1.91 + 2.09 * pow(1.0 + ufl.exp((-(AV_Fn -  3.41749999999999983e-13)) / 1.367e-15), (-1.0))
        AV_v_infinity = 1.0 - pow(1.0 + ufl.exp((-(AV_Fn - 6.835e-14)) / 1.367e-15), (-1.0))
        ydot[16] = ((-3.0) * AV_i_NaK - (3.0 * AV_i_NaCa + AV_i_B_Na + AV_i_Na)) / (self.AC_V_i * self.AC_F)
        
        ydot[0] = (-(AV_i_Na + AV_i_K1 + AV_i_to + AV_i_Kur + AV_i_Kr + AV_i_Ks + AV_i_B_Na + AV_i_B_Ca + AV_i_NaK + AV_i_CaP + AV_i_NaCa + AV_i_Ca_L)) / self.AC_Cm
        ydot[13] = (AV_u_infinity - NV_Ith_S(y, 13)) / self.AC_tau_u
        ydot[14] = (AV_v_infinity - NV_Ith_S(y, 14)) / AV_tau_v
        
        AV_B1 = (2.0 * AV_i_NaCa - (AV_i_CaP + AV_i_Ca_L + AV_i_B_Ca)) / (2.0 * self.AC_V_i * self.AC_F) + (self.AC_V_up * (AV_i_up_leak - AV_i_up) + AV_i_rel * self.AC_V_rel) / self.AC_V_i
        AV_B2 = 1.0 + self.AC_TRPN_max * self.AC_Km_TRPN / pow(NV_Ith_S(y, 17) + self.AC_Km_TRPN, 2.0) + self.AC_CMDN_max * self.AC_Km_CMDN / pow(NV_Ith_S(y, 17) + self.AC_Km_CMDN, 2.0)
        ydot[17] = AV_B1 / AV_B2

        return ydot
    
    def f_expl(self, y):
        ydot = [None]*self.size        

        # Non Linear (in the gating variables) terms

        #/* L_type_Ca_channel_f_Ca_gate */
        AV_f_Ca_infinity = pow(1.0 + NV_Ith_S(y, 17) / 0.00035, (-1.0))
        ydot[12] = (AV_f_Ca_infinity - NV_Ith_S(y, 12)) / self.AC_tau_f_Ca

        #/* transfer_current_from_NSR_to_JSR */
        AV_i_tr = (NV_Ith_S(y, 20) - NV_Ith_S(y, 19)) / self.AC_tau_tr
        
        #/* Ca_leak_current_by_the_NSR */
        AV_i_up_leak = self.AC_I_up_max * NV_Ith_S(y, 20) / self.AC_Ca_up_max
        
        #/* Ca_release_current_from_JSR */
        AV_i_rel = self.AC_K_rel * pow(NV_Ith_S(y, 13), 2.0) * NV_Ith_S(y, 14) * NV_Ith_S(y, 15) * (NV_Ith_S(y, 19) - NV_Ith_S(y, 17))
        
        #/* intracellular_ion_concentrations */
        ydot[19] = (AV_i_tr - AV_i_rel) * pow(1.0 + self.AC_CSQN_max * self.AC_Km_CSQN / pow(NV_Ith_S(y, 19) + self.AC_Km_CSQN, 2.0), (-1.0))

        #/* Ca_uptake_current_by_the_NSR */
        AV_i_up = self.AC_I_up_max / (1.0 + self.AC_K_up / NV_Ith_S(y, 17))
        ydot[20] = AV_i_up - (AV_i_up_leak + AV_i_tr * self.AC_V_rel / self.AC_V_up)
        
        #/* sarcolemmal_calcium_pump_current */
        AV_i_CaP = self.AC_Cm * self.AC_i_CaP_max * NV_Ith_S(y, 17) / (0.0005 + NV_Ith_S(y, 17))
        
        #/* sodium_potassium_pump */
        AV_f_NaK = pow(1.0 + 0.1245 * ufl.exp((-0.1) * self.AC_F * NV_Ith_S(y, 0) / (self.AC_R * self.AC_T)) + 0.0365 * self.AC_sigma * ufl.exp((-self.AC_F) * NV_Ith_S(y, 0) / (self.AC_R * self.AC_T)), (-1.0))
        AV_i_NaK = self.AC_Cm * self.AC_i_NaK_max * AV_f_NaK * 1.0 / (1.0 + pow(self.AC_Km_Na_i / NV_Ith_S(y, 16), 1.5)) * self.AC_K_o / (self.AC_K_o + self.AC_Km_K_o)
        
        #/* time_independent_potassium_current */
        AV_E_K = self.AC_R * self.AC_T / self.AC_F * ufl.ln(self.AC_K_o / NV_Ith_S(y, 18))
        AV_i_K1 = self.AC_Cm * self.AC_g_K1 * (NV_Ith_S(y, 0) - AV_E_K) / (1.0 + ufl.exp(0.07 * (NV_Ith_S(y, 0) + 80.0)))
        
        #/* transient_outward_K_current */
        AV_i_to = self.AC_Cm * self.AC_g_to * pow(NV_Ith_S(y, 4), 3.0) * NV_Ith_S(y, 5) * (NV_Ith_S(y, 0) - AV_E_K)                                
        
        #/* ultrarapid_delayed_rectifier_K_current */
        AV_g_Kur = 0.005 + 0.05 / (1.0 + ufl.exp((NV_Ith_S(y, 0) - 15.0) / (-13.0)))
        AV_i_Kur = self.AC_Cm * AV_g_Kur * pow(NV_Ith_S(y, 6), 3.0) * NV_Ith_S(y, 7) * (NV_Ith_S(y, 0) - AV_E_K)                                
        
        #/* *remaining* */
        AV_i_Ca_L = self.AC_Cm * self.AC_g_Ca_L * NV_Ith_S(y, 10) * NV_Ith_S(y, 11) * NV_Ith_S(y, 12) * (NV_Ith_S(y, 0) - 65.0)
        AV_i_NaCa = self.AC_Cm * self.AC_I_NaCa_max * (ufl.exp(self.AC_Na_Ca_exchanger_current_gamma * self.AC_F * NV_Ith_S(y, 0) / (self.AC_R * self.AC_T)) * pow(NV_Ith_S(y, 16), 3.0) * self.AC_Ca_o - ufl.exp((self.AC_Na_Ca_exchanger_current_gamma - 1.0) * self.AC_F * NV_Ith_S(y, 0) / (self.AC_R * self.AC_T)) * pow(self.AC_Na_o, 3.0) * NV_Ith_S(y, 17)) / ((pow(self.AC_K_mNa, 3.0) + pow(self.AC_Na_o, 3.0)) * (self.AC_K_mCa + self.AC_Ca_o) * (1.0 + self.AC_K_sat * ufl.exp((self.AC_Na_Ca_exchanger_current_gamma - 1.0) * NV_Ith_S(y, 0) * self.AC_F / (self.AC_R * self.AC_T))))
        AV_E_Ca = self.AC_R * self.AC_T / (2.0 * self.AC_F) * ufl.ln(self.AC_Ca_o / NV_Ith_S(y, 17))
        AV_i_B_K = self.AC_Cm * self.AC_g_B_K * (NV_Ith_S(y, 0) - AV_E_K)
        AV_E_Na = self.AC_R * self.AC_T / self.AC_F * ufl.ln(self.AC_Na_o / NV_Ith_S(y, 16))
        AV_i_Kr = self.AC_Cm * self.AC_g_Kr * NV_Ith_S(y, 8) * (NV_Ith_S(y, 0) - AV_E_K) / (1.0 + ufl.exp((NV_Ith_S(y, 0) + 15.0) / 22.4))
        AV_i_Ks = self.AC_Cm * self.AC_g_Ks * pow(NV_Ith_S(y, 9), 2.0) * (NV_Ith_S(y, 0) - AV_E_K)
        AV_Fn = 1000.0 * (1e-15 * self.AC_V_rel * AV_i_rel - 1e-15 / (2.0 * self.AC_F) * (0.5 * AV_i_Ca_L - 0.2 * AV_i_NaCa))
        AV_i_B_Ca = self.AC_Cm * self.AC_g_B_Ca * (NV_Ith_S(y, 0) - AV_E_Ca)
        AV_i_B_Na = self.AC_Cm * self.AC_g_B_Na * (NV_Ith_S(y, 0) - AV_E_Na)
        AV_i_Na = self.AC_Cm * self.AC_g_Na * pow(NV_Ith_S(y, 1), 3.0) * NV_Ith_S(y, 2) * NV_Ith_S(y, 3) * (NV_Ith_S(y, 0) - AV_E_Na)
        ydot[18] = (2.0 * AV_i_NaK - (AV_i_K1 + AV_i_to + AV_i_Kur + AV_i_Kr + AV_i_Ks + AV_i_B_K)) / (self.AC_V_i * self.AC_F)
        AV_u_infinity = pow(1.0 + ufl.exp((-(AV_Fn -  3.41749999999999983e-13)) / 1.367e-15), (-1.0))
        AV_tau_v = 1.91 + 2.09 * pow(1.0 + ufl.exp((-(AV_Fn -  3.41749999999999983e-13)) / 1.367e-15), (-1.0))
        AV_v_infinity = 1.0 - pow(1.0 + ufl.exp((-(AV_Fn - 6.835e-14)) / 1.367e-15), (-1.0))
        ydot[16] = ((-3.0) * AV_i_NaK - (3.0 * AV_i_NaCa + AV_i_B_Na + AV_i_Na)) / (self.AC_V_i * self.AC_F)
        
        ydot[0] = (-(AV_i_Na + AV_i_K1 + AV_i_to + AV_i_Kur + AV_i_Kr + AV_i_Ks + AV_i_B_Na + AV_i_B_Ca + AV_i_NaK + AV_i_CaP + AV_i_NaCa + AV_i_Ca_L)) / self.AC_Cm
        ydot[13] = (AV_u_infinity - NV_Ith_S(y, 13)) / self.AC_tau_u
        ydot[14] = (AV_v_infinity - NV_Ith_S(y, 14)) / AV_tau_v
        
        AV_B1 = (2.0 * AV_i_NaCa - (AV_i_CaP + AV_i_Ca_L + AV_i_B_Ca)) / (2.0 * self.AC_V_i * self.AC_F) + (self.AC_V_up * (AV_i_up_leak - AV_i_up) + AV_i_rel * self.AC_V_rel) / self.AC_V_i
        AV_B2 = 1.0 + self.AC_TRPN_max * self.AC_Km_TRPN / pow(NV_Ith_S(y, 17) + self.AC_Km_TRPN, 2.0) + self.AC_CMDN_max * self.AC_Km_CMDN / pow(NV_Ith_S(y, 17) + self.AC_Km_CMDN, 2.0)
        ydot[17] = AV_B1 / AV_B2

        return ydot
    
    def u_exp_coeffs(self, y):
        
        yinf = [None]*self.size
        tau = [None]*self.size

        # Linear (in the gating variables) terms
        
        #/* Ca_release_current_from_JSR_w_gate */
        tau[15] = ufl.conditional(ufl.lt(  ufl.algebra.Abs(NV_Ith_S(y, 0) - 7.9), 1e-10), \
                                    6.0 * 0.2 / 1.3, \
                                    6.0 * (1.0 - ufl.exp((-(NV_Ith_S(y, 0) - 7.9)) / 5.0)) / ((1.0 + 0.3 * ufl.exp((-(NV_Ith_S(y, 0) - 7.9)) / 5.0)) * 1.0 * (NV_Ith_S(y, 0) - 7.9)))
        yinf[15] = 1.0 - pow(1.0 + ufl.exp((-(NV_Ith_S(y, 0) - 40.0)) / 17.0), (-1.0))
        
        #/* L_type_Ca_channel_d_gate */
        yinf[10] = pow(1.0 + ufl.exp((NV_Ith_S(y, 0) + 10.0) / (-8.0)), (-1.0))
        tau[10] = ufl.conditional(ufl.lt(  ufl.algebra.Abs(NV_Ith_S(y, 0) + 10.0), 1e-10), 4.579 / (1.0 + ufl.exp((NV_Ith_S(y, 0) + 10.0) / (-6.24))), (1.0 - ufl.exp((NV_Ith_S(y, 0) + 10.0) / (-6.24))) / (0.035 * (NV_Ith_S(y, 0) + 10.0) * (1.0 + ufl.exp((NV_Ith_S(y, 0) + 10.0) / (-6.24)))))
                
        #/* L_type_Ca_channel_f_gate */
        yinf[11] = ufl.exp((-(NV_Ith_S(y, 0) + 28.0)) / 6.9) / (1.0 + ufl.exp((-(NV_Ith_S(y, 0) + 28.0)) / 6.9))
        tau[11] = 9.0 * pow(0.0197 * ufl.exp((-pow(0.0337, 2.0)) * pow(NV_Ith_S(y, 0) + 10.0, 2.0)) + 0.02, (-1.0))
        
        #/* fast_sodium_current_h_gate */
        AV_alpha_h = ufl.conditional(ufl.lt(NV_Ith_S(y, 0), (-40.0)), 0.135 * ufl.exp((NV_Ith_S(y, 0) + 80.0) / (-6.8)), 0.0)
        AV_beta_h = ufl.conditional(ufl.lt(NV_Ith_S(y, 0) , (-40.0)),3.56 * ufl.exp(0.079 * NV_Ith_S(y, 0)) + 310000.0 * ufl.exp(0.35 * NV_Ith_S(y, 0)), 1.0 / (0.13 * (1.0 + ufl.exp((NV_Ith_S(y, 0) + 10.66) / (-11.1)))))
        yinf[2] = AV_alpha_h / (AV_alpha_h + AV_beta_h)
        tau[2] = 1.0 / (AV_alpha_h + AV_beta_h)
        
        #/* fast_sodium_current_j_gate */
        AV_alpha_j = ufl.conditional(ufl.lt(NV_Ith_S(y, 0) , (-40.0)),((-127140.0) * ufl.exp(0.2444 * NV_Ith_S(y, 0)) - 3.474e-05 * ufl.exp((-0.04391) * NV_Ith_S(y, 0))) * (NV_Ith_S(y, 0) + 37.78) / (1.0 + ufl.exp(0.311 * (NV_Ith_S(y, 0) + 79.23))), 0.0)
        AV_beta_j = ufl.conditional(ufl.lt(NV_Ith_S(y, 0), (-40.0)),0.1212 * ufl.exp((-0.01052) * NV_Ith_S(y, 0)) / (1.0 + ufl.exp((-0.1378) * (NV_Ith_S(y, 0) + 40.14))), 0.3 * ufl.exp((-2.535e-07) * NV_Ith_S(y, 0)) / (1.0 + ufl.exp((-0.1) * (NV_Ith_S(y, 0) + 32.0))))
        yinf[3] = AV_alpha_j / (AV_alpha_j + AV_beta_j)
        tau[3] = 1.0 / (AV_alpha_j + AV_beta_j)
        
        #/* fast_sodium_current_m_gate */
        AV_alpha_m = ufl.conditional(ufl.eq(NV_Ith_S(y, 0), (-47.13)),3.2, 0.32 * (NV_Ith_S(y, 0) + 47.13) / (1.0 - ufl.exp((-0.1) * (NV_Ith_S(y, 0) + 47.13))))
        AV_beta_m = 0.08 * ufl.exp((-NV_Ith_S(y, 0)) / 11.0)
        yinf[1] = AV_alpha_m / (AV_alpha_m + AV_beta_m)
        tau[1] = 1.0 / (AV_alpha_m + AV_beta_m)
        
        #/* rapid_delayed_rectifier_K_current_xr_gate */
        AV_alpha_xr = ufl.conditional(ufl.lt(  ufl.algebra.Abs(NV_Ith_S(y, 0) + 14.1) , 1e-10),0.0015, 0.0003 * (NV_Ith_S(y, 0) + 14.1) / (1.0 - ufl.exp((NV_Ith_S(y, 0) + 14.1) / (-5.0))))
        AV_beta_xr = ufl.conditional(ufl.lt(  ufl.algebra.Abs(NV_Ith_S(y, 0) - 3.3328), 1e-10),3.78361180000000004e-04,  7.38980000000000030e-05 * (NV_Ith_S(y, 0) - 3.3328) / (ufl.exp((NV_Ith_S(y, 0) - 3.3328) / 5.1237) - 1.0))
        yinf[8] = pow(1.0 + ufl.exp((NV_Ith_S(y, 0) + 14.1) / (-6.5)), (-1.0))
        tau[8] = pow(AV_alpha_xr + AV_beta_xr, (-1.0))
        
        #/* slow_delayed_rectifier_K_current_xs_gate */
        AV_alpha_xs = ufl.conditional(ufl.lt(  ufl.algebra.Abs(NV_Ith_S(y, 0) - 19.9) , 1e-10),0.00068, 4e-05 * (NV_Ith_S(y, 0) - 19.9) / (1.0 - ufl.exp((NV_Ith_S(y, 0) - 19.9) / (-17.0))))
        AV_beta_xs = ufl.conditional(ufl.lt(  ufl.algebra.Abs(NV_Ith_S(y, 0) - 19.9) , 1e-10),0.000315, 3.5e-05 * (NV_Ith_S(y, 0) - 19.9) / (ufl.exp((NV_Ith_S(y, 0) - 19.9) / 9.0) - 1.0))
        yinf[9] = pow(1.0 + ufl.exp((NV_Ith_S(y, 0) - 19.9) / (-12.7)), (-0.5))
        tau[9] = 0.5 * pow(AV_alpha_xs + AV_beta_xs, (-1.0))

        #/* transient_outward_K_current_oa_gate */
        AV_alpha_oa = 0.65 * pow(ufl.exp((NV_Ith_S(y, 0) - (-10.0)) / (-8.5)) + ufl.exp((NV_Ith_S(y, 0) - (-10.0) - 40.0) / (-59.0)), (-1.0))
        AV_beta_oa = 0.65 * pow(2.5 + ufl.exp((NV_Ith_S(y, 0) - (-10.0) + 72.0) / 17.0), (-1.0))
        yinf[4] = pow(1.0 + ufl.exp((NV_Ith_S(y, 0) - (-10.0) + 10.47) / (-17.54)), (-1.0))
        tau[4] = pow(AV_alpha_oa + AV_beta_oa, (-1.0)) / self.AC_K_Q10

        #/* transient_outward_K_current_oi_gate */
        AV_alpha_oi = pow(18.53 + 1.0 * ufl.exp((NV_Ith_S(y, 0) - (-10.0) + 103.7) / 10.95), (-1.0))
        AV_beta_oi = pow(35.56 + 1.0 * ufl.exp((NV_Ith_S(y, 0) - (-10.0) - 8.74) / (-7.44)), (-1.0))
        yinf[5] = pow(1.0 + ufl.exp((NV_Ith_S(y, 0) - (-10.0) + 33.1) / 5.3), (-1.0))
        tau[5] = pow(AV_alpha_oi + AV_beta_oi, (-1.0)) / self.AC_K_Q10

        #/* ultrarapid_delayed_rectifier_K_current_ua_gate */
        AV_alpha_ua = 0.65 * pow(ufl.exp((NV_Ith_S(y, 0) - (-10.0)) / (-8.5)) + ufl.exp((NV_Ith_S(y, 0) - (-10.0) - 40.0) / (-59.0)), (-1.0))
        AV_beta_ua = 0.65 * pow(2.5 + ufl.exp((NV_Ith_S(y, 0) - (-10.0) + 72.0) / 17.0), (-1.0))
        yinf[6] = pow(1.0 + ufl.exp((NV_Ith_S(y, 0) - (-10.0) + 20.3) / (-9.6)), (-1.0))
        tau[6] = pow(AV_alpha_ua + AV_beta_ua, (-1.0)) / self.AC_K_Q10

        #/* ultrarapid_delayed_rectifier_K_current_ui_gate */
        AV_alpha_ui = pow(21.0 + 1.0 * ufl.exp((NV_Ith_S(y, 0) - (-10.0) - 195.0) / (-28.0)), (-1.0))
        AV_beta_ui = 1.0 / ufl.exp((NV_Ith_S(y, 0) - (-10.0) - 168.0) / (-16.0))
        yinf[7] = pow(1.0 + ufl.exp((NV_Ith_S(y, 0) - (-10.0) - 109.45) / 27.48), (-1.0))
        tau[7] = pow(AV_alpha_ui + AV_beta_ui, (-1.0)) / self.AC_K_Q10

        return yinf,tau
    

    def initial_values(self):

        y0 = [None]*self.size
        
        y0[0] = -81.18
        y0[1] = 0.002908
        y0[2] = 0.9649
        y0[3] = 0.9775
        y0[4] = 0.03043
        y0[5] = 0.9992
        y0[6] = 0.004966
        y0[7] = 0.9986
        y0[8] = 3.296e-05
        y0[9] = 0.01869
        y0[10] = 0.0001367
        y0[11] = 0.9996
        y0[12] = 0.7755
        y0[13] = 2.35e-112
        y0[14] = 1.0
        y0[15] = 0.9992
        y0[16] = 11.17
        y0[17] = 0.0001013
        y0[18] = 139.0
        y0[19] = 1.488
        y0[20] = 1.488

        return y0
    

# Mildly stiff with rho_max ~ 450
class Fox2002(IonicModel):
    def __init__(self):
        super(Fox2002,self).__init__()

        self.size = 13

        self.AC_K_mfCa = 0.18
        self.AC_tau_f_Ca = 30.0
        
        self.AC_shift_h = 0.0
        
        self.AC_shift_j = 0.0
        
        self.AC_K_mpCa = 0.05
        self.AC_i_pCa_max = 0.05
        
        self.AC_Ca_o = 2000.0
        self.AC_K_i = 149.4
        self.AC_K_o = 4.0
        self.AC_Na_i = 10.0
        self.AC_Na_o = 138.0
        
        self.AC_C_sc = 1.0
        self.AC_P_Ca = 2.26e-05
        self.AC_P_CaK = 5.79e-07
        self.AC_i_Ca_half = (-0.265)
        
        self.AC_K_NaCa = 1500.0
        self.AC_K_mCa = 1380.0
        self.AC_K_mNa = 87.5
        self.AC_K_sat = 0.2
        self.AC_eta = 0.35
        
        self.AC_g_Cab = 0.0003842
        
        self.AC_A_Cap = 0.0001534
        self.AC_CMDN_tot = 10.0
        self.AC_CSQN_tot = 10000.0
        self.AC_K_mCMDN = 2.0
        self.AC_K_mCSQN = 600.0
        self.AC_K_mup = 0.32
        self.AC_P_leak = 1e-06
        self.AC_P_rel = 6.0
        self.AC_V_SR = 2e-06
        self.AC_V_myo = 2.584e-05
        self.AC_V_up = 0.1
        
        self.AC_g_Na = 12.8
        
        self.AC_F = 96.5
        self.AC_R = 8.314
        self.AC_T = 310.0
        self.AC_stim_amplitude = (-80.0)
        self.AC_stim_end = 9000.0
        
        self.AC_g_Kp = 0.002216
        
        self.AC_E_K = self.AC_R * self.AC_T / self.AC_F * np.log(self.AC_K_o / self.AC_K_i)
        self.AC_g_Kr = 0.0136
        
        self.AC_E_Ks = self.AC_R * self.AC_T / self.AC_F * np.log((self.AC_K_o + 0.01833 * self.AC_Na_o) / (self.AC_K_i + 0.01833 * self.AC_Na_i))
        self.AC_g_Ks = 0.0245
        
        self.AC_g_Nab = 0.0031
        
        self.AC_K_mKo = 1.5
        self.AC_K_mNai = 10.0
        self.AC_i_NaK_max = 0.693
        self.AC_sigma = 1.0 / 7.0 * (np.exp(self.AC_Na_o / 67.3) - 1.0)
        
        self.AC_K_mK1 = 13.0
        self.AC_g_K1 = 2.8
    
        self.AC_g_to = 0.23815
        
        self.AC_E_Na = self.AC_R * self.AC_T / self.AC_F * np.log(self.AC_Na_o / self.AC_Na_i)

    def f(self, y):

        ydot = [None]*self.size
            
        #/* L_type_Ca_current_d_gate */
        AV_L_type_Ca_current_d_gate_E0_m = NV_Ith_S(y, 0) + 40.0
        AV_d_infinity = 1.0 / (1.0 + ufl.exp((NV_Ith_S(y, 0) + 10.0) / (-6.24)))
        AV_tau_d = 1.0 / (0.25 * ufl.exp((-0.01) * NV_Ith_S(y, 0)) / (1.0 + ufl.exp((-0.07) * NV_Ith_S(y, 0))) + 0.07 * ufl.exp((-0.05) * AV_L_type_Ca_current_d_gate_E0_m) / (1.0 + ufl.exp(0.05 * AV_L_type_Ca_current_d_gate_E0_m)))
        ydot[9] = (AV_d_infinity - NV_Ith_S(y, 9)) / AV_tau_d
        
        #/* L_type_Ca_current_f_Ca_gate */
        AV_f_Ca_infinity = 1.0 / (1.0 + pow(NV_Ith_S(y, 11) / self.AC_K_mfCa, 3.0))
        ydot[10] = (AV_f_Ca_infinity - NV_Ith_S(y, 10)) / self.AC_tau_f_Ca
        
        #/* L_type_Ca_current_f_gate */
        AV_f_infinity = 1.0 / (1.0 + ufl.exp((NV_Ith_S(y, 0) + 12.5) / 5.0))
        AV_tau_f = 30.0 + 200.0 / (1.0 + ufl.exp((NV_Ith_S(y, 0) + 20.0) / 9.5))
        ydot[8] = (AV_f_infinity - NV_Ith_S(y, 8)) / AV_tau_f
    
        
        #/* fast_sodium_current_h_gate */
        AV_alpha_h = 0.135 * ufl.exp((NV_Ith_S(y, 0) + 80.0 - self.AC_shift_h) / (-6.8))
        AV_beta_h = 7.5 / (1.0 + ufl.exp((-0.1) * (NV_Ith_S(y, 0) + 11.0 - self.AC_shift_h)))
        ydot[2] = AV_alpha_h * (1.0 - NV_Ith_S(y, 2)) - AV_beta_h * NV_Ith_S(y, 2)
        
        #/* fast_sodium_current_j_gate */
        AV_alpha_j = 0.175 * ufl.exp((NV_Ith_S(y, 0) + 100.0 - self.AC_shift_j) / (-23.0)) / (1.0 + ufl.exp(0.15 * (NV_Ith_S(y, 0) + 79.0 - self.AC_shift_j)))
        AV_beta_j = 0.3 / (1.0 + ufl.exp((-0.1) * (NV_Ith_S(y, 0) + 32.0 - self.AC_shift_j)))
        ydot[3] = AV_alpha_j * (1.0 - NV_Ith_S(y, 3)) - AV_beta_j * NV_Ith_S(y, 3)
        
        #/* fast_sodium_current_m_gate */
        AV_fast_sodium_current_m_gate_E0_m = NV_Ith_S(y, 0) + 47.13
        AV_beta_m = 0.08 * ufl.exp((-NV_Ith_S(y, 0)) / 11.0)
        AV_alpha_m = 0.32 * AV_fast_sodium_current_m_gate_E0_m / (1.0 - ufl.exp((-0.1) * AV_fast_sodium_current_m_gate_E0_m))
        ydot[1] = AV_alpha_m * (1.0 - NV_Ith_S(y, 1)) - AV_beta_m * NV_Ith_S(y, 1)
        
        #/* plateau_potassium_current_Kp_gate */
        AV_Kp_V = 1.0 / (1.0 + ufl.exp((7.488 - NV_Ith_S(y, 0)) / 5.98))
        
        #/* rapid_activating_delayed_rectifiyer_K_current_X_kr_gate */
        AV_X_kr_inf = 1.0 / (1.0 + ufl.exp((-2.182) - 0.1819 * NV_Ith_S(y, 0)))
        AV_tau_X_kr = 43.0 + 1.0 / (ufl.exp((-5.495) + 0.1691 * NV_Ith_S(y, 0)) + ufl.exp((-7.677) - 0.0128 * NV_Ith_S(y, 0)))
        ydot[4] = (AV_X_kr_inf - NV_Ith_S(y, 4)) / AV_tau_X_kr
        
        #/* sarcolemmal_calcium_pump */
        AV_i_p_Ca = self.AC_i_pCa_max * NV_Ith_S(y, 11) / (self.AC_K_mpCa + NV_Ith_S(y, 11))
        
        #/* slow_activating_delayed_rectifiyer_K_current_X_ks_gate */
        AV_X_ks_infinity = 1.0 / (1.0 + ufl.exp((NV_Ith_S(y, 0) - 16.0) / (-13.6)))
        AV_tau_X_ks = 1.0 / (7.19e-05 * (NV_Ith_S(y, 0) - 10.0) / (1.0 - ufl.exp((-0.148) * (NV_Ith_S(y, 0) - 10.0))) + 0.000131 * (NV_Ith_S(y, 0) - 10.0) / (ufl.exp(0.0687 * (NV_Ith_S(y, 0) - 10.0)) - 1.0))
        ydot[5] = (AV_X_ks_infinity - NV_Ith_S(y, 5)) / AV_tau_X_ks
        
        #/* transient_outward_potassium_current_X_to_gate */
        AV_alpha_X_to = 0.04516 * ufl.exp(0.03577 * NV_Ith_S(y, 0))
        AV_beta_X_to = 0.0989 * ufl.exp((-0.06237) * NV_Ith_S(y, 0))
        ydot[6] = AV_alpha_X_to * (1.0 - NV_Ith_S(y, 6)) - AV_beta_X_to * NV_Ith_S(y, 6)
        
        #/* transient_outward_potassium_current_Y_to_gate */
        AV_alpha_Y_to = 0.005415 * ufl.exp((NV_Ith_S(y, 0) + 33.5) / (-5.0)) / (1.0 + 0.051335 * ufl.exp((NV_Ith_S(y, 0) + 33.5) / (-5.0)))
        AV_beta_Y_to = 0.005415 * ufl.exp((NV_Ith_S(y, 0) + 33.5) / 5.0) / (1.0 + 0.051335 * ufl.exp((NV_Ith_S(y, 0) + 33.5) / 5.0))
        ydot[7] = AV_alpha_Y_to * (1.0 - NV_Ith_S(y, 7)) - AV_beta_Y_to * NV_Ith_S(y, 7)
        
        #/* calcium_dynamics */
        AV_calcium_dynamics_gamma = 1.0 / (1.0 + pow(2000.0 / NV_Ith_S(y, 12), 3.0))
        AV_J_leak = self.AC_P_leak * (NV_Ith_S(y, 12) - NV_Ith_S(y, 11))
        AV_J_rel = self.AC_P_rel * NV_Ith_S(y, 8) * NV_Ith_S(y, 9) * NV_Ith_S(y, 10) * (AV_calcium_dynamics_gamma * NV_Ith_S(y, 12) - NV_Ith_S(y, 11)) / (1.0 + 1.65 * ufl.exp(NV_Ith_S(y, 0) / 20.0))
        AV_J_up = self.AC_V_up / (1.0 + pow(self.AC_K_mup / NV_Ith_S(y, 11), 2.0))
        AV_beta_SR = 1.0 / (1.0 + self.AC_CSQN_tot * self.AC_K_mCSQN / pow(self.AC_K_mCSQN + NV_Ith_S(y, 12), 2.0))
        AV_beta_i = 1.0 / (1.0 + self.AC_CMDN_tot * self.AC_K_mCMDN / pow(self.AC_K_mCMDN + NV_Ith_S(y, 11), 2.0))
        ydot[12] = AV_beta_SR * (AV_J_up - AV_J_leak - AV_J_rel) * self.AC_V_myo / self.AC_V_SR
            
        #/* rapid_activating_delayed_rectifiyer_K_current */
        AV_R_V = 1.0 / (1.0 + 2.5 * ufl.exp(0.1 * (NV_Ith_S(y, 0) + 28.0)))
        AV_i_Kr = self.AC_g_Kr * AV_R_V * NV_Ith_S(y, 4) * np.sqrt(self.AC_K_o / 4.0) * (NV_Ith_S(y, 0) - self.AC_E_K)
        
        #/* slow_activating_delayed_rectifiyer_K_current */
        AV_i_Ks = self.AC_g_Ks * pow(NV_Ith_S(y, 5), 2.0) * (NV_Ith_S(y, 0) - self.AC_E_Ks)
        
        #/* sodium_potassium_pump */
        AV_f_NaK = 1.0 / (1.0 + 0.1245 * ufl.exp((-0.1) * NV_Ith_S(y, 0) * self.AC_F / (self.AC_R * self.AC_T)) + 0.0365 * self.AC_sigma * ufl.exp((-NV_Ith_S(y, 0)) * self.AC_F / (self.AC_R * self.AC_T)))
        AV_i_NaK = self.AC_i_NaK_max * AV_f_NaK / (1.0 + pow(self.AC_K_mNai / self.AC_Na_i, 1.5)) * self.AC_K_o / (self.AC_K_o + self.AC_K_mKo)
        
        #/* time_independent_potassium_current_K1_gate */
        AV_K1_infinity = 1.0 / (2.0 + ufl.exp(1.62 * self.AC_F / (self.AC_R * self.AC_T) * (NV_Ith_S(y, 0) - self.AC_E_K)))
        
        #/* transient_outward_potassium_current */
        AV_i_to = self.AC_g_to * NV_Ith_S(y, 6) * NV_Ith_S(y, 7) * (NV_Ith_S(y, 0) - self.AC_E_K)
        
        #/* *remaining* */
        AV_i_Ca_max = self.AC_P_Ca / self.AC_C_sc * 4.0 * NV_Ith_S(y, 0) * pow(self.AC_F, 2.0) / (self.AC_R * self.AC_T) * (NV_Ith_S(y, 11) * ufl.exp(2.0 * NV_Ith_S(y, 0) * self.AC_F / (self.AC_R * self.AC_T)) - 0.341 * self.AC_Ca_o) / (ufl.exp(2.0 * NV_Ith_S(y, 0) * self.AC_F / (self.AC_R * self.AC_T)) - 1.0)
        AV_i_NaCa = self.AC_K_NaCa / ((pow(self.AC_K_mNa, 3.0) + pow(self.AC_Na_o, 3.0)) * (self.AC_K_mCa + self.AC_Ca_o) * (1.0 + self.AC_K_sat * ufl.exp((self.AC_eta - 1.0) * NV_Ith_S(y, 0) * self.AC_F / (self.AC_R * self.AC_T)))) * (ufl.exp(self.AC_eta * NV_Ith_S(y, 0) * self.AC_F / (self.AC_R * self.AC_T)) * pow(self.AC_Na_i, 3.0) * self.AC_Ca_o - ufl.exp((self.AC_eta - 1.0) * NV_Ith_S(y, 0) * self.AC_F / (self.AC_R * self.AC_T)) * pow(self.AC_Na_o, 3.0) * NV_Ith_S(y, 11))
        AV_E_Ca = self.AC_R * self.AC_T / (2.0 * self.AC_F) * ufl.ln(self.AC_Ca_o / NV_Ith_S(y, 11))
        AV_i_Kp = self.AC_g_Kp * AV_Kp_V * (NV_Ith_S(y, 0) - self.AC_E_K)
        AV_i_K1 = self.AC_g_K1 * AV_K1_infinity * self.AC_K_o / (self.AC_K_o + self.AC_K_mK1) * (NV_Ith_S(y, 0) - self.AC_E_K)
        AV_i_Ca = AV_i_Ca_max * NV_Ith_S(y, 8) * NV_Ith_S(y, 9) * NV_Ith_S(y, 10)
        AV_i_CaK = self.AC_P_CaK / self.AC_C_sc * NV_Ith_S(y, 8) * NV_Ith_S(y, 9) * NV_Ith_S(y, 10) / (1.0 + AV_i_Ca_max / self.AC_i_Ca_half) * 1000.0 * NV_Ith_S(y, 0) * pow(self.AC_F, 2.0) / (self.AC_R * self.AC_T) * (self.AC_K_i * ufl.exp(NV_Ith_S(y, 0) * self.AC_F / (self.AC_R * self.AC_T)) - self.AC_K_o) / (ufl.exp(NV_Ith_S(y, 0) * self.AC_F / (self.AC_R * self.AC_T)) - 1.0)
        AV_i_Ca_b = self.AC_g_Cab * (NV_Ith_S(y, 0) - AV_E_Ca)
        AV_i_Na = self.AC_g_Na * pow(NV_Ith_S(y, 1), 3.0) * NV_Ith_S(y, 2) * NV_Ith_S(y, 3) * (NV_Ith_S(y, 0) - self.AC_E_Na)
        AV_i_Na_b = self.AC_g_Nab * (NV_Ith_S(y, 0) - self.AC_E_Na)
        ydot[11] = AV_beta_i * (AV_J_rel + AV_J_leak - AV_J_up - self.AC_A_Cap * self.AC_C_sc / (2.0 * self.AC_F * self.AC_V_myo) * (AV_i_Ca + AV_i_Ca_b + AV_i_p_Ca - 2.0 * AV_i_NaCa))
        ydot[0] = (-(AV_i_Na + AV_i_Ca + AV_i_CaK + AV_i_Kr + AV_i_Ks + AV_i_to + AV_i_K1 + AV_i_Kp + AV_i_NaCa + AV_i_NaK + AV_i_p_Ca + AV_i_Na_b + AV_i_Ca_b))
        
        return ydot
    

    def initial_values(self):

        y0 = [None]*self.size
        
        y0[0] = -94.7
        y0[1] =  2.46760000000000002e-04
        y0[2] = 0.99869
        y0[3] = 0.99887
        y0[4] = 0.229
        y0[5] = 0.0001
        y0[6] = 3.742e-05
        y0[7] = 1.0
        y0[8] = 0.983
        y0[9] = 0.0001
        y0[10] = 0.942
        y0[11] = 0.0472
        y0[12] = 320.0

        return y0
    

# Stiff
class TenTusscher2006_epi(IonicModel):
    def __init__(self):
        super(TenTusscher2006_epi,self).__init__()

        self.size = 19

        self.AC_Cm = 185.0
        self.AC_K_pCa = 0.0005
        self.AC_g_pCa = 0.1238
        self.AC_g_CaL = 0.0398
        self.AC_g_bca = 0.000592
        self.AC_Buf_c = 0.2
        self.AC_Buf_sr = 10.0
        self.AC_Buf_ss = 0.4
        self.AC_Ca_o = 2.0
        self.AC_EC = 1.5
        self.AC_K_buf_c = 0.001
        self.AC_K_buf_sr = 0.3
        self.AC_K_buf_ss = 0.00025
        self.AC_K_up = 0.00025
        self.AC_V_leak = 0.00036
        self.AC_V_rel = 0.102
        self.AC_V_sr = 1094.0
        self.AC_V_ss = 54.68
        self.AC_V_xfer = 0.0038
        self.AC_Vmax_up = 0.006375
        self.AC_k1_prime = 0.15
        self.AC_k2_prime = 0.045
        self.AC_k3 = 0.06
        self.AC_k4 = 0.005
        self.AC_max_sr = 2.5
        self.AC_min_sr = 1.0
        self.AC_g_Na = 14.838
        self.AC_g_K1 = 5.405
        self.AC_F = 96.485
        self.AC_R = 8.314
        self.AC_T = 310.0
        self.AC_V_c = 16404.0
        self.AC_stim_amplitude = (-52.0)
        self.AC_K_o = 5.4
        self.AC_g_pK = 0.0146
        self.AC_g_Kr = 0.153
        self.AC_P_kna = 0.03
        self.AC_g_Ks = 0.392
        self.AC_g_bna = 0.00029
        self.AC_K_NaCa = 1000.0
        self.AC_K_sat = 0.1
        self.AC_Km_Ca = 1.38
        self.AC_Km_Nai = 87.5
        self.AC_alpha = 2.5
        self.AC_sodium_calcium_exchanger_current_gamma = 0.35
        self.AC_Na_o = 140.0
        self.AC_K_mNa = 40.0
        self.AC_K_mk = 1.0
        self.AC_P_NaK = 2.724
        self.AC_g_to = 0.294

        #overall stiffness of f is 1000
        # sitff indeces:
        # 0 : no, 0.5
        # 1 : no, mostly <1, sometimes 8
        # 2: no, 1.4
        # 3: no, almost 0
        # 4: yes, 1000
        # 5: no, 6
        # 6: a bit, 20
        # 7: no, 3
        # 8: no, almost 0
        # 9: no, almost 0
        # 10: no, 0.5
        # 11: no, 0.3
        # 12: no, 1
        # 13: Unknown, spectral radius did not converge
        # 14: no, almost 0
        # 15: Unknown, spectral radius did not converge
        # 16: no, almost 0
        # 17: no, almost 0
        # 18: no, almost 0
        
        self.f_nonstiff_args = list(range(self.size))
        self.f_stiff_args = [0,4]
        self.f_expl_args = list(range(self.size))

    def f(self, y):

        ydot = [None]*self.size
        
        # Linear in gating variables

        # /* L_type_Ca_current_d_gate */
        AV_alpha_d = 1.4 / (1.0 + ufl.exp(((-35.0) - NV_Ith_S(y, 0)) / 13.0)) + 0.25
        AV_beta_d = 1.4 / (1.0 + ufl.exp((NV_Ith_S(y, 0) + 5.0) / 5.0))
        AV_d_inf = 1.0 / (1.0 + ufl.exp(((-8.0) - NV_Ith_S(y, 0)) / 7.5))
        AV_gamma_d = 1.0 / (1.0 + ufl.exp((50.0 - NV_Ith_S(y, 0)) / 20.0))
        AV_tau_d = 1.0 * AV_alpha_d * AV_beta_d + AV_gamma_d
        ydot[7] = (AV_d_inf - NV_Ith_S(y, 7)) / AV_tau_d
        
        # /* L_type_Ca_current_f2_gate */
        AV_f2_inf = 0.67 / (1.0 + ufl.exp((NV_Ith_S(y, 0) + 35.0) / 7.0)) + 0.33
        AV_tau_f2 = 562.0 * ufl.exp((-pow(NV_Ith_S(y, 0) + 27.0, 2.0)) / 240.0) + 31.0 / (1.0 + ufl.exp((25.0 - NV_Ith_S(y, 0)) / 10.0)) + 80.0 / (1.0 + ufl.exp((NV_Ith_S(y, 0) + 30.0) / 10.0))
        ydot[9] = (AV_f2_inf - NV_Ith_S(y, 9)) / AV_tau_f2
        
        # /* L_type_Ca_current_fCass_gate */
        AV_fCass_inf = 0.6 / (1.0 + pow(NV_Ith_S(y, 15) / 0.05, 2.0)) + 0.4
        AV_tau_fCass = 80.0 / (1.0 + pow(NV_Ith_S(y, 15) / 0.05, 2.0)) + 2.0
        ydot[10] = (AV_fCass_inf - NV_Ith_S(y, 10)) / AV_tau_fCass
        
        # /* L_type_Ca_current_f_gate */
        AV_f_inf = 1.0 / (1.0 + ufl.exp((NV_Ith_S(y, 0) + 20.0) / 7.0))
        AV_tau_f = 1102.5 * ufl.exp((-pow(NV_Ith_S(y, 0) + 27.0, 2.0)) / 225.0) + 200.0 / (1.0 + ufl.exp((13.0 - NV_Ith_S(y, 0)) / 10.0)) + 180.0 / (1.0 + ufl.exp((NV_Ith_S(y, 0) + 30.0) / 10.0)) + 20.0
        ydot[8] = (AV_f_inf - NV_Ith_S(y, 8)) / AV_tau_f                
        
        # /* fast_sodium_current_h_gate */
        AV_alpha_h = ufl.conditional(ufl.lt(NV_Ith_S(y, 0) , (-40.0)) , 0.057 * ufl.exp((-(NV_Ith_S(y, 0) + 80.0)) / 6.8) , 0.0)
        AV_beta_h = ufl.conditional(ufl.lt(NV_Ith_S(y, 0) , (-40.0)) , 2.7 * ufl.exp(0.079 * NV_Ith_S(y, 0)) + 310000.0 * ufl.exp(0.3485 * NV_Ith_S(y, 0)) , 0.77 / (0.13 * (1.0 + ufl.exp((NV_Ith_S(y, 0) + 10.66) / (-11.1)))))
        AV_h_inf = 1.0 / pow(1.0 + ufl.exp((NV_Ith_S(y, 0) + 71.55) / 7.43), 2.0)
        AV_tau_h = 1.0 / (AV_alpha_h + AV_beta_h)
        ydot[5] = (AV_h_inf - NV_Ith_S(y, 5)) / AV_tau_h
        
        # /* fast_sodium_current_j_gate */
        AV_alpha_j = ufl.conditional(ufl.lt(NV_Ith_S(y, 0) , (-40.0)) , ((-25428.0) * ufl.exp(0.2444 * NV_Ith_S(y, 0)) - 6.948e-06 * ufl.exp((-0.04391) * NV_Ith_S(y, 0))) * (NV_Ith_S(y, 0) + 37.78) / 1.0 / (1.0 + ufl.exp(0.311 * (NV_Ith_S(y, 0) + 79.23))) , 0.0)
        AV_beta_j = ufl.conditional(ufl.lt(NV_Ith_S(y, 0) , (-40.0)) , 0.02424 * ufl.exp((-0.01052) * NV_Ith_S(y, 0)) / (1.0 + ufl.exp((-0.1378) * (NV_Ith_S(y, 0) + 40.14))) , 0.6 * ufl.exp(0.057 * NV_Ith_S(y, 0)) / (1.0 + ufl.exp((-0.1) * (NV_Ith_S(y, 0) + 32.0))))
        AV_j_inf = 1.0 / pow(1.0 + ufl.exp((NV_Ith_S(y, 0) + 71.55) / 7.43), 2.0)
        AV_tau_j = 1.0 / (AV_alpha_j + AV_beta_j)
        ydot[6] = (AV_j_inf - NV_Ith_S(y, 6)) / AV_tau_j
        
        # /* fast_sodium_current_m_gate */
        AV_alpha_m = 1.0 / (1.0 + ufl.exp(((-60.0) - NV_Ith_S(y, 0)) / 5.0))
        AV_beta_m = 0.1 / (1.0 + ufl.exp((NV_Ith_S(y, 0) + 35.0) / 5.0)) + 0.1 / (1.0 + ufl.exp((NV_Ith_S(y, 0) - 50.0) / 200.0))
        AV_m_inf = 1.0 / pow(1.0 + ufl.exp(((-56.86) - NV_Ith_S(y, 0)) / 9.03), 2.0)
        AV_tau_m = 1.0 * AV_alpha_m * AV_beta_m
        ydot[4] = (AV_m_inf - NV_Ith_S(y, 4)) / AV_tau_m
        
        # /* rapid_time_dependent_potassium_current_Xr1_gate */
        AV_alpha_xr1 = 450.0 / (1.0 + ufl.exp(((-45.0) - NV_Ith_S(y, 0)) / 10.0))
        AV_beta_xr1 = 6.0 / (1.0 + ufl.exp((NV_Ith_S(y, 0) + 30.0) / 11.5))
        AV_xr1_inf = 1.0 / (1.0 + ufl.exp(((-26.0) - NV_Ith_S(y, 0)) / 7.0))
        AV_tau_xr1 = 1.0 * AV_alpha_xr1 * AV_beta_xr1
        ydot[1] = (AV_xr1_inf - NV_Ith_S(y, 1)) / AV_tau_xr1
        
        # /* rapid_time_dependent_potassium_current_Xr2_gate */
        AV_alpha_xr2 = 3.0 / (1.0 + ufl.exp(((-60.0) - NV_Ith_S(y, 0)) / 20.0))
        AV_beta_xr2 = 1.12 / (1.0 + ufl.exp((NV_Ith_S(y, 0) - 60.0) / 20.0))
        AV_xr2_inf = 1.0 / (1.0 + ufl.exp((NV_Ith_S(y, 0) + 88.0) / 24.0))
        AV_tau_xr2 = 1.0 * AV_alpha_xr2 * AV_beta_xr2
        ydot[2] = (AV_xr2_inf - NV_Ith_S(y, 2)) / AV_tau_xr2
        
        # /* slow_time_dependent_potassium_current_Xs_gate */
        AV_alpha_xs = 1400.0 / ufl.sqrt(1.0 + ufl.exp((5.0 - NV_Ith_S(y, 0)) / 6.0))
        AV_beta_xs = 1.0 / (1.0 + ufl.exp((NV_Ith_S(y, 0) - 35.0) / 15.0))
        AV_xs_inf = 1.0 / (1.0 + ufl.exp(((-5.0) - NV_Ith_S(y, 0)) / 14.0))
        AV_tau_xs = 1.0 * AV_alpha_xs * AV_beta_xs + 80.0
        ydot[3] = (AV_xs_inf - NV_Ith_S(y, 3)) / AV_tau_xs
        
        # /* transient_outward_current_r_gate */
        AV_r_inf = 1.0 / (1.0 + ufl.exp((20.0 - NV_Ith_S(y, 0)) / 6.0))
        AV_tau_r = 9.5 * ufl.exp((-pow(NV_Ith_S(y, 0) + 40.0, 2.0)) / 1800.0) + 0.8
        ydot[12] = (AV_r_inf - NV_Ith_S(y, 12)) / AV_tau_r
        
        # /* transient_outward_current_s_gate */
        AV_s_inf = 1.0 / (1.0 + ufl.exp((NV_Ith_S(y, 0) + 20.0) / 5.0))
        AV_tau_s = 85.0 * ufl.exp((-pow(NV_Ith_S(y, 0) + 45.0, 2.0)) / 320.0) + 5.0 / (1.0 + ufl.exp((NV_Ith_S(y, 0) - 20.0) / 5.0)) + 3.0
        ydot[11] = (AV_s_inf - NV_Ith_S(y, 11)) / AV_tau_s
        
        # Non linear in gating variables

        # /* calcium_dynamics */
        AV_f_JCa_i_free = 1.0 / (1.0 + self.AC_Buf_c * self.AC_K_buf_c / pow(NV_Ith_S(y, 13) + self.AC_K_buf_c, 2.0))
        AV_f_JCa_sr_free = 1.0 / (1.0 + self.AC_Buf_sr * self.AC_K_buf_sr / pow(NV_Ith_S(y, 14) + self.AC_K_buf_sr, 2.0))
        AV_f_JCa_ss_free = 1.0 / (1.0 + self.AC_Buf_ss * self.AC_K_buf_ss / pow(NV_Ith_S(y, 15) + self.AC_K_buf_ss, 2.0))
        AV_i_leak = self.AC_V_leak * (NV_Ith_S(y, 14) - NV_Ith_S(y, 13))
        AV_i_up = self.AC_Vmax_up / (1.0 + pow(self.AC_K_up, 2.0) / pow(NV_Ith_S(y, 13), 2.0))
        AV_i_xfer = self.AC_V_xfer * (NV_Ith_S(y, 15) - NV_Ith_S(y, 13))
        AV_kcasr = self.AC_max_sr - (self.AC_max_sr - self.AC_min_sr) / (1.0 + pow(self.AC_EC / NV_Ith_S(y, 14), 2.0))
        AV_k1 = self.AC_k1_prime / AV_kcasr
        AV_k2 = self.AC_k2_prime * AV_kcasr
        AV_O = AV_k1 * pow(NV_Ith_S(y, 15), 2.0) * NV_Ith_S(y, 16) / (self.AC_k3 + AV_k1 * pow(NV_Ith_S(y, 15), 2.0))
        ydot[16] = (-AV_k2) * NV_Ith_S(y, 15) * NV_Ith_S(y, 16) + self.AC_k4 * (1.0 - NV_Ith_S(y, 16))
        AV_i_rel = self.AC_V_rel * AV_O * (NV_Ith_S(y, 14) - NV_Ith_S(y, 15))
        AV_ddt_Ca_sr_total = AV_i_up - (AV_i_rel + AV_i_leak)
        ydot[14] = AV_ddt_Ca_sr_total * AV_f_JCa_sr_free
        
        # /* reversal_potentials */
        AV_E_Ca = 0.5 * self.AC_R * self.AC_T / self.AC_F * ufl.ln(self.AC_Ca_o / NV_Ith_S(y, 13))
        AV_E_K = self.AC_R * self.AC_T / self.AC_F * ufl.ln(self.AC_K_o / NV_Ith_S(y, 18))
        
        # /* sodium_potassium_pump_current */
        AV_i_NaK = self.AC_P_NaK * self.AC_K_o / (self.AC_K_o + self.AC_K_mk) * NV_Ith_S(y, 17) / (NV_Ith_S(y, 17) + self.AC_K_mNa) / (1.0 + 0.1245 * ufl.exp((-0.1) * NV_Ith_S(y, 0) * self.AC_F / (self.AC_R * self.AC_T)) + 0.0353 * ufl.exp((-NV_Ith_S(y, 0)) * self.AC_F / (self.AC_R * self.AC_T)))
        
        # /* transient_outward_current */
        AV_i_to = self.AC_g_to * NV_Ith_S(y, 12) * NV_Ith_S(y, 11) * (NV_Ith_S(y, 0) - AV_E_K)
        
        # /* calcium_pump_current */
        AV_i_p_Ca = self.AC_g_pCa * NV_Ith_S(y, 13) / (NV_Ith_S(y, 13) + self.AC_K_pCa)

        # /* *remaining* */
        AV_i_CaL = self.AC_g_CaL * NV_Ith_S(y, 7) * NV_Ith_S(y, 8) * NV_Ith_S(y, 9) * NV_Ith_S(y, 10) * 4.0 * (NV_Ith_S(y, 0) - 15.0) * pow(self.AC_F, 2.0) / (self.AC_R * self.AC_T) * (0.25 * NV_Ith_S(y, 15) * ufl.exp(2.0 * (NV_Ith_S(y, 0) - 15.0) * self.AC_F / (self.AC_R * self.AC_T)) - self.AC_Ca_o) / (ufl.exp(2.0 * (NV_Ith_S(y, 0) - 15.0) * self.AC_F / (self.AC_R * self.AC_T)) - 1.0)
        AV_i_b_Ca = self.AC_g_bca * (NV_Ith_S(y, 0) - AV_E_Ca)
        AV_alpha_K1 = 0.1 / (1.0 + ufl.exp(0.06 * (NV_Ith_S(y, 0) - AV_E_K - 200.0)))
        AV_beta_K1 = (3.0 * ufl.exp(0.0002 * (NV_Ith_S(y, 0) - AV_E_K + 100.0)) + ufl.exp(0.1 * (NV_Ith_S(y, 0) - AV_E_K - 10.0))) / (1.0 + ufl.exp((-0.5) * (NV_Ith_S(y, 0) - AV_E_K)))
        AV_i_p_K = self.AC_g_pK * (NV_Ith_S(y, 0) - AV_E_K) / (1.0 + ufl.exp((25.0 - NV_Ith_S(y, 0)) / 5.98))
        AV_i_Kr = self.AC_g_Kr * np.sqrt(self.AC_K_o / 5.4) * NV_Ith_S(y, 1) * NV_Ith_S(y, 2) * (NV_Ith_S(y, 0) - AV_E_K)
        AV_E_Ks = self.AC_R * self.AC_T / self.AC_F * ufl.ln((self.AC_K_o + self.AC_P_kna * self.AC_Na_o) / (NV_Ith_S(y, 18) + self.AC_P_kna * NV_Ith_S(y, 17)))
        AV_E_Na = self.AC_R * self.AC_T / self.AC_F * ufl.ln(self.AC_Na_o / NV_Ith_S(y, 17))
        AV_i_NaCa = self.AC_K_NaCa * (ufl.exp(self.AC_sodium_calcium_exchanger_current_gamma * NV_Ith_S(y, 0) * self.AC_F / (self.AC_R * self.AC_T)) * pow(NV_Ith_S(y, 17), 3.0) * self.AC_Ca_o - ufl.exp((self.AC_sodium_calcium_exchanger_current_gamma - 1.0) * NV_Ith_S(y, 0) * self.AC_F / (self.AC_R * self.AC_T)) * pow(self.AC_Na_o, 3.0) * NV_Ith_S(y, 13) * self.AC_alpha) / ((pow(self.AC_Km_Nai, 3.0) + pow(self.AC_Na_o, 3.0)) * (self.AC_Km_Ca + self.AC_Ca_o) * (1.0 + self.AC_K_sat * ufl.exp((self.AC_sodium_calcium_exchanger_current_gamma - 1.0) * NV_Ith_S(y, 0) * self.AC_F / (self.AC_R * self.AC_T))))
        AV_ddt_Ca_i_total = (-(AV_i_b_Ca + AV_i_p_Ca - 2.0 * AV_i_NaCa)) * self.AC_Cm / (2.0 * self.AC_V_c * self.AC_F) + (AV_i_leak - AV_i_up) * self.AC_V_sr / self.AC_V_c + AV_i_xfer
        AV_ddt_Ca_ss_total = (-AV_i_CaL) * self.AC_Cm / (2.0 * self.AC_V_ss * self.AC_F) + AV_i_rel * self.AC_V_sr / self.AC_V_ss - AV_i_xfer * self.AC_V_c / self.AC_V_ss
        AV_i_Na = self.AC_g_Na * pow(NV_Ith_S(y, 4), 3.0) * NV_Ith_S(y, 5) * NV_Ith_S(y, 6) * (NV_Ith_S(y, 0) - AV_E_Na)
        AV_xK1_inf = AV_alpha_K1 / (AV_alpha_K1 + AV_beta_K1)
        AV_i_Ks = self.AC_g_Ks * pow(NV_Ith_S(y, 3), 2.0) * (NV_Ith_S(y, 0) - AV_E_Ks)
        AV_i_b_Na = self.AC_g_bna * (NV_Ith_S(y, 0) - AV_E_Na)
        ydot[13] = AV_ddt_Ca_i_total * AV_f_JCa_i_free
        ydot[15] = AV_ddt_Ca_ss_total * AV_f_JCa_ss_free
        AV_i_K1 = self.AC_g_K1 * AV_xK1_inf * np.sqrt(self.AC_K_o / 5.4) * (NV_Ith_S(y, 0) - AV_E_K)
        ydot[17] = (-(AV_i_Na + AV_i_b_Na + 3.0 * AV_i_NaK + 3.0 * AV_i_NaCa)) / (self.AC_V_c * self.AC_F) * self.AC_Cm
        ydot[0] = (-(AV_i_K1 + AV_i_to + AV_i_Kr + AV_i_Ks + AV_i_CaL + AV_i_NaK + AV_i_Na + AV_i_b_Na + AV_i_NaCa + AV_i_b_Ca + AV_i_p_K + AV_i_p_Ca))
        ydot[18] = (-(AV_i_K1 + AV_i_to + AV_i_Kr + AV_i_Ks + AV_i_p_K - 2.0 * AV_i_NaK)) / (self.AC_V_c * self.AC_F) * self.AC_Cm                

        return ydot

    def f_stiff(self, y):
        ydot = [None]*self.size
        
        # /* fast_sodium_current_m_gate */
        AV_alpha_m = 1.0 / (1.0 + ufl.exp(((-60.0) - NV_Ith_S(y, 0)) / 5.0))
        AV_beta_m = 0.1 / (1.0 + ufl.exp((NV_Ith_S(y, 0) + 35.0) / 5.0)) + 0.1 / (1.0 + ufl.exp((NV_Ith_S(y, 0) - 50.0) / 200.0))
        AV_m_inf = 1.0 / pow(1.0 + ufl.exp(((-56.86) - NV_Ith_S(y, 0)) / 9.03), 2.0)
        AV_tau_m = 1.0 * AV_alpha_m * AV_beta_m
        ydot[4] = (AV_m_inf - NV_Ith_S(y, 4)) / AV_tau_m

        # AV_alpha_j = ufl.conditional(ufl.lt(NV_Ith_S(y, 0) , (-40.0)) , ((-25428.0) * ufl.exp(0.2444 * NV_Ith_S(y, 0)) - 6.948e-06 * ufl.exp((-0.04391) * NV_Ith_S(y, 0))) * (NV_Ith_S(y, 0) + 37.78) / 1.0 / (1.0 + ufl.exp(0.311 * (NV_Ith_S(y, 0) + 79.23))) , 0.0)
        # AV_beta_j = ufl.conditional(ufl.lt(NV_Ith_S(y, 0) , (-40.0)) , 0.02424 * ufl.exp((-0.01052) * NV_Ith_S(y, 0)) / (1.0 + ufl.exp((-0.1378) * (NV_Ith_S(y, 0) + 40.14))) , 0.6 * ufl.exp(0.057 * NV_Ith_S(y, 0)) / (1.0 + ufl.exp((-0.1) * (NV_Ith_S(y, 0) + 32.0))))
        # AV_j_inf = 1.0 / pow(1.0 + ufl.exp((NV_Ith_S(y, 0) + 71.55) / 7.43), 2.0)
        # AV_tau_j = 1.0 / (AV_alpha_j + AV_beta_j)
        # ydot[6] = (AV_j_inf - NV_Ith_S(y, 6)) / AV_tau_j
 
        return ydot
    
    def u_stiff_coeffs(self, y):
        yinf = [None]*self.size
        tau = [None]*self.size
        
        # /* fast_sodium_current_m_gate */
        AV_alpha_m = 1.0 / (1.0 + ufl.exp(((-60.0) - NV_Ith_S(y, 0)) / 5.0))
        AV_beta_m = 0.1 / (1.0 + ufl.exp((NV_Ith_S(y, 0) + 35.0) / 5.0)) + 0.1 / (1.0 + ufl.exp((NV_Ith_S(y, 0) - 50.0) / 200.0))
        yinf[4] = 1.0 / pow(1.0 + ufl.exp(((-56.86) - NV_Ith_S(y, 0)) / 9.03), 2.0)
        tau[4] = 1.0 * AV_alpha_m * AV_beta_m     

        # AV_alpha_j = ufl.conditional(ufl.lt(NV_Ith_S(y, 0) , (-40.0)) , ((-25428.0) * ufl.exp(0.2444 * NV_Ith_S(y, 0)) - 6.948e-06 * ufl.exp((-0.04391) * NV_Ith_S(y, 0))) * (NV_Ith_S(y, 0) + 37.78) / 1.0 / (1.0 + ufl.exp(0.311 * (NV_Ith_S(y, 0) + 79.23))) , 0.0)
        # AV_beta_j = ufl.conditional(ufl.lt(NV_Ith_S(y, 0) , (-40.0)) , 0.02424 * ufl.exp((-0.01052) * NV_Ith_S(y, 0)) / (1.0 + ufl.exp((-0.1378) * (NV_Ith_S(y, 0) + 40.14))) , 0.6 * ufl.exp(0.057 * NV_Ith_S(y, 0)) / (1.0 + ufl.exp((-0.1) * (NV_Ith_S(y, 0) + 32.0))))
        # yinf[6] = 1.0 / pow(1.0 + ufl.exp((NV_Ith_S(y, 0) + 71.55) / 7.43), 2.0)
        # tau[6] = 1.0 / (AV_alpha_j + AV_beta_j)

        return yinf,tau
    
    def f_nonstiff(self, y):
        ydot = [None]*self.size
            
        # /* L_type_Ca_current_d_gate */
        AV_alpha_d = 1.4 / (1.0 + ufl.exp(((-35.0) - NV_Ith_S(y, 0)) / 13.0)) + 0.25
        AV_beta_d = 1.4 / (1.0 + ufl.exp((NV_Ith_S(y, 0) + 5.0) / 5.0))
        AV_d_inf = 1.0 / (1.0 + ufl.exp(((-8.0) - NV_Ith_S(y, 0)) / 7.5))
        AV_gamma_d = 1.0 / (1.0 + ufl.exp((50.0 - NV_Ith_S(y, 0)) / 20.0))
        AV_tau_d = 1.0 * AV_alpha_d * AV_beta_d + AV_gamma_d
        ydot[7] = (AV_d_inf - NV_Ith_S(y, 7)) / AV_tau_d
        
        # /* L_type_Ca_current_f2_gate */
        AV_f2_inf = 0.67 / (1.0 + ufl.exp((NV_Ith_S(y, 0) + 35.0) / 7.0)) + 0.33
        AV_tau_f2 = 562.0 * ufl.exp((-pow(NV_Ith_S(y, 0) + 27.0, 2.0)) / 240.0) + 31.0 / (1.0 + ufl.exp((25.0 - NV_Ith_S(y, 0)) / 10.0)) + 80.0 / (1.0 + ufl.exp((NV_Ith_S(y, 0) + 30.0) / 10.0))
        ydot[9] = (AV_f2_inf - NV_Ith_S(y, 9)) / AV_tau_f2
        
        # /* L_type_Ca_current_fCass_gate */
        AV_fCass_inf = 0.6 / (1.0 + pow(NV_Ith_S(y, 15) / 0.05, 2.0)) + 0.4
        AV_tau_fCass = 80.0 / (1.0 + pow(NV_Ith_S(y, 15) / 0.05, 2.0)) + 2.0
        ydot[10] = (AV_fCass_inf - NV_Ith_S(y, 10)) / AV_tau_fCass
        
        # /* L_type_Ca_current_f_gate */
        AV_f_inf = 1.0 / (1.0 + ufl.exp((NV_Ith_S(y, 0) + 20.0) / 7.0))
        AV_tau_f = 1102.5 * ufl.exp((-pow(NV_Ith_S(y, 0) + 27.0, 2.0)) / 225.0) + 200.0 / (1.0 + ufl.exp((13.0 - NV_Ith_S(y, 0)) / 10.0)) + 180.0 / (1.0 + ufl.exp((NV_Ith_S(y, 0) + 30.0) / 10.0)) + 20.0
        ydot[8] = (AV_f_inf - NV_Ith_S(y, 8)) / AV_tau_f
        
        # /* calcium_pump_current */
        AV_i_p_Ca = self.AC_g_pCa * NV_Ith_S(y, 13) / (NV_Ith_S(y, 13) + self.AC_K_pCa)
        
        # /* fast_sodium_current_h_gate */
        AV_alpha_h = ufl.conditional(ufl.lt(NV_Ith_S(y, 0) , (-40.0)) , 0.057 * ufl.exp((-(NV_Ith_S(y, 0) + 80.0)) / 6.8) , 0.0)
        AV_beta_h = ufl.conditional(ufl.lt(NV_Ith_S(y, 0) , (-40.0)) , 2.7 * ufl.exp(0.079 * NV_Ith_S(y, 0)) + 310000.0 * ufl.exp(0.3485 * NV_Ith_S(y, 0)) , 0.77 / (0.13 * (1.0 + ufl.exp((NV_Ith_S(y, 0) + 10.66) / (-11.1)))))
        AV_h_inf = 1.0 / pow(1.0 + ufl.exp((NV_Ith_S(y, 0) + 71.55) / 7.43), 2.0)
        AV_tau_h = 1.0 / (AV_alpha_h + AV_beta_h)
        ydot[5] = (AV_h_inf - NV_Ith_S(y, 5)) / AV_tau_h
        
        # /* fast_sodium_current_j_gate */
        AV_alpha_j = ufl.conditional(ufl.lt(NV_Ith_S(y, 0) , (-40.0)) , ((-25428.0) * ufl.exp(0.2444 * NV_Ith_S(y, 0)) - 6.948e-06 * ufl.exp((-0.04391) * NV_Ith_S(y, 0))) * (NV_Ith_S(y, 0) + 37.78) / 1.0 / (1.0 + ufl.exp(0.311 * (NV_Ith_S(y, 0) + 79.23))) , 0.0)
        AV_beta_j = ufl.conditional(ufl.lt(NV_Ith_S(y, 0) , (-40.0)) , 0.02424 * ufl.exp((-0.01052) * NV_Ith_S(y, 0)) / (1.0 + ufl.exp((-0.1378) * (NV_Ith_S(y, 0) + 40.14))) , 0.6 * ufl.exp(0.057 * NV_Ith_S(y, 0)) / (1.0 + ufl.exp((-0.1) * (NV_Ith_S(y, 0) + 32.0))))
        AV_j_inf = 1.0 / pow(1.0 + ufl.exp((NV_Ith_S(y, 0) + 71.55) / 7.43), 2.0)
        AV_tau_j = 1.0 / (AV_alpha_j + AV_beta_j)
        ydot[6] = (AV_j_inf - NV_Ith_S(y, 6)) / AV_tau_j
        
        # # /* fast_sodium_current_m_gate */
        # AV_alpha_m = 1.0 / (1.0 + ufl.exp(((-60.0) - NV_Ith_S(y, 0)) / 5.0))
        # AV_beta_m = 0.1 / (1.0 + ufl.exp((NV_Ith_S(y, 0) + 35.0) / 5.0)) + 0.1 / (1.0 + ufl.exp((NV_Ith_S(y, 0) - 50.0) / 200.0))
        # AV_m_inf = 1.0 / pow(1.0 + ufl.exp(((-56.86) - NV_Ith_S(y, 0)) / 9.03), 2.0)
        # AV_tau_m = 1.0 * AV_alpha_m * AV_beta_m
        # ydot[4] = (AV_m_inf - NV_Ith_S(y, 4)) / AV_tau_m
        
        # /* rapid_time_dependent_potassium_current_Xr1_gate */
        AV_alpha_xr1 = 450.0 / (1.0 + ufl.exp(((-45.0) - NV_Ith_S(y, 0)) / 10.0))
        AV_beta_xr1 = 6.0 / (1.0 + ufl.exp((NV_Ith_S(y, 0) + 30.0) / 11.5))
        AV_xr1_inf = 1.0 / (1.0 + ufl.exp(((-26.0) - NV_Ith_S(y, 0)) / 7.0))
        AV_tau_xr1 = 1.0 * AV_alpha_xr1 * AV_beta_xr1
        ydot[1] = (AV_xr1_inf - NV_Ith_S(y, 1)) / AV_tau_xr1
        
        # /* rapid_time_dependent_potassium_current_Xr2_gate */
        AV_alpha_xr2 = 3.0 / (1.0 + ufl.exp(((-60.0) - NV_Ith_S(y, 0)) / 20.0))
        AV_beta_xr2 = 1.12 / (1.0 + ufl.exp((NV_Ith_S(y, 0) - 60.0) / 20.0))
        AV_xr2_inf = 1.0 / (1.0 + ufl.exp((NV_Ith_S(y, 0) + 88.0) / 24.0))
        AV_tau_xr2 = 1.0 * AV_alpha_xr2 * AV_beta_xr2
        ydot[2] = (AV_xr2_inf - NV_Ith_S(y, 2)) / AV_tau_xr2
        
        # /* slow_time_dependent_potassium_current_Xs_gate */
        AV_alpha_xs = 1400.0 / ufl.sqrt(1.0 + ufl.exp((5.0 - NV_Ith_S(y, 0)) / 6.0))
        AV_beta_xs = 1.0 / (1.0 + ufl.exp((NV_Ith_S(y, 0) - 35.0) / 15.0))
        AV_xs_inf = 1.0 / (1.0 + ufl.exp(((-5.0) - NV_Ith_S(y, 0)) / 14.0))
        AV_tau_xs = 1.0 * AV_alpha_xs * AV_beta_xs + 80.0
        ydot[3] = (AV_xs_inf - NV_Ith_S(y, 3)) / AV_tau_xs
        
        # /* transient_outward_current_r_gate */
        AV_r_inf = 1.0 / (1.0 + ufl.exp((20.0 - NV_Ith_S(y, 0)) / 6.0))
        AV_tau_r = 9.5 * ufl.exp((-pow(NV_Ith_S(y, 0) + 40.0, 2.0)) / 1800.0) + 0.8
        ydot[12] = (AV_r_inf - NV_Ith_S(y, 12)) / AV_tau_r
        
        # /* transient_outward_current_s_gate */
        AV_s_inf = 1.0 / (1.0 + ufl.exp((NV_Ith_S(y, 0) + 20.0) / 5.0))
        AV_tau_s = 85.0 * ufl.exp((-pow(NV_Ith_S(y, 0) + 45.0, 2.0)) / 320.0) + 5.0 / (1.0 + ufl.exp((NV_Ith_S(y, 0) - 20.0) / 5.0)) + 3.0
        ydot[11] = (AV_s_inf - NV_Ith_S(y, 11)) / AV_tau_s
        
        # /* calcium_dynamics */
        AV_f_JCa_i_free = 1.0 / (1.0 + self.AC_Buf_c * self.AC_K_buf_c / pow(NV_Ith_S(y, 13) + self.AC_K_buf_c, 2.0))
        AV_f_JCa_sr_free = 1.0 / (1.0 + self.AC_Buf_sr * self.AC_K_buf_sr / pow(NV_Ith_S(y, 14) + self.AC_K_buf_sr, 2.0))
        AV_f_JCa_ss_free = 1.0 / (1.0 + self.AC_Buf_ss * self.AC_K_buf_ss / pow(NV_Ith_S(y, 15) + self.AC_K_buf_ss, 2.0))
        AV_i_leak = self.AC_V_leak * (NV_Ith_S(y, 14) - NV_Ith_S(y, 13))
        AV_i_up = self.AC_Vmax_up / (1.0 + pow(self.AC_K_up, 2.0) / pow(NV_Ith_S(y, 13), 2.0))
        AV_i_xfer = self.AC_V_xfer * (NV_Ith_S(y, 15) - NV_Ith_S(y, 13))
        AV_kcasr = self.AC_max_sr - (self.AC_max_sr - self.AC_min_sr) / (1.0 + pow(self.AC_EC / NV_Ith_S(y, 14), 2.0))
        AV_k1 = self.AC_k1_prime / AV_kcasr
        AV_k2 = self.AC_k2_prime * AV_kcasr
        AV_O = AV_k1 * pow(NV_Ith_S(y, 15), 2.0) * NV_Ith_S(y, 16) / (self.AC_k3 + AV_k1 * pow(NV_Ith_S(y, 15), 2.0))
        ydot[16] = (-AV_k2) * NV_Ith_S(y, 15) * NV_Ith_S(y, 16) + self.AC_k4 * (1.0 - NV_Ith_S(y, 16))
        AV_i_rel = self.AC_V_rel * AV_O * (NV_Ith_S(y, 14) - NV_Ith_S(y, 15))
        AV_ddt_Ca_sr_total = AV_i_up - (AV_i_rel + AV_i_leak)
        ydot[14] = AV_ddt_Ca_sr_total * AV_f_JCa_sr_free
        
        # /* reversal_potentials */
        AV_E_Ca = 0.5 * self.AC_R * self.AC_T / self.AC_F * ufl.ln(self.AC_Ca_o / NV_Ith_S(y, 13))
        AV_E_K = self.AC_R * self.AC_T / self.AC_F * ufl.ln(self.AC_K_o / NV_Ith_S(y, 18))
        
        # /* sodium_potassium_pump_current */
        AV_i_NaK = self.AC_P_NaK * self.AC_K_o / (self.AC_K_o + self.AC_K_mk) * NV_Ith_S(y, 17) / (NV_Ith_S(y, 17) + self.AC_K_mNa) / (1.0 + 0.1245 * ufl.exp((-0.1) * NV_Ith_S(y, 0) * self.AC_F / (self.AC_R * self.AC_T)) + 0.0353 * ufl.exp((-NV_Ith_S(y, 0)) * self.AC_F / (self.AC_R * self.AC_T)))
        
        # /* transient_outward_current */
        AV_i_to = self.AC_g_to * NV_Ith_S(y, 12) * NV_Ith_S(y, 11) * (NV_Ith_S(y, 0) - AV_E_K)
        
        # /* *remaining* */
        AV_i_CaL = self.AC_g_CaL * NV_Ith_S(y, 7) * NV_Ith_S(y, 8) * NV_Ith_S(y, 9) * NV_Ith_S(y, 10) * 4.0 * (NV_Ith_S(y, 0) - 15.0) * pow(self.AC_F, 2.0) / (self.AC_R * self.AC_T) * (0.25 * NV_Ith_S(y, 15) * ufl.exp(2.0 * (NV_Ith_S(y, 0) - 15.0) * self.AC_F / (self.AC_R * self.AC_T)) - self.AC_Ca_o) / (ufl.exp(2.0 * (NV_Ith_S(y, 0) - 15.0) * self.AC_F / (self.AC_R * self.AC_T)) - 1.0)
        AV_i_b_Ca = self.AC_g_bca * (NV_Ith_S(y, 0) - AV_E_Ca)
        AV_alpha_K1 = 0.1 / (1.0 + ufl.exp(0.06 * (NV_Ith_S(y, 0) - AV_E_K - 200.0)))
        AV_beta_K1 = (3.0 * ufl.exp(0.0002 * (NV_Ith_S(y, 0) - AV_E_K + 100.0)) + ufl.exp(0.1 * (NV_Ith_S(y, 0) - AV_E_K - 10.0))) / (1.0 + ufl.exp((-0.5) * (NV_Ith_S(y, 0) - AV_E_K)))
        AV_i_p_K = self.AC_g_pK * (NV_Ith_S(y, 0) - AV_E_K) / (1.0 + ufl.exp((25.0 - NV_Ith_S(y, 0)) / 5.98))
        AV_i_Kr = self.AC_g_Kr * np.sqrt(self.AC_K_o / 5.4) * NV_Ith_S(y, 1) * NV_Ith_S(y, 2) * (NV_Ith_S(y, 0) - AV_E_K)
        AV_E_Ks = self.AC_R * self.AC_T / self.AC_F * ufl.ln((self.AC_K_o + self.AC_P_kna * self.AC_Na_o) / (NV_Ith_S(y, 18) + self.AC_P_kna * NV_Ith_S(y, 17)))
        AV_E_Na = self.AC_R * self.AC_T / self.AC_F * ufl.ln(self.AC_Na_o / NV_Ith_S(y, 17))
        AV_i_NaCa = self.AC_K_NaCa * (ufl.exp(self.AC_sodium_calcium_exchanger_current_gamma * NV_Ith_S(y, 0) * self.AC_F / (self.AC_R * self.AC_T)) * pow(NV_Ith_S(y, 17), 3.0) * self.AC_Ca_o - ufl.exp((self.AC_sodium_calcium_exchanger_current_gamma - 1.0) * NV_Ith_S(y, 0) * self.AC_F / (self.AC_R * self.AC_T)) * pow(self.AC_Na_o, 3.0) * NV_Ith_S(y, 13) * self.AC_alpha) / ((pow(self.AC_Km_Nai, 3.0) + pow(self.AC_Na_o, 3.0)) * (self.AC_Km_Ca + self.AC_Ca_o) * (1.0 + self.AC_K_sat * ufl.exp((self.AC_sodium_calcium_exchanger_current_gamma - 1.0) * NV_Ith_S(y, 0) * self.AC_F / (self.AC_R * self.AC_T))))
        AV_ddt_Ca_i_total = (-(AV_i_b_Ca + AV_i_p_Ca - 2.0 * AV_i_NaCa)) * self.AC_Cm / (2.0 * self.AC_V_c * self.AC_F) + (AV_i_leak - AV_i_up) * self.AC_V_sr / self.AC_V_c + AV_i_xfer
        AV_ddt_Ca_ss_total = (-AV_i_CaL) * self.AC_Cm / (2.0 * self.AC_V_ss * self.AC_F) + AV_i_rel * self.AC_V_sr / self.AC_V_ss - AV_i_xfer * self.AC_V_c / self.AC_V_ss
        AV_i_Na = self.AC_g_Na * pow(NV_Ith_S(y, 4), 3.0) * NV_Ith_S(y, 5) * NV_Ith_S(y, 6) * (NV_Ith_S(y, 0) - AV_E_Na)
        AV_xK1_inf = AV_alpha_K1 / (AV_alpha_K1 + AV_beta_K1)
        AV_i_Ks = self.AC_g_Ks * pow(NV_Ith_S(y, 3), 2.0) * (NV_Ith_S(y, 0) - AV_E_Ks)
        AV_i_b_Na = self.AC_g_bna * (NV_Ith_S(y, 0) - AV_E_Na)
        ydot[13] = AV_ddt_Ca_i_total * AV_f_JCa_i_free
        ydot[15] = AV_ddt_Ca_ss_total * AV_f_JCa_ss_free
        AV_i_K1 = self.AC_g_K1 * AV_xK1_inf * np.sqrt(self.AC_K_o / 5.4) * (NV_Ith_S(y, 0) - AV_E_K)
        ydot[17] = (-(AV_i_Na + AV_i_b_Na + 3.0 * AV_i_NaK + 3.0 * AV_i_NaCa)) / (self.AC_V_c * self.AC_F) * self.AC_Cm
        ydot[0] = (-(AV_i_K1 + AV_i_to + AV_i_Kr + AV_i_Ks + AV_i_CaL + AV_i_NaK + AV_i_Na + AV_i_b_Na + AV_i_NaCa + AV_i_b_Ca + AV_i_p_K + AV_i_p_Ca))
        ydot[18] = (-(AV_i_K1 + AV_i_to + AV_i_Kr + AV_i_Ks + AV_i_p_K - 2.0 * AV_i_NaK)) / (self.AC_V_c * self.AC_F) * self.AC_Cm                

        return ydot
    
    def f_expl(self, y):
        ydot = [None]*self.size

        # /* calcium_dynamics */
        AV_f_JCa_i_free = 1.0 / (1.0 + self.AC_Buf_c * self.AC_K_buf_c / pow(NV_Ith_S(y, 13) + self.AC_K_buf_c, 2.0))
        AV_f_JCa_sr_free = 1.0 / (1.0 + self.AC_Buf_sr * self.AC_K_buf_sr / pow(NV_Ith_S(y, 14) + self.AC_K_buf_sr, 2.0))
        AV_f_JCa_ss_free = 1.0 / (1.0 + self.AC_Buf_ss * self.AC_K_buf_ss / pow(NV_Ith_S(y, 15) + self.AC_K_buf_ss, 2.0))
        AV_i_leak = self.AC_V_leak * (NV_Ith_S(y, 14) - NV_Ith_S(y, 13))
        AV_i_up = self.AC_Vmax_up / (1.0 + pow(self.AC_K_up, 2.0) / pow(NV_Ith_S(y, 13), 2.0))
        AV_i_xfer = self.AC_V_xfer * (NV_Ith_S(y, 15) - NV_Ith_S(y, 13))
        AV_kcasr = self.AC_max_sr - (self.AC_max_sr - self.AC_min_sr) / (1.0 + pow(self.AC_EC / NV_Ith_S(y, 14), 2.0))
        AV_k1 = self.AC_k1_prime / AV_kcasr
        AV_k2 = self.AC_k2_prime * AV_kcasr
        AV_O = AV_k1 * pow(NV_Ith_S(y, 15), 2.0) * NV_Ith_S(y, 16) / (self.AC_k3 + AV_k1 * pow(NV_Ith_S(y, 15), 2.0))
        ydot[16] = (-AV_k2) * NV_Ith_S(y, 15) * NV_Ith_S(y, 16) + self.AC_k4 * (1.0 - NV_Ith_S(y, 16))
        AV_i_rel = self.AC_V_rel * AV_O * (NV_Ith_S(y, 14) - NV_Ith_S(y, 15))
        AV_ddt_Ca_sr_total = AV_i_up - (AV_i_rel + AV_i_leak)
        ydot[14] = AV_ddt_Ca_sr_total * AV_f_JCa_sr_free
        
        # /* reversal_potentials */
        AV_E_Ca = 0.5 * self.AC_R * self.AC_T / self.AC_F * ufl.ln(self.AC_Ca_o / NV_Ith_S(y, 13))
        AV_E_K = self.AC_R * self.AC_T / self.AC_F * ufl.ln(self.AC_K_o / NV_Ith_S(y, 18))
        
        # /* sodium_potassium_pump_current */
        AV_i_NaK = self.AC_P_NaK * self.AC_K_o / (self.AC_K_o + self.AC_K_mk) * NV_Ith_S(y, 17) / (NV_Ith_S(y, 17) + self.AC_K_mNa) / (1.0 + 0.1245 * ufl.exp((-0.1) * NV_Ith_S(y, 0) * self.AC_F / (self.AC_R * self.AC_T)) + 0.0353 * ufl.exp((-NV_Ith_S(y, 0)) * self.AC_F / (self.AC_R * self.AC_T)))
        
        # /* transient_outward_current */
        AV_i_to = self.AC_g_to * NV_Ith_S(y, 12) * NV_Ith_S(y, 11) * (NV_Ith_S(y, 0) - AV_E_K)
        
        # /* calcium_pump_current */
        AV_i_p_Ca = self.AC_g_pCa * NV_Ith_S(y, 13) / (NV_Ith_S(y, 13) + self.AC_K_pCa)

        # /* *remaining* */
        AV_i_CaL = self.AC_g_CaL * NV_Ith_S(y, 7) * NV_Ith_S(y, 8) * NV_Ith_S(y, 9) * NV_Ith_S(y, 10) * 4.0 * (NV_Ith_S(y, 0) - 15.0) * pow(self.AC_F, 2.0) / (self.AC_R * self.AC_T) * (0.25 * NV_Ith_S(y, 15) * ufl.exp(2.0 * (NV_Ith_S(y, 0) - 15.0) * self.AC_F / (self.AC_R * self.AC_T)) - self.AC_Ca_o) / (ufl.exp(2.0 * (NV_Ith_S(y, 0) - 15.0) * self.AC_F / (self.AC_R * self.AC_T)) - 1.0)
        AV_i_b_Ca = self.AC_g_bca * (NV_Ith_S(y, 0) - AV_E_Ca)
        AV_alpha_K1 = 0.1 / (1.0 + ufl.exp(0.06 * (NV_Ith_S(y, 0) - AV_E_K - 200.0)))
        AV_beta_K1 = (3.0 * ufl.exp(0.0002 * (NV_Ith_S(y, 0) - AV_E_K + 100.0)) + ufl.exp(0.1 * (NV_Ith_S(y, 0) - AV_E_K - 10.0))) / (1.0 + ufl.exp((-0.5) * (NV_Ith_S(y, 0) - AV_E_K)))
        AV_i_p_K = self.AC_g_pK * (NV_Ith_S(y, 0) - AV_E_K) / (1.0 + ufl.exp((25.0 - NV_Ith_S(y, 0)) / 5.98))
        AV_i_Kr = self.AC_g_Kr * np.sqrt(self.AC_K_o / 5.4) * NV_Ith_S(y, 1) * NV_Ith_S(y, 2) * (NV_Ith_S(y, 0) - AV_E_K)
        AV_E_Ks = self.AC_R * self.AC_T / self.AC_F * ufl.ln((self.AC_K_o + self.AC_P_kna * self.AC_Na_o) / (NV_Ith_S(y, 18) + self.AC_P_kna * NV_Ith_S(y, 17)))
        AV_E_Na = self.AC_R * self.AC_T / self.AC_F * ufl.ln(self.AC_Na_o / NV_Ith_S(y, 17))
        AV_i_NaCa = self.AC_K_NaCa * (ufl.exp(self.AC_sodium_calcium_exchanger_current_gamma * NV_Ith_S(y, 0) * self.AC_F / (self.AC_R * self.AC_T)) * pow(NV_Ith_S(y, 17), 3.0) * self.AC_Ca_o - ufl.exp((self.AC_sodium_calcium_exchanger_current_gamma - 1.0) * NV_Ith_S(y, 0) * self.AC_F / (self.AC_R * self.AC_T)) * pow(self.AC_Na_o, 3.0) * NV_Ith_S(y, 13) * self.AC_alpha) / ((pow(self.AC_Km_Nai, 3.0) + pow(self.AC_Na_o, 3.0)) * (self.AC_Km_Ca + self.AC_Ca_o) * (1.0 + self.AC_K_sat * ufl.exp((self.AC_sodium_calcium_exchanger_current_gamma - 1.0) * NV_Ith_S(y, 0) * self.AC_F / (self.AC_R * self.AC_T))))
        AV_ddt_Ca_i_total = (-(AV_i_b_Ca + AV_i_p_Ca - 2.0 * AV_i_NaCa)) * self.AC_Cm / (2.0 * self.AC_V_c * self.AC_F) + (AV_i_leak - AV_i_up) * self.AC_V_sr / self.AC_V_c + AV_i_xfer
        AV_ddt_Ca_ss_total = (-AV_i_CaL) * self.AC_Cm / (2.0 * self.AC_V_ss * self.AC_F) + AV_i_rel * self.AC_V_sr / self.AC_V_ss - AV_i_xfer * self.AC_V_c / self.AC_V_ss
        AV_i_Na = self.AC_g_Na * pow(NV_Ith_S(y, 4), 3.0) * NV_Ith_S(y, 5) * NV_Ith_S(y, 6) * (NV_Ith_S(y, 0) - AV_E_Na)
        AV_xK1_inf = AV_alpha_K1 / (AV_alpha_K1 + AV_beta_K1)
        AV_i_Ks = self.AC_g_Ks * pow(NV_Ith_S(y, 3), 2.0) * (NV_Ith_S(y, 0) - AV_E_Ks)
        AV_i_b_Na = self.AC_g_bna * (NV_Ith_S(y, 0) - AV_E_Na)
        ydot[13] = AV_ddt_Ca_i_total * AV_f_JCa_i_free
        ydot[15] = AV_ddt_Ca_ss_total * AV_f_JCa_ss_free
        AV_i_K1 = self.AC_g_K1 * AV_xK1_inf * np.sqrt(self.AC_K_o / 5.4) * (NV_Ith_S(y, 0) - AV_E_K)
        ydot[17] = (-(AV_i_Na + AV_i_b_Na + 3.0 * AV_i_NaK + 3.0 * AV_i_NaCa)) / (self.AC_V_c * self.AC_F) * self.AC_Cm
        ydot[0] = (-(AV_i_K1 + AV_i_to + AV_i_Kr + AV_i_Ks + AV_i_CaL + AV_i_NaK + AV_i_Na + AV_i_b_Na + AV_i_NaCa + AV_i_b_Ca + AV_i_p_K + AV_i_p_Ca))
        ydot[18] = (-(AV_i_K1 + AV_i_to + AV_i_Kr + AV_i_Ks + AV_i_p_K - 2.0 * AV_i_NaK)) / (self.AC_V_c * self.AC_F) * self.AC_Cm   

        return ydot
    
    def u_exp_coeffs(self, y):
        yinf = [None]*self.size
        tau = [None]*self.size

        # /* L_type_Ca_current_d_gate */
        AV_alpha_d = 1.4 / (1.0 + ufl.exp(((-35.0) - NV_Ith_S(y, 0)) / 13.0)) + 0.25
        AV_beta_d = 1.4 / (1.0 + ufl.exp((NV_Ith_S(y, 0) + 5.0) / 5.0))
        yinf[7] = 1.0 / (1.0 + ufl.exp(((-8.0) - NV_Ith_S(y, 0)) / 7.5))
        AV_gamma_d = 1.0 / (1.0 + ufl.exp((50.0 - NV_Ith_S(y, 0)) / 20.0))
        tau[7] = 1.0 * AV_alpha_d * AV_beta_d + AV_gamma_d
        
        # /* L_type_Ca_current_f2_gate */
        yinf[9] = 0.67 / (1.0 + ufl.exp((NV_Ith_S(y, 0) + 35.0) / 7.0)) + 0.33
        tau[9] = 562.0 * ufl.exp((-pow(NV_Ith_S(y, 0) + 27.0, 2.0)) / 240.0) + 31.0 / (1.0 + ufl.exp((25.0 - NV_Ith_S(y, 0)) / 10.0)) + 80.0 / (1.0 + ufl.exp((NV_Ith_S(y, 0) + 30.0) / 10.0))
        
        # /* L_type_Ca_current_fCass_gate */
        yinf[10] = 0.6 / (1.0 + pow(NV_Ith_S(y, 15) / 0.05, 2.0)) + 0.4
        tau[10] = 80.0 / (1.0 + pow(NV_Ith_S(y, 15) / 0.05, 2.0)) + 2.0
        
        # /* L_type_Ca_current_f_gate */
        yinf[8] = 1.0 / (1.0 + ufl.exp((NV_Ith_S(y, 0) + 20.0) / 7.0))
        tau[8] = 1102.5 * ufl.exp((-pow(NV_Ith_S(y, 0) + 27.0, 2.0)) / 225.0) + 200.0 / (1.0 + ufl.exp((13.0 - NV_Ith_S(y, 0)) / 10.0)) + 180.0 / (1.0 + ufl.exp((NV_Ith_S(y, 0) + 30.0) / 10.0)) + 20.0
        
        # /* fast_sodium_current_h_gate */
        AV_alpha_h = ufl.conditional(ufl.lt(NV_Ith_S(y, 0) , (-40.0)) , 0.057 * ufl.exp((-(NV_Ith_S(y, 0) + 80.0)) / 6.8) , 0.0)
        AV_beta_h = ufl.conditional(ufl.lt(NV_Ith_S(y, 0) , (-40.0)) , 2.7 * ufl.exp(0.079 * NV_Ith_S(y, 0)) + 310000.0 * ufl.exp(0.3485 * NV_Ith_S(y, 0)) , 0.77 / (0.13 * (1.0 + ufl.exp((NV_Ith_S(y, 0) + 10.66) / (-11.1)))))
        yinf[5] = 1.0 / pow(1.0 + ufl.exp((NV_Ith_S(y, 0) + 71.55) / 7.43), 2.0)
        tau[5] = 1.0 / (AV_alpha_h + AV_beta_h)
        
        # /* fast_sodium_current_j_gate */
        AV_alpha_j = ufl.conditional(ufl.lt(NV_Ith_S(y, 0) , (-40.0)) , ((-25428.0) * ufl.exp(0.2444 * NV_Ith_S(y, 0)) - 6.948e-06 * ufl.exp((-0.04391) * NV_Ith_S(y, 0))) * (NV_Ith_S(y, 0) + 37.78) / 1.0 / (1.0 + ufl.exp(0.311 * (NV_Ith_S(y, 0) + 79.23))) , 0.0)
        AV_beta_j = ufl.conditional(ufl.lt(NV_Ith_S(y, 0) , (-40.0)) , 0.02424 * ufl.exp((-0.01052) * NV_Ith_S(y, 0)) / (1.0 + ufl.exp((-0.1378) * (NV_Ith_S(y, 0) + 40.14))) , 0.6 * ufl.exp(0.057 * NV_Ith_S(y, 0)) / (1.0 + ufl.exp((-0.1) * (NV_Ith_S(y, 0) + 32.0))))
        yinf[6] = 1.0 / pow(1.0 + ufl.exp((NV_Ith_S(y, 0) + 71.55) / 7.43), 2.0)
        tau[6] = 1.0 / (AV_alpha_j + AV_beta_j)
        
        # /* fast_sodium_current_m_gate */
        AV_alpha_m = 1.0 / (1.0 + ufl.exp(((-60.0) - NV_Ith_S(y, 0)) / 5.0))
        AV_beta_m = 0.1 / (1.0 + ufl.exp((NV_Ith_S(y, 0) + 35.0) / 5.0)) + 0.1 / (1.0 + ufl.exp((NV_Ith_S(y, 0) - 50.0) / 200.0))
        yinf[4] = 1.0 / pow(1.0 + ufl.exp(((-56.86) - NV_Ith_S(y, 0)) / 9.03), 2.0)
        tau[4] = 1.0 * AV_alpha_m * AV_beta_m
        
        # /* rapid_time_dependent_potassium_current_Xr1_gate */
        AV_alpha_xr1 = 450.0 / (1.0 + ufl.exp(((-45.0) - NV_Ith_S(y, 0)) / 10.0))
        AV_beta_xr1 = 6.0 / (1.0 + ufl.exp((NV_Ith_S(y, 0) + 30.0) / 11.5))
        yinf[1] = 1.0 / (1.0 + ufl.exp(((-26.0) - NV_Ith_S(y, 0)) / 7.0))
        tau[1] = 1.0 * AV_alpha_xr1 * AV_beta_xr1
        
        # /* rapid_time_dependent_potassium_current_Xr2_gate */
        AV_alpha_xr2 = 3.0 / (1.0 + ufl.exp(((-60.0) - NV_Ith_S(y, 0)) / 20.0))
        AV_beta_xr2 = 1.12 / (1.0 + ufl.exp((NV_Ith_S(y, 0) - 60.0) / 20.0))
        yinf[2] = 1.0 / (1.0 + ufl.exp((NV_Ith_S(y, 0) + 88.0) / 24.0))
        tau[2] = 1.0 * AV_alpha_xr2 * AV_beta_xr2
        
        # /* slow_time_dependent_potassium_current_Xs_gate */
        AV_alpha_xs = 1400.0 / ufl.sqrt(1.0 + ufl.exp((5.0 - NV_Ith_S(y, 0)) / 6.0))
        AV_beta_xs = 1.0 / (1.0 + ufl.exp((NV_Ith_S(y, 0) - 35.0) / 15.0))
        yinf[3] = 1.0 / (1.0 + ufl.exp(((-5.0) - NV_Ith_S(y, 0)) / 14.0))
        tau[3] = 1.0 * AV_alpha_xs * AV_beta_xs + 80.0
        
        # /* transient_outward_current_r_gate */
        yinf[12] = 1.0 / (1.0 + ufl.exp((20.0 - NV_Ith_S(y, 0)) / 6.0))
        tau[12] = 9.5 * ufl.exp((-pow(NV_Ith_S(y, 0) + 40.0, 2.0)) / 1800.0) + 0.8
        
        # /* transient_outward_current_s_gate */
        yinf[11] = 1.0 / (1.0 + ufl.exp((NV_Ith_S(y, 0) + 20.0) / 5.0))
        tau[11] = 85.0 * ufl.exp((-pow(NV_Ith_S(y, 0) + 45.0, 2.0)) / 320.0) + 5.0 / (1.0 + ufl.exp((NV_Ith_S(y, 0) - 20.0) / 5.0)) + 3.0

        return yinf,tau

    def initial_values(self):

        y0 = [None]*self.size
        
        y0[0] = -85.23
        y0[1] = 0.00621
        y0[2] = 0.4712
        y0[3] = 0.0095
        y0[4] = 0.00172
        y0[5] = 0.7444
        y0[6] = 0.7045
        y0[7] = 3.373e-05
        y0[8] = 0.7888
        y0[9] = 0.9755
        y0[10] = 0.9953
        y0[11] = 0.999998
        y0[12] = 2.42e-08
        y0[13] = 0.000126
        y0[14] = 3.64
        y0[15] = 0.00036
        y0[16] = 0.9073
        y0[17] = 8.604
        y0[18] = 136.89

        return y0