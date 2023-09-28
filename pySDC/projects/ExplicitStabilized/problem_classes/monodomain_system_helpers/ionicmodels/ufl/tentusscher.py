import numpy as np
import ufl
from pySDC.projects.ExplicitStabilized.problem_classes.monodomain_system_helpers.ionicmodels.ufl.ionicmodel import IonicModel
from pySDC.projects.ExplicitStabilized.problem_classes.monodomain_system_helpers.ionicmodels.ufl.ionicmodel import NV_Ith_S


# Stiff with rho_max ~ 950
class TenTusscher2006_epi(IonicModel):
    def __init__(self, scale):
        super(TenTusscher2006_epi, self).__init__(scale)

        self.size = 19

        self.AC_Cm = 1.0  # 185.0
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
        self.AC_stim_amplitude = -52.0
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

        # overall stiffness of f is 1000
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

        self.f_nonstiff_args = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]  # all
        self.f_stiff_args = [0, 4]
        self.f_expl_args = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]  # all
        self.f_exp_args = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15]

        self.f_nonstiff_indeces = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]  # all except 4
        self.f_stiff_indeces = [4]
        self.f_expl_indeces = [0, 13, 14, 15, 16, 17, 18]
        self.f_exp_indeces = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    def initial_values(self):
        y0 = [None] * self.size

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

    @property
    def f(self):
        y = self.y
        ydot = [None] * self.size

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
        AV_tau_f = (
            1102.5 * ufl.exp((-pow(NV_Ith_S(y, 0) + 27.0, 2.0)) / 225.0) + 200.0 / (1.0 + ufl.exp((13.0 - NV_Ith_S(y, 0)) / 10.0)) + 180.0 / (1.0 + ufl.exp((NV_Ith_S(y, 0) + 30.0) / 10.0)) + 20.0
        )
        ydot[8] = (AV_f_inf - NV_Ith_S(y, 8)) / AV_tau_f

        # /* fast_sodium_current_h_gate */
        AV_alpha_h = ufl.conditional(ufl.lt(NV_Ith_S(y, 0), (-40.0)), 0.057 * ufl.exp((-(NV_Ith_S(y, 0) + 80.0)) / 6.8), 0.0)
        AV_beta_h = ufl.conditional(
            ufl.lt(NV_Ith_S(y, 0), (-40.0)), 2.7 * ufl.exp(0.079 * NV_Ith_S(y, 0)) + 310000.0 * ufl.exp(0.3485 * NV_Ith_S(y, 0)), 0.77 / (0.13 * (1.0 + ufl.exp((NV_Ith_S(y, 0) + 10.66) / (-11.1))))
        )
        AV_h_inf = 1.0 / pow(1.0 + ufl.exp((NV_Ith_S(y, 0) + 71.55) / 7.43), 2.0)
        AV_tau_h = 1.0 / (AV_alpha_h + AV_beta_h)
        ydot[5] = (AV_h_inf - NV_Ith_S(y, 5)) / AV_tau_h

        # /* fast_sodium_current_j_gate */
        AV_alpha_j = ufl.conditional(
            ufl.lt(NV_Ith_S(y, 0), (-40.0)),
            ((-25428.0) * ufl.exp(0.2444 * NV_Ith_S(y, 0)) - 6.948e-06 * ufl.exp((-0.04391) * NV_Ith_S(y, 0))) * (NV_Ith_S(y, 0) + 37.78) / 1.0 / (1.0 + ufl.exp(0.311 * (NV_Ith_S(y, 0) + 79.23))),
            0.0,
        )
        AV_beta_j = ufl.conditional(
            ufl.lt(NV_Ith_S(y, 0), (-40.0)),
            0.02424 * ufl.exp((-0.01052) * NV_Ith_S(y, 0)) / (1.0 + ufl.exp((-0.1378) * (NV_Ith_S(y, 0) + 40.14))),
            0.6 * ufl.exp(0.057 * NV_Ith_S(y, 0)) / (1.0 + ufl.exp((-0.1) * (NV_Ith_S(y, 0) + 32.0))),
        )
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
        AV_i_NaK = (
            self.AC_P_NaK
            * self.AC_K_o
            / (self.AC_K_o + self.AC_K_mk)
            * NV_Ith_S(y, 17)
            / (NV_Ith_S(y, 17) + self.AC_K_mNa)
            / (1.0 + 0.1245 * ufl.exp((-0.1) * NV_Ith_S(y, 0) * self.AC_F / (self.AC_R * self.AC_T)) + 0.0353 * ufl.exp((-NV_Ith_S(y, 0)) * self.AC_F / (self.AC_R * self.AC_T)))
        )

        # /* transient_outward_current */
        AV_i_to = self.AC_g_to * NV_Ith_S(y, 12) * NV_Ith_S(y, 11) * (NV_Ith_S(y, 0) - AV_E_K)

        # /* calcium_pump_current */
        AV_i_p_Ca = self.AC_g_pCa * NV_Ith_S(y, 13) / (NV_Ith_S(y, 13) + self.AC_K_pCa)

        # /* *remaining* */
        AV_i_CaL = (
            self.AC_g_CaL
            * NV_Ith_S(y, 7)
            * NV_Ith_S(y, 8)
            * NV_Ith_S(y, 9)
            * NV_Ith_S(y, 10)
            * 4.0
            * (NV_Ith_S(y, 0) - 15.0)
            * pow(self.AC_F, 2.0)
            / (self.AC_R * self.AC_T)
            * (0.25 * NV_Ith_S(y, 15) * ufl.exp(2.0 * (NV_Ith_S(y, 0) - 15.0) * self.AC_F / (self.AC_R * self.AC_T)) - self.AC_Ca_o)
            / (ufl.exp(2.0 * (NV_Ith_S(y, 0) - 15.0) * self.AC_F / (self.AC_R * self.AC_T)) - 1.0)
        )
        AV_i_b_Ca = self.AC_g_bca * (NV_Ith_S(y, 0) - AV_E_Ca)
        AV_alpha_K1 = 0.1 / (1.0 + ufl.exp(0.06 * (NV_Ith_S(y, 0) - AV_E_K - 200.0)))
        AV_beta_K1 = (3.0 * ufl.exp(0.0002 * (NV_Ith_S(y, 0) - AV_E_K + 100.0)) + ufl.exp(0.1 * (NV_Ith_S(y, 0) - AV_E_K - 10.0))) / (1.0 + ufl.exp((-0.5) * (NV_Ith_S(y, 0) - AV_E_K)))
        AV_i_p_K = self.AC_g_pK * (NV_Ith_S(y, 0) - AV_E_K) / (1.0 + ufl.exp((25.0 - NV_Ith_S(y, 0)) / 5.98))
        AV_i_Kr = self.AC_g_Kr * np.sqrt(self.AC_K_o / 5.4) * NV_Ith_S(y, 1) * NV_Ith_S(y, 2) * (NV_Ith_S(y, 0) - AV_E_K)
        AV_E_Ks = self.AC_R * self.AC_T / self.AC_F * ufl.ln((self.AC_K_o + self.AC_P_kna * self.AC_Na_o) / (NV_Ith_S(y, 18) + self.AC_P_kna * NV_Ith_S(y, 17)))
        AV_E_Na = self.AC_R * self.AC_T / self.AC_F * ufl.ln(self.AC_Na_o / NV_Ith_S(y, 17))
        AV_i_NaCa = (
            self.AC_K_NaCa
            * (
                ufl.exp(self.AC_sodium_calcium_exchanger_current_gamma * NV_Ith_S(y, 0) * self.AC_F / (self.AC_R * self.AC_T)) * pow(NV_Ith_S(y, 17), 3.0) * self.AC_Ca_o
                - ufl.exp((self.AC_sodium_calcium_exchanger_current_gamma - 1.0) * NV_Ith_S(y, 0) * self.AC_F / (self.AC_R * self.AC_T)) * pow(self.AC_Na_o, 3.0) * NV_Ith_S(y, 13) * self.AC_alpha
            )
            / (
                (pow(self.AC_Km_Nai, 3.0) + pow(self.AC_Na_o, 3.0))
                * (self.AC_Km_Ca + self.AC_Ca_o)
                * (1.0 + self.AC_K_sat * ufl.exp((self.AC_sodium_calcium_exchanger_current_gamma - 1.0) * NV_Ith_S(y, 0) * self.AC_F / (self.AC_R * self.AC_T)))
            )
        )
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
        ydot[0] = self.scale * (-(AV_i_K1 + AV_i_to + AV_i_Kr + AV_i_Ks + AV_i_CaL + AV_i_NaK + AV_i_Na + AV_i_b_Na + AV_i_NaCa + AV_i_b_Ca + AV_i_p_K + AV_i_p_Ca))
        ydot[18] = (-(AV_i_K1 + AV_i_to + AV_i_Kr + AV_i_Ks + AV_i_p_K - 2.0 * AV_i_NaK)) / (self.AC_V_c * self.AC_F) * self.AC_Cm

        return self.expression_list(ydot)

    @property
    def f_expl(self):
        y = self.y
        ydot = [None] * self.size

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
        AV_i_NaK = (
            self.AC_P_NaK
            * self.AC_K_o
            / (self.AC_K_o + self.AC_K_mk)
            * NV_Ith_S(y, 17)
            / (NV_Ith_S(y, 17) + self.AC_K_mNa)
            / (1.0 + 0.1245 * ufl.exp((-0.1) * NV_Ith_S(y, 0) * self.AC_F / (self.AC_R * self.AC_T)) + 0.0353 * ufl.exp((-NV_Ith_S(y, 0)) * self.AC_F / (self.AC_R * self.AC_T)))
        )

        # /* transient_outward_current */
        AV_i_to = self.AC_g_to * NV_Ith_S(y, 12) * NV_Ith_S(y, 11) * (NV_Ith_S(y, 0) - AV_E_K)

        # /* calcium_pump_current */
        AV_i_p_Ca = self.AC_g_pCa * NV_Ith_S(y, 13) / (NV_Ith_S(y, 13) + self.AC_K_pCa)

        # /* *remaining* */
        AV_i_CaL = (
            self.AC_g_CaL
            * NV_Ith_S(y, 7)
            * NV_Ith_S(y, 8)
            * NV_Ith_S(y, 9)
            * NV_Ith_S(y, 10)
            * 4.0
            * (NV_Ith_S(y, 0) - 15.0)
            * pow(self.AC_F, 2.0)
            / (self.AC_R * self.AC_T)
            * (0.25 * NV_Ith_S(y, 15) * ufl.exp(2.0 * (NV_Ith_S(y, 0) - 15.0) * self.AC_F / (self.AC_R * self.AC_T)) - self.AC_Ca_o)
            / (ufl.exp(2.0 * (NV_Ith_S(y, 0) - 15.0) * self.AC_F / (self.AC_R * self.AC_T)) - 1.0)
        )
        AV_i_b_Ca = self.AC_g_bca * (NV_Ith_S(y, 0) - AV_E_Ca)
        AV_alpha_K1 = 0.1 / (1.0 + ufl.exp(0.06 * (NV_Ith_S(y, 0) - AV_E_K - 200.0)))
        AV_beta_K1 = (3.0 * ufl.exp(0.0002 * (NV_Ith_S(y, 0) - AV_E_K + 100.0)) + ufl.exp(0.1 * (NV_Ith_S(y, 0) - AV_E_K - 10.0))) / (1.0 + ufl.exp((-0.5) * (NV_Ith_S(y, 0) - AV_E_K)))
        AV_i_p_K = self.AC_g_pK * (NV_Ith_S(y, 0) - AV_E_K) / (1.0 + ufl.exp((25.0 - NV_Ith_S(y, 0)) / 5.98))
        AV_i_Kr = self.AC_g_Kr * np.sqrt(self.AC_K_o / 5.4) * NV_Ith_S(y, 1) * NV_Ith_S(y, 2) * (NV_Ith_S(y, 0) - AV_E_K)
        AV_E_Ks = self.AC_R * self.AC_T / self.AC_F * ufl.ln((self.AC_K_o + self.AC_P_kna * self.AC_Na_o) / (NV_Ith_S(y, 18) + self.AC_P_kna * NV_Ith_S(y, 17)))
        AV_E_Na = self.AC_R * self.AC_T / self.AC_F * ufl.ln(self.AC_Na_o / NV_Ith_S(y, 17))
        AV_i_NaCa = (
            self.AC_K_NaCa
            * (
                ufl.exp(self.AC_sodium_calcium_exchanger_current_gamma * NV_Ith_S(y, 0) * self.AC_F / (self.AC_R * self.AC_T)) * pow(NV_Ith_S(y, 17), 3.0) * self.AC_Ca_o
                - ufl.exp((self.AC_sodium_calcium_exchanger_current_gamma - 1.0) * NV_Ith_S(y, 0) * self.AC_F / (self.AC_R * self.AC_T)) * pow(self.AC_Na_o, 3.0) * NV_Ith_S(y, 13) * self.AC_alpha
            )
            / (
                (pow(self.AC_Km_Nai, 3.0) + pow(self.AC_Na_o, 3.0))
                * (self.AC_Km_Ca + self.AC_Ca_o)
                * (1.0 + self.AC_K_sat * ufl.exp((self.AC_sodium_calcium_exchanger_current_gamma - 1.0) * NV_Ith_S(y, 0) * self.AC_F / (self.AC_R * self.AC_T)))
            )
        )
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
        ydot[0] = self.scale * (-(AV_i_K1 + AV_i_to + AV_i_Kr + AV_i_Ks + AV_i_CaL + AV_i_NaK + AV_i_Na + AV_i_b_Na + AV_i_NaCa + AV_i_b_Ca + AV_i_p_K + AV_i_p_Ca))
        ydot[18] = (-(AV_i_K1 + AV_i_to + AV_i_Kr + AV_i_Ks + AV_i_p_K - 2.0 * AV_i_NaK)) / (self.AC_V_c * self.AC_F) * self.AC_Cm

        return self.expression_list(ydot)

    def f_exp_coeffs(self):
        y = self.y
        yinf = [None] * self.size
        tau = [None] * self.size

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
        AV_alpha_h = ufl.conditional(ufl.lt(NV_Ith_S(y, 0), (-40.0)), 0.057 * ufl.exp((-(NV_Ith_S(y, 0) + 80.0)) / 6.8), 0.0)
        AV_beta_h = ufl.conditional(
            ufl.lt(NV_Ith_S(y, 0), (-40.0)), 2.7 * ufl.exp(0.079 * NV_Ith_S(y, 0)) + 310000.0 * ufl.exp(0.3485 * NV_Ith_S(y, 0)), 0.77 / (0.13 * (1.0 + ufl.exp((NV_Ith_S(y, 0) + 10.66) / (-11.1))))
        )
        yinf[5] = 1.0 / pow(1.0 + ufl.exp((NV_Ith_S(y, 0) + 71.55) / 7.43), 2.0)
        tau[5] = 1.0 / (AV_alpha_h + AV_beta_h)

        # /* fast_sodium_current_j_gate */
        AV_alpha_j = ufl.conditional(
            ufl.lt(NV_Ith_S(y, 0), (-40.0)),
            ((-25428.0) * ufl.exp(0.2444 * NV_Ith_S(y, 0)) - 6.948e-06 * ufl.exp((-0.04391) * NV_Ith_S(y, 0))) * (NV_Ith_S(y, 0) + 37.78) / 1.0 / (1.0 + ufl.exp(0.311 * (NV_Ith_S(y, 0) + 79.23))),
            0.0,
        )
        AV_beta_j = ufl.conditional(
            ufl.lt(NV_Ith_S(y, 0), (-40.0)),
            0.02424 * ufl.exp((-0.01052) * NV_Ith_S(y, 0)) / (1.0 + ufl.exp((-0.1378) * (NV_Ith_S(y, 0) + 40.14))),
            0.6 * ufl.exp(0.057 * NV_Ith_S(y, 0)) / (1.0 + ufl.exp((-0.1) * (NV_Ith_S(y, 0) + 32.0))),
        )
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

        lmbda = [None] * self.size
        for i in range(self.size):
            if tau[i] is not None:
                lmbda[i] = -1.0 / tau[i]

        return lmbda, yinf
    
    @property
    def lmbda_yinf_exp(self):
        lmbda, yinf = self.f_exp_coeffs()
        return self.expression_list(lmbda), self.expression_list(yinf)
    
    @property
    def lmbda_exp(self):
        lmbda_expr, yinf_expr = self.lmbda_yinf_exp
        return lmbda_expr

    @property
    def f_exp(self):
        lmbda, yinf = self.f_exp_coeffs()
        return self.expression_list(self.f_from_lmbda_yinf(lmbda, yinf))

    @property
    def phi_f_exp(self):
        lmbda, yinf = self.f_exp_coeffs()
        return self.expression_list(self.apply_phi(lmbda, yinf))

    @property
    def f_stiff(self):
        y = self.y
        ydot = [None] * self.size

        # /* fast_sodium_current_m_gate */
        AV_alpha_m = 1.0 / (1.0 + ufl.exp(((-60.0) - NV_Ith_S(y, 0)) / 5.0))
        AV_beta_m = 0.1 / (1.0 + ufl.exp((NV_Ith_S(y, 0) + 35.0) / 5.0)) + 0.1 / (1.0 + ufl.exp((NV_Ith_S(y, 0) - 50.0) / 200.0))
        AV_m_inf = 1.0 / pow(1.0 + ufl.exp(((-56.86) - NV_Ith_S(y, 0)) / 9.03), 2.0)
        AV_tau_m = 1.0 * AV_alpha_m * AV_beta_m
        ydot[4] = (AV_m_inf - NV_Ith_S(y, 4)) / AV_tau_m

        return self.expression_list(ydot)

    @property
    def phi_f_stiff(self):
        y = self.y
        yinf = [None] * self.size
        tau = [None] * self.size

        # /* fast_sodium_current_m_gate */
        AV_alpha_m = 1.0 / (1.0 + ufl.exp(((-60.0) - NV_Ith_S(y, 0)) / 5.0))
        AV_beta_m = 0.1 / (1.0 + ufl.exp((NV_Ith_S(y, 0) + 35.0) / 5.0)) + 0.1 / (1.0 + ufl.exp((NV_Ith_S(y, 0) - 50.0) / 200.0))
        yinf[4] = 1.0 / pow(1.0 + ufl.exp(((-56.86) - NV_Ith_S(y, 0)) / 9.03), 2.0)
        tau[4] = 1.0 * AV_alpha_m * AV_beta_m

        lmbda = [None] * self.size
        lmbda[4] = -1.0 / tau[4]

        return self.expression_list(self.apply_phi(lmbda, yinf))

    @property
    def f_nonstiff(self):
        y = self.y
        ydot = [None] * self.size

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
        AV_tau_f = (
            1102.5 * ufl.exp((-pow(NV_Ith_S(y, 0) + 27.0, 2.0)) / 225.0) + 200.0 / (1.0 + ufl.exp((13.0 - NV_Ith_S(y, 0)) / 10.0)) + 180.0 / (1.0 + ufl.exp((NV_Ith_S(y, 0) + 30.0) / 10.0)) + 20.0
        )
        ydot[8] = (AV_f_inf - NV_Ith_S(y, 8)) / AV_tau_f

        # /* calcium_pump_current */
        AV_i_p_Ca = self.AC_g_pCa * NV_Ith_S(y, 13) / (NV_Ith_S(y, 13) + self.AC_K_pCa)

        # /* fast_sodium_current_h_gate */
        AV_alpha_h = ufl.conditional(ufl.lt(NV_Ith_S(y, 0), (-40.0)), 0.057 * ufl.exp((-(NV_Ith_S(y, 0) + 80.0)) / 6.8), 0.0)
        AV_beta_h = ufl.conditional(
            ufl.lt(NV_Ith_S(y, 0), (-40.0)), 2.7 * ufl.exp(0.079 * NV_Ith_S(y, 0)) + 310000.0 * ufl.exp(0.3485 * NV_Ith_S(y, 0)), 0.77 / (0.13 * (1.0 + ufl.exp((NV_Ith_S(y, 0) + 10.66) / (-11.1))))
        )
        AV_h_inf = 1.0 / pow(1.0 + ufl.exp((NV_Ith_S(y, 0) + 71.55) / 7.43), 2.0)
        AV_tau_h = 1.0 / (AV_alpha_h + AV_beta_h)
        ydot[5] = (AV_h_inf - NV_Ith_S(y, 5)) / AV_tau_h

        # /* fast_sodium_current_j_gate */
        AV_alpha_j = ufl.conditional(
            ufl.lt(NV_Ith_S(y, 0), (-40.0)),
            ((-25428.0) * ufl.exp(0.2444 * NV_Ith_S(y, 0)) - 6.948e-06 * ufl.exp((-0.04391) * NV_Ith_S(y, 0))) * (NV_Ith_S(y, 0) + 37.78) / 1.0 / (1.0 + ufl.exp(0.311 * (NV_Ith_S(y, 0) + 79.23))),
            0.0,
        )
        AV_beta_j = ufl.conditional(
            ufl.lt(NV_Ith_S(y, 0), (-40.0)),
            0.02424 * ufl.exp((-0.01052) * NV_Ith_S(y, 0)) / (1.0 + ufl.exp((-0.1378) * (NV_Ith_S(y, 0) + 40.14))),
            0.6 * ufl.exp(0.057 * NV_Ith_S(y, 0)) / (1.0 + ufl.exp((-0.1) * (NV_Ith_S(y, 0) + 32.0))),
        )
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
        AV_i_NaK = (
            self.AC_P_NaK
            * self.AC_K_o
            / (self.AC_K_o + self.AC_K_mk)
            * NV_Ith_S(y, 17)
            / (NV_Ith_S(y, 17) + self.AC_K_mNa)
            / (1.0 + 0.1245 * ufl.exp((-0.1) * NV_Ith_S(y, 0) * self.AC_F / (self.AC_R * self.AC_T)) + 0.0353 * ufl.exp((-NV_Ith_S(y, 0)) * self.AC_F / (self.AC_R * self.AC_T)))
        )

        # /* transient_outward_current */
        AV_i_to = self.AC_g_to * NV_Ith_S(y, 12) * NV_Ith_S(y, 11) * (NV_Ith_S(y, 0) - AV_E_K)

        # /* *remaining* */
        AV_i_CaL = (
            self.AC_g_CaL
            * NV_Ith_S(y, 7)
            * NV_Ith_S(y, 8)
            * NV_Ith_S(y, 9)
            * NV_Ith_S(y, 10)
            * 4.0
            * (NV_Ith_S(y, 0) - 15.0)
            * pow(self.AC_F, 2.0)
            / (self.AC_R * self.AC_T)
            * (0.25 * NV_Ith_S(y, 15) * ufl.exp(2.0 * (NV_Ith_S(y, 0) - 15.0) * self.AC_F / (self.AC_R * self.AC_T)) - self.AC_Ca_o)
            / (ufl.exp(2.0 * (NV_Ith_S(y, 0) - 15.0) * self.AC_F / (self.AC_R * self.AC_T)) - 1.0)
        )
        AV_i_b_Ca = self.AC_g_bca * (NV_Ith_S(y, 0) - AV_E_Ca)
        AV_alpha_K1 = 0.1 / (1.0 + ufl.exp(0.06 * (NV_Ith_S(y, 0) - AV_E_K - 200.0)))
        AV_beta_K1 = (3.0 * ufl.exp(0.0002 * (NV_Ith_S(y, 0) - AV_E_K + 100.0)) + ufl.exp(0.1 * (NV_Ith_S(y, 0) - AV_E_K - 10.0))) / (1.0 + ufl.exp((-0.5) * (NV_Ith_S(y, 0) - AV_E_K)))
        AV_i_p_K = self.AC_g_pK * (NV_Ith_S(y, 0) - AV_E_K) / (1.0 + ufl.exp((25.0 - NV_Ith_S(y, 0)) / 5.98))
        AV_i_Kr = self.AC_g_Kr * np.sqrt(self.AC_K_o / 5.4) * NV_Ith_S(y, 1) * NV_Ith_S(y, 2) * (NV_Ith_S(y, 0) - AV_E_K)
        AV_E_Ks = self.AC_R * self.AC_T / self.AC_F * ufl.ln((self.AC_K_o + self.AC_P_kna * self.AC_Na_o) / (NV_Ith_S(y, 18) + self.AC_P_kna * NV_Ith_S(y, 17)))
        AV_E_Na = self.AC_R * self.AC_T / self.AC_F * ufl.ln(self.AC_Na_o / NV_Ith_S(y, 17))
        AV_i_NaCa = (
            self.AC_K_NaCa
            * (
                ufl.exp(self.AC_sodium_calcium_exchanger_current_gamma * NV_Ith_S(y, 0) * self.AC_F / (self.AC_R * self.AC_T)) * pow(NV_Ith_S(y, 17), 3.0) * self.AC_Ca_o
                - ufl.exp((self.AC_sodium_calcium_exchanger_current_gamma - 1.0) * NV_Ith_S(y, 0) * self.AC_F / (self.AC_R * self.AC_T)) * pow(self.AC_Na_o, 3.0) * NV_Ith_S(y, 13) * self.AC_alpha
            )
            / (
                (pow(self.AC_Km_Nai, 3.0) + pow(self.AC_Na_o, 3.0))
                * (self.AC_Km_Ca + self.AC_Ca_o)
                * (1.0 + self.AC_K_sat * ufl.exp((self.AC_sodium_calcium_exchanger_current_gamma - 1.0) * NV_Ith_S(y, 0) * self.AC_F / (self.AC_R * self.AC_T)))
            )
        )
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
        ydot[0] = self.scale * (-(AV_i_K1 + AV_i_to + AV_i_Kr + AV_i_Ks + AV_i_CaL + AV_i_NaK + AV_i_Na + AV_i_b_Na + AV_i_NaCa + AV_i_b_Ca + AV_i_p_K + AV_i_p_Ca))
        ydot[18] = (-(AV_i_K1 + AV_i_to + AV_i_Kr + AV_i_Ks + AV_i_p_K - 2.0 * AV_i_NaK)) / (self.AC_V_c * self.AC_F) * self.AC_Cm

        return self.expression_list(ydot)
