import numpy as np
import ufl
from pySDC.projects.ExplicitStabilized.problem_classes.monodomain_system_helpers.ionicmodels.ufl.ionicmodel import IonicModel
from pySDC.projects.ExplicitStabilized.problem_classes.monodomain_system_helpers.ionicmodels.ufl.ionicmodel import NV_Ith_S


# Mildly stiff with rho_max ~ 130
class Courtemanche1998(IonicModel):
    def __init__(self, scale):
        super(Courtemanche1998, self).__init__(scale)

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
        self.AC_g_B_Na = 6.74437500000000015e-04

        self.AC_g_Na = 7.8

        self.AC_V_cell = 20100.0
        self.AC_V_i = self.AC_V_cell * 0.68
        self.AC_V_rel = 0.0048 * self.AC_V_cell
        self.AC_V_up = 0.0552 * self.AC_V_cell

        self.AC_Cm = 1.0  # 100.0
        self.AC_F = 96.4867
        self.AC_R = 8.3143
        self.AC_T = 310.0

        self.AC_g_Kr = 2.94117649999999994e-02

        self.AC_i_CaP_max = 0.275

        self.AC_g_Ks = 1.29411759999999987e-01

        self.AC_Km_K_o = 1.5
        self.AC_Km_Na_i = 10.0
        self.AC_i_NaK_max = 5.99338739999999981e-01
        self.AC_sigma = 1.0 / 7.0 * (np.exp(self.AC_Na_o / 67.3) - 1.0)

        self.AC_g_K1 = 0.09

        self.AC_K_Q10 = 3.0
        self.AC_g_to = 0.1652

        # overall stiffness of f is 130 circa
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

        # list of indeces needed to compute or which are affected by the given function
        self.f_nonstiff_args = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]  # all
        self.f_stiff_args = [0, 1]
        self.f_expl_args = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]  # all
        self.f_exp_args = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15]

        # list of indeces affected by a given function
        self.f_stiff_indeces = [1]
        self.f_nonstiff_indeces = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]  # all except 1
        self.f_expl_indeces = [0, 12, 13, 14, 16, 17, 18, 19, 20]
        self.f_exp_indeces = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15]

    def initial_values(self):
        y0 = [None] * self.size

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

    @property
    def f(self):
        y = self.y
        ydot = [None] * self.size

        # Linear (in the gating variables) terms

        # /* Ca_release_current_from_JSR_w_gate */
        AV_tau_w = ufl.conditional(
            ufl.lt(ufl.algebra.Abs(NV_Ith_S(y, 0) - 7.9), 1e-10),
            6.0 * 0.2 / 1.3,
            6.0 * (1.0 - ufl.exp((-(NV_Ith_S(y, 0) - 7.9)) / 5.0)) / ((1.0 + 0.3 * ufl.exp((-(NV_Ith_S(y, 0) - 7.9)) / 5.0)) * 1.0 * (NV_Ith_S(y, 0) - 7.9)),
        )
        AV_w_infinity = 1.0 - pow(1.0 + ufl.exp((-(NV_Ith_S(y, 0) - 40.0)) / 17.0), (-1.0))
        ydot[15] = (AV_w_infinity - NV_Ith_S(y, 15)) / AV_tau_w

        # /* L_type_Ca_channel_d_gate */
        AV_d_infinity = pow(1.0 + ufl.exp((NV_Ith_S(y, 0) + 10.0) / (-8.0)), (-1.0))
        AV_tau_d = ufl.conditional(
            ufl.lt(ufl.algebra.Abs(NV_Ith_S(y, 0) + 10.0), 1e-10),
            4.579 / (1.0 + ufl.exp((NV_Ith_S(y, 0) + 10.0) / (-6.24))),
            (1.0 - ufl.exp((NV_Ith_S(y, 0) + 10.0) / (-6.24))) / (0.035 * (NV_Ith_S(y, 0) + 10.0) * (1.0 + ufl.exp((NV_Ith_S(y, 0) + 10.0) / (-6.24)))),
        )
        ydot[10] = (AV_d_infinity - NV_Ith_S(y, 10)) / AV_tau_d

        # /* L_type_Ca_channel_f_gate */
        AV_f_infinity = ufl.exp((-(NV_Ith_S(y, 0) + 28.0)) / 6.9) / (1.0 + ufl.exp((-(NV_Ith_S(y, 0) + 28.0)) / 6.9))
        AV_tau_f = 9.0 * pow(0.0197 * ufl.exp((-pow(0.0337, 2.0)) * pow(NV_Ith_S(y, 0) + 10.0, 2.0)) + 0.02, (-1.0))
        ydot[11] = (AV_f_infinity - NV_Ith_S(y, 11)) / AV_tau_f

        # /* fast_sodium_current_h_gate */
        AV_alpha_h = ufl.conditional(ufl.lt(NV_Ith_S(y, 0), (-40.0)), 0.135 * ufl.exp((NV_Ith_S(y, 0) + 80.0) / (-6.8)), 0.0)
        AV_beta_h = ufl.conditional(
            ufl.lt(NV_Ith_S(y, 0), (-40.0)), 3.56 * ufl.exp(0.079 * NV_Ith_S(y, 0)) + 310000.0 * ufl.exp(0.35 * NV_Ith_S(y, 0)), 1.0 / (0.13 * (1.0 + ufl.exp((NV_Ith_S(y, 0) + 10.66) / (-11.1))))
        )
        AV_h_inf = AV_alpha_h / (AV_alpha_h + AV_beta_h)
        AV_tau_h = 1.0 / (AV_alpha_h + AV_beta_h)
        ydot[2] = (AV_h_inf - NV_Ith_S(y, 2)) / AV_tau_h

        # /* fast_sodium_current_j_gate */
        AV_alpha_j = ufl.conditional(
            ufl.lt(NV_Ith_S(y, 0), (-40.0)),
            ((-127140.0) * ufl.exp(0.2444 * NV_Ith_S(y, 0)) - 3.474e-05 * ufl.exp((-0.04391) * NV_Ith_S(y, 0))) * (NV_Ith_S(y, 0) + 37.78) / (1.0 + ufl.exp(0.311 * (NV_Ith_S(y, 0) + 79.23))),
            0.0,
        )
        AV_beta_j = ufl.conditional(
            ufl.lt(NV_Ith_S(y, 0), (-40.0)),
            0.1212 * ufl.exp((-0.01052) * NV_Ith_S(y, 0)) / (1.0 + ufl.exp((-0.1378) * (NV_Ith_S(y, 0) + 40.14))),
            0.3 * ufl.exp((-2.535e-07) * NV_Ith_S(y, 0)) / (1.0 + ufl.exp((-0.1) * (NV_Ith_S(y, 0) + 32.0))),
        )
        AV_j_inf = AV_alpha_j / (AV_alpha_j + AV_beta_j)
        AV_tau_j = 1.0 / (AV_alpha_j + AV_beta_j)
        ydot[3] = (AV_j_inf - NV_Ith_S(y, 3)) / AV_tau_j

        # /* fast_sodium_current_m_gate */
        AV_alpha_m = ufl.conditional(ufl.eq(NV_Ith_S(y, 0), (-47.13)), 3.2, 0.32 * (NV_Ith_S(y, 0) + 47.13) / (1.0 - ufl.exp((-0.1) * (NV_Ith_S(y, 0) + 47.13))))
        AV_beta_m = 0.08 * ufl.exp((-NV_Ith_S(y, 0)) / 11.0)
        AV_m_inf = AV_alpha_m / (AV_alpha_m + AV_beta_m)
        AV_tau_m = 1.0 / (AV_alpha_m + AV_beta_m)
        ydot[1] = (AV_m_inf - NV_Ith_S(y, 1)) / AV_tau_m

        # /* rapid_delayed_rectifier_K_current_xr_gate */
        AV_alpha_xr = ufl.conditional(ufl.lt(ufl.algebra.Abs(NV_Ith_S(y, 0) + 14.1), 1e-10), 0.0015, 0.0003 * (NV_Ith_S(y, 0) + 14.1) / (1.0 - ufl.exp((NV_Ith_S(y, 0) + 14.1) / (-5.0))))
        AV_beta_xr = ufl.conditional(
            ufl.lt(ufl.algebra.Abs(NV_Ith_S(y, 0) - 3.3328), 1e-10), 3.78361180000000004e-04, 7.38980000000000030e-05 * (NV_Ith_S(y, 0) - 3.3328) / (ufl.exp((NV_Ith_S(y, 0) - 3.3328) / 5.1237) - 1.0)
        )
        AV_xr_infinity = pow(1.0 + ufl.exp((NV_Ith_S(y, 0) + 14.1) / (-6.5)), (-1.0))
        AV_tau_xr = pow(AV_alpha_xr + AV_beta_xr, (-1.0))
        ydot[8] = (AV_xr_infinity - NV_Ith_S(y, 8)) / AV_tau_xr

        # /* slow_delayed_rectifier_K_current_xs_gate */
        AV_alpha_xs = ufl.conditional(ufl.lt(ufl.algebra.Abs(NV_Ith_S(y, 0) - 19.9), 1e-10), 0.00068, 4e-05 * (NV_Ith_S(y, 0) - 19.9) / (1.0 - ufl.exp((NV_Ith_S(y, 0) - 19.9) / (-17.0))))
        AV_beta_xs = ufl.conditional(ufl.lt(ufl.algebra.Abs(NV_Ith_S(y, 0) - 19.9), 1e-10), 0.000315, 3.5e-05 * (NV_Ith_S(y, 0) - 19.9) / (ufl.exp((NV_Ith_S(y, 0) - 19.9) / 9.0) - 1.0))
        AV_xs_infinity = pow(1.0 + ufl.exp((NV_Ith_S(y, 0) - 19.9) / (-12.7)), (-0.5))
        AV_tau_xs = 0.5 * pow(AV_alpha_xs + AV_beta_xs, (-1.0))
        ydot[9] = (AV_xs_infinity - NV_Ith_S(y, 9)) / AV_tau_xs

        # /* transient_outward_K_current_oa_gate */
        AV_alpha_oa = 0.65 * pow(ufl.exp((NV_Ith_S(y, 0) - (-10.0)) / (-8.5)) + ufl.exp((NV_Ith_S(y, 0) - (-10.0) - 40.0) / (-59.0)), (-1.0))
        AV_beta_oa = 0.65 * pow(2.5 + ufl.exp((NV_Ith_S(y, 0) - (-10.0) + 72.0) / 17.0), (-1.0))
        AV_oa_infinity = pow(1.0 + ufl.exp((NV_Ith_S(y, 0) - (-10.0) + 10.47) / (-17.54)), (-1.0))
        AV_tau_oa = pow(AV_alpha_oa + AV_beta_oa, (-1.0)) / self.AC_K_Q10
        ydot[4] = (AV_oa_infinity - NV_Ith_S(y, 4)) / AV_tau_oa

        # /* transient_outward_K_current_oi_gate */
        AV_alpha_oi = pow(18.53 + 1.0 * ufl.exp((NV_Ith_S(y, 0) - (-10.0) + 103.7) / 10.95), (-1.0))
        AV_beta_oi = pow(35.56 + 1.0 * ufl.exp((NV_Ith_S(y, 0) - (-10.0) - 8.74) / (-7.44)), (-1.0))
        AV_oi_infinity = pow(1.0 + ufl.exp((NV_Ith_S(y, 0) - (-10.0) + 33.1) / 5.3), (-1.0))
        AV_tau_oi = pow(AV_alpha_oi + AV_beta_oi, (-1.0)) / self.AC_K_Q10
        ydot[5] = (AV_oi_infinity - NV_Ith_S(y, 5)) / AV_tau_oi

        # /* ultrarapid_delayed_rectifier_K_current_ua_gate */
        AV_alpha_ua = 0.65 * pow(ufl.exp((NV_Ith_S(y, 0) - (-10.0)) / (-8.5)) + ufl.exp((NV_Ith_S(y, 0) - (-10.0) - 40.0) / (-59.0)), (-1.0))
        AV_beta_ua = 0.65 * pow(2.5 + ufl.exp((NV_Ith_S(y, 0) - (-10.0) + 72.0) / 17.0), (-1.0))
        AV_ua_infinity = pow(1.0 + ufl.exp((NV_Ith_S(y, 0) - (-10.0) + 20.3) / (-9.6)), (-1.0))
        AV_tau_ua = pow(AV_alpha_ua + AV_beta_ua, (-1.0)) / self.AC_K_Q10
        ydot[6] = (AV_ua_infinity - NV_Ith_S(y, 6)) / AV_tau_ua

        # /* ultrarapid_delayed_rectifier_K_current_ui_gate */
        AV_alpha_ui = pow(21.0 + 1.0 * ufl.exp((NV_Ith_S(y, 0) - (-10.0) - 195.0) / (-28.0)), (-1.0))
        AV_beta_ui = 1.0 / ufl.exp((NV_Ith_S(y, 0) - (-10.0) - 168.0) / (-16.0))
        AV_ui_infinity = pow(1.0 + ufl.exp((NV_Ith_S(y, 0) - (-10.0) - 109.45) / 27.48), (-1.0))
        AV_tau_ui = pow(AV_alpha_ui + AV_beta_ui, (-1.0)) / self.AC_K_Q10
        ydot[7] = (AV_ui_infinity - NV_Ith_S(y, 7)) / AV_tau_ui

        # Non Linear (in the gating variables) terms

        # /* L_type_Ca_channel_f_Ca_gate */
        AV_f_Ca_infinity = pow(1.0 + NV_Ith_S(y, 17) / 0.00035, (-1.0))
        ydot[12] = (AV_f_Ca_infinity - NV_Ith_S(y, 12)) / self.AC_tau_f_Ca

        # /* transfer_current_from_NSR_to_JSR */
        AV_i_tr = (NV_Ith_S(y, 20) - NV_Ith_S(y, 19)) / self.AC_tau_tr

        # /* Ca_leak_current_by_the_NSR */
        AV_i_up_leak = self.AC_I_up_max * NV_Ith_S(y, 20) / self.AC_Ca_up_max

        # /* Ca_release_current_from_JSR */
        AV_i_rel = self.AC_K_rel * pow(NV_Ith_S(y, 13), 2.0) * NV_Ith_S(y, 14) * NV_Ith_S(y, 15) * (NV_Ith_S(y, 19) - NV_Ith_S(y, 17))

        # /* intracellular_ion_concentrations */
        ydot[19] = (AV_i_tr - AV_i_rel) * pow(1.0 + self.AC_CSQN_max * self.AC_Km_CSQN / pow(NV_Ith_S(y, 19) + self.AC_Km_CSQN, 2.0), (-1.0))

        # /* Ca_uptake_current_by_the_NSR */
        AV_i_up = self.AC_I_up_max / (1.0 + self.AC_K_up / NV_Ith_S(y, 17))
        ydot[20] = AV_i_up - (AV_i_up_leak + AV_i_tr * self.AC_V_rel / self.AC_V_up)

        # /* sarcolemmal_calcium_pump_current */
        AV_i_CaP = self.AC_Cm * self.AC_i_CaP_max * NV_Ith_S(y, 17) / (0.0005 + NV_Ith_S(y, 17))

        # /* sodium_potassium_pump */
        AV_f_NaK = pow(
            1.0 + 0.1245 * ufl.exp((-0.1) * self.AC_F * NV_Ith_S(y, 0) / (self.AC_R * self.AC_T)) + 0.0365 * self.AC_sigma * ufl.exp((-self.AC_F) * NV_Ith_S(y, 0) / (self.AC_R * self.AC_T)), (-1.0)
        )
        AV_i_NaK = self.AC_Cm * self.AC_i_NaK_max * AV_f_NaK * 1.0 / (1.0 + pow(self.AC_Km_Na_i / NV_Ith_S(y, 16), 1.5)) * self.AC_K_o / (self.AC_K_o + self.AC_Km_K_o)

        # /* time_independent_potassium_current */
        AV_E_K = self.AC_R * self.AC_T / self.AC_F * ufl.ln(self.AC_K_o / NV_Ith_S(y, 18))
        AV_i_K1 = self.AC_Cm * self.AC_g_K1 * (NV_Ith_S(y, 0) - AV_E_K) / (1.0 + ufl.exp(0.07 * (NV_Ith_S(y, 0) + 80.0)))

        # /* transient_outward_K_current */
        AV_i_to = self.AC_Cm * self.AC_g_to * pow(NV_Ith_S(y, 4), 3.0) * NV_Ith_S(y, 5) * (NV_Ith_S(y, 0) - AV_E_K)

        # /* ultrarapid_delayed_rectifier_K_current */
        AV_g_Kur = 0.005 + 0.05 / (1.0 + ufl.exp((NV_Ith_S(y, 0) - 15.0) / (-13.0)))
        AV_i_Kur = self.AC_Cm * AV_g_Kur * pow(NV_Ith_S(y, 6), 3.0) * NV_Ith_S(y, 7) * (NV_Ith_S(y, 0) - AV_E_K)

        # /* *remaining* */
        AV_i_Ca_L = self.AC_Cm * self.AC_g_Ca_L * NV_Ith_S(y, 10) * NV_Ith_S(y, 11) * NV_Ith_S(y, 12) * (NV_Ith_S(y, 0) - 65.0)
        AV_i_NaCa = (
            self.AC_Cm
            * self.AC_I_NaCa_max
            * (
                ufl.exp(self.AC_Na_Ca_exchanger_current_gamma * self.AC_F * NV_Ith_S(y, 0) / (self.AC_R * self.AC_T)) * pow(NV_Ith_S(y, 16), 3.0) * self.AC_Ca_o
                - ufl.exp((self.AC_Na_Ca_exchanger_current_gamma - 1.0) * self.AC_F * NV_Ith_S(y, 0) / (self.AC_R * self.AC_T)) * pow(self.AC_Na_o, 3.0) * NV_Ith_S(y, 17)
            )
            / (
                (pow(self.AC_K_mNa, 3.0) + pow(self.AC_Na_o, 3.0))
                * (self.AC_K_mCa + self.AC_Ca_o)
                * (1.0 + self.AC_K_sat * ufl.exp((self.AC_Na_Ca_exchanger_current_gamma - 1.0) * NV_Ith_S(y, 0) * self.AC_F / (self.AC_R * self.AC_T)))
            )
        )
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
        AV_u_infinity = pow(1.0 + ufl.exp((-(AV_Fn - 3.41749999999999983e-13)) / 1.367e-15), (-1.0))
        AV_tau_v = 1.91 + 2.09 * pow(1.0 + ufl.exp((-(AV_Fn - 3.41749999999999983e-13)) / 1.367e-15), (-1.0))
        AV_v_infinity = 1.0 - pow(1.0 + ufl.exp((-(AV_Fn - 6.835e-14)) / 1.367e-15), (-1.0))
        ydot[16] = ((-3.0) * AV_i_NaK - (3.0 * AV_i_NaCa + AV_i_B_Na + AV_i_Na)) / (self.AC_V_i * self.AC_F)

        ydot[0] = self.scale * (-(AV_i_Na + AV_i_K1 + AV_i_to + AV_i_Kur + AV_i_Kr + AV_i_Ks + AV_i_B_Na + AV_i_B_Ca + AV_i_NaK + AV_i_CaP + AV_i_NaCa + AV_i_Ca_L)) / self.AC_Cm
        ydot[13] = (AV_u_infinity - NV_Ith_S(y, 13)) / self.AC_tau_u
        ydot[14] = (AV_v_infinity - NV_Ith_S(y, 14)) / AV_tau_v

        AV_B1 = (2.0 * AV_i_NaCa - (AV_i_CaP + AV_i_Ca_L + AV_i_B_Ca)) / (2.0 * self.AC_V_i * self.AC_F) + (self.AC_V_up * (AV_i_up_leak - AV_i_up) + AV_i_rel * self.AC_V_rel) / self.AC_V_i
        AV_B2 = 1.0 + self.AC_TRPN_max * self.AC_Km_TRPN / pow(NV_Ith_S(y, 17) + self.AC_Km_TRPN, 2.0) + self.AC_CMDN_max * self.AC_Km_CMDN / pow(NV_Ith_S(y, 17) + self.AC_Km_CMDN, 2.0)
        ydot[17] = AV_B1 / AV_B2

        return self.expression_list(ydot)

    @property
    def f_expl(self):
        y = self.y
        ydot = [None] * self.size

        # Non Linear (in the gating variables) terms

        # /* L_type_Ca_channel_f_Ca_gate */
        AV_f_Ca_infinity = pow(1.0 + NV_Ith_S(y, 17) / 0.00035, (-1.0))
        ydot[12] = (AV_f_Ca_infinity - NV_Ith_S(y, 12)) / self.AC_tau_f_Ca

        # /* transfer_current_from_NSR_to_JSR */
        AV_i_tr = (NV_Ith_S(y, 20) - NV_Ith_S(y, 19)) / self.AC_tau_tr

        # /* Ca_leak_current_by_the_NSR */
        AV_i_up_leak = self.AC_I_up_max * NV_Ith_S(y, 20) / self.AC_Ca_up_max

        # /* Ca_release_current_from_JSR */
        AV_i_rel = self.AC_K_rel * pow(NV_Ith_S(y, 13), 2.0) * NV_Ith_S(y, 14) * NV_Ith_S(y, 15) * (NV_Ith_S(y, 19) - NV_Ith_S(y, 17))

        # /* intracellular_ion_concentrations */
        ydot[19] = (AV_i_tr - AV_i_rel) * pow(1.0 + self.AC_CSQN_max * self.AC_Km_CSQN / pow(NV_Ith_S(y, 19) + self.AC_Km_CSQN, 2.0), (-1.0))

        # /* Ca_uptake_current_by_the_NSR */
        AV_i_up = self.AC_I_up_max / (1.0 + self.AC_K_up / NV_Ith_S(y, 17))
        ydot[20] = AV_i_up - (AV_i_up_leak + AV_i_tr * self.AC_V_rel / self.AC_V_up)

        # /* sarcolemmal_calcium_pump_current */
        AV_i_CaP = self.AC_Cm * self.AC_i_CaP_max * NV_Ith_S(y, 17) / (0.0005 + NV_Ith_S(y, 17))

        # /* sodium_potassium_pump */
        AV_f_NaK = pow(
            1.0 + 0.1245 * ufl.exp((-0.1) * self.AC_F * NV_Ith_S(y, 0) / (self.AC_R * self.AC_T)) + 0.0365 * self.AC_sigma * ufl.exp((-self.AC_F) * NV_Ith_S(y, 0) / (self.AC_R * self.AC_T)), (-1.0)
        )
        AV_i_NaK = self.AC_Cm * self.AC_i_NaK_max * AV_f_NaK * 1.0 / (1.0 + pow(self.AC_Km_Na_i / NV_Ith_S(y, 16), 1.5)) * self.AC_K_o / (self.AC_K_o + self.AC_Km_K_o)

        # /* time_independent_potassium_current */
        AV_E_K = self.AC_R * self.AC_T / self.AC_F * ufl.ln(self.AC_K_o / NV_Ith_S(y, 18))
        AV_i_K1 = self.AC_Cm * self.AC_g_K1 * (NV_Ith_S(y, 0) - AV_E_K) / (1.0 + ufl.exp(0.07 * (NV_Ith_S(y, 0) + 80.0)))

        # /* transient_outward_K_current */
        AV_i_to = self.AC_Cm * self.AC_g_to * pow(NV_Ith_S(y, 4), 3.0) * NV_Ith_S(y, 5) * (NV_Ith_S(y, 0) - AV_E_K)

        # /* ultrarapid_delayed_rectifier_K_current */
        AV_g_Kur = 0.005 + 0.05 / (1.0 + ufl.exp((NV_Ith_S(y, 0) - 15.0) / (-13.0)))
        AV_i_Kur = self.AC_Cm * AV_g_Kur * pow(NV_Ith_S(y, 6), 3.0) * NV_Ith_S(y, 7) * (NV_Ith_S(y, 0) - AV_E_K)

        # /* *remaining* */
        AV_i_Ca_L = self.AC_Cm * self.AC_g_Ca_L * NV_Ith_S(y, 10) * NV_Ith_S(y, 11) * NV_Ith_S(y, 12) * (NV_Ith_S(y, 0) - 65.0)
        AV_i_NaCa = (
            self.AC_Cm
            * self.AC_I_NaCa_max
            * (
                ufl.exp(self.AC_Na_Ca_exchanger_current_gamma * self.AC_F * NV_Ith_S(y, 0) / (self.AC_R * self.AC_T)) * pow(NV_Ith_S(y, 16), 3.0) * self.AC_Ca_o
                - ufl.exp((self.AC_Na_Ca_exchanger_current_gamma - 1.0) * self.AC_F * NV_Ith_S(y, 0) / (self.AC_R * self.AC_T)) * pow(self.AC_Na_o, 3.0) * NV_Ith_S(y, 17)
            )
            / (
                (pow(self.AC_K_mNa, 3.0) + pow(self.AC_Na_o, 3.0))
                * (self.AC_K_mCa + self.AC_Ca_o)
                * (1.0 + self.AC_K_sat * ufl.exp((self.AC_Na_Ca_exchanger_current_gamma - 1.0) * NV_Ith_S(y, 0) * self.AC_F / (self.AC_R * self.AC_T)))
            )
        )
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
        AV_u_infinity = pow(1.0 + ufl.exp((-(AV_Fn - 3.41749999999999983e-13)) / 1.367e-15), (-1.0))
        AV_tau_v = 1.91 + 2.09 * pow(1.0 + ufl.exp((-(AV_Fn - 3.41749999999999983e-13)) / 1.367e-15), (-1.0))
        AV_v_infinity = 1.0 - pow(1.0 + ufl.exp((-(AV_Fn - 6.835e-14)) / 1.367e-15), (-1.0))
        ydot[16] = ((-3.0) * AV_i_NaK - (3.0 * AV_i_NaCa + AV_i_B_Na + AV_i_Na)) / (self.AC_V_i * self.AC_F)

        ydot[0] = self.scale * (-(AV_i_Na + AV_i_K1 + AV_i_to + AV_i_Kur + AV_i_Kr + AV_i_Ks + AV_i_B_Na + AV_i_B_Ca + AV_i_NaK + AV_i_CaP + AV_i_NaCa + AV_i_Ca_L)) / self.AC_Cm
        ydot[13] = (AV_u_infinity - NV_Ith_S(y, 13)) / self.AC_tau_u
        ydot[14] = (AV_v_infinity - NV_Ith_S(y, 14)) / AV_tau_v

        AV_B1 = (2.0 * AV_i_NaCa - (AV_i_CaP + AV_i_Ca_L + AV_i_B_Ca)) / (2.0 * self.AC_V_i * self.AC_F) + (self.AC_V_up * (AV_i_up_leak - AV_i_up) + AV_i_rel * self.AC_V_rel) / self.AC_V_i
        AV_B2 = 1.0 + self.AC_TRPN_max * self.AC_Km_TRPN / pow(NV_Ith_S(y, 17) + self.AC_Km_TRPN, 2.0) + self.AC_CMDN_max * self.AC_Km_CMDN / pow(NV_Ith_S(y, 17) + self.AC_Km_CMDN, 2.0)
        ydot[17] = AV_B1 / AV_B2

        return self.expression_list(ydot)

    def f_exp_coeffs(self):
        y = self.y
        yinf = [None] * self.size
        tau = [None] * self.size

        # Linear (in the gating variables) terms

        # /* Ca_release_current_from_JSR_w_gate */
        tau[15] = ufl.conditional(
            ufl.lt(ufl.algebra.Abs(NV_Ith_S(y, 0) - 7.9), 1e-10),
            6.0 * 0.2 / 1.3,
            6.0 * (1.0 - ufl.exp((-(NV_Ith_S(y, 0) - 7.9)) / 5.0)) / ((1.0 + 0.3 * ufl.exp((-(NV_Ith_S(y, 0) - 7.9)) / 5.0)) * 1.0 * (NV_Ith_S(y, 0) - 7.9)),
        )
        yinf[15] = 1.0 - pow(1.0 + ufl.exp((-(NV_Ith_S(y, 0) - 40.0)) / 17.0), (-1.0))

        # /* L_type_Ca_channel_d_gate */
        yinf[10] = pow(1.0 + ufl.exp((NV_Ith_S(y, 0) + 10.0) / (-8.0)), (-1.0))
        tau[10] = ufl.conditional(
            ufl.lt(ufl.algebra.Abs(NV_Ith_S(y, 0) + 10.0), 1e-10),
            4.579 / (1.0 + ufl.exp((NV_Ith_S(y, 0) + 10.0) / (-6.24))),
            (1.0 - ufl.exp((NV_Ith_S(y, 0) + 10.0) / (-6.24))) / (0.035 * (NV_Ith_S(y, 0) + 10.0) * (1.0 + ufl.exp((NV_Ith_S(y, 0) + 10.0) / (-6.24)))),
        )

        # /* L_type_Ca_channel_f_gate */
        yinf[11] = ufl.exp((-(NV_Ith_S(y, 0) + 28.0)) / 6.9) / (1.0 + ufl.exp((-(NV_Ith_S(y, 0) + 28.0)) / 6.9))
        tau[11] = 9.0 * pow(0.0197 * ufl.exp((-pow(0.0337, 2.0)) * pow(NV_Ith_S(y, 0) + 10.0, 2.0)) + 0.02, (-1.0))

        # /* fast_sodium_current_h_gate */
        AV_alpha_h = ufl.conditional(ufl.lt(NV_Ith_S(y, 0), (-40.0)), 0.135 * ufl.exp((NV_Ith_S(y, 0) + 80.0) / (-6.8)), 0.0)
        AV_beta_h = ufl.conditional(
            ufl.lt(NV_Ith_S(y, 0), (-40.0)), 3.56 * ufl.exp(0.079 * NV_Ith_S(y, 0)) + 310000.0 * ufl.exp(0.35 * NV_Ith_S(y, 0)), 1.0 / (0.13 * (1.0 + ufl.exp((NV_Ith_S(y, 0) + 10.66) / (-11.1))))
        )
        yinf[2] = AV_alpha_h / (AV_alpha_h + AV_beta_h)
        tau[2] = 1.0 / (AV_alpha_h + AV_beta_h)

        # /* fast_sodium_current_j_gate */
        AV_alpha_j = ufl.conditional(
            ufl.lt(NV_Ith_S(y, 0), (-40.0)),
            ((-127140.0) * ufl.exp(0.2444 * NV_Ith_S(y, 0)) - 3.474e-05 * ufl.exp((-0.04391) * NV_Ith_S(y, 0))) * (NV_Ith_S(y, 0) + 37.78) / (1.0 + ufl.exp(0.311 * (NV_Ith_S(y, 0) + 79.23))),
            0.0,
        )
        AV_beta_j = ufl.conditional(
            ufl.lt(NV_Ith_S(y, 0), (-40.0)),
            0.1212 * ufl.exp((-0.01052) * NV_Ith_S(y, 0)) / (1.0 + ufl.exp((-0.1378) * (NV_Ith_S(y, 0) + 40.14))),
            0.3 * ufl.exp((-2.535e-07) * NV_Ith_S(y, 0)) / (1.0 + ufl.exp((-0.1) * (NV_Ith_S(y, 0) + 32.0))),
        )
        yinf[3] = AV_alpha_j / (AV_alpha_j + AV_beta_j)
        tau[3] = 1.0 / (AV_alpha_j + AV_beta_j)

        # /* fast_sodium_current_m_gate */
        AV_alpha_m = ufl.conditional(ufl.eq(NV_Ith_S(y, 0), (-47.13)), 3.2, 0.32 * (NV_Ith_S(y, 0) + 47.13) / (1.0 - ufl.exp((-0.1) * (NV_Ith_S(y, 0) + 47.13))))
        AV_beta_m = 0.08 * ufl.exp((-NV_Ith_S(y, 0)) / 11.0)
        yinf[1] = AV_alpha_m / (AV_alpha_m + AV_beta_m)
        tau[1] = 1.0 / (AV_alpha_m + AV_beta_m)

        # /* rapid_delayed_rectifier_K_current_xr_gate */
        AV_alpha_xr = ufl.conditional(ufl.lt(ufl.algebra.Abs(NV_Ith_S(y, 0) + 14.1), 1e-10), 0.0015, 0.0003 * (NV_Ith_S(y, 0) + 14.1) / (1.0 - ufl.exp((NV_Ith_S(y, 0) + 14.1) / (-5.0))))
        AV_beta_xr = ufl.conditional(
            ufl.lt(ufl.algebra.Abs(NV_Ith_S(y, 0) - 3.3328), 1e-10), 3.78361180000000004e-04, 7.38980000000000030e-05 * (NV_Ith_S(y, 0) - 3.3328) / (ufl.exp((NV_Ith_S(y, 0) - 3.3328) / 5.1237) - 1.0)
        )
        yinf[8] = pow(1.0 + ufl.exp((NV_Ith_S(y, 0) + 14.1) / (-6.5)), (-1.0))
        tau[8] = pow(AV_alpha_xr + AV_beta_xr, (-1.0))

        # /* slow_delayed_rectifier_K_current_xs_gate */
        AV_alpha_xs = ufl.conditional(ufl.lt(ufl.algebra.Abs(NV_Ith_S(y, 0) - 19.9), 1e-10), 0.00068, 4e-05 * (NV_Ith_S(y, 0) - 19.9) / (1.0 - ufl.exp((NV_Ith_S(y, 0) - 19.9) / (-17.0))))
        AV_beta_xs = ufl.conditional(ufl.lt(ufl.algebra.Abs(NV_Ith_S(y, 0) - 19.9), 1e-10), 0.000315, 3.5e-05 * (NV_Ith_S(y, 0) - 19.9) / (ufl.exp((NV_Ith_S(y, 0) - 19.9) / 9.0) - 1.0))
        yinf[9] = pow(1.0 + ufl.exp((NV_Ith_S(y, 0) - 19.9) / (-12.7)), (-0.5))
        tau[9] = 0.5 * pow(AV_alpha_xs + AV_beta_xs, (-1.0))

        # /* transient_outward_K_current_oa_gate */
        AV_alpha_oa = 0.65 * pow(ufl.exp((NV_Ith_S(y, 0) - (-10.0)) / (-8.5)) + ufl.exp((NV_Ith_S(y, 0) - (-10.0) - 40.0) / (-59.0)), (-1.0))
        AV_beta_oa = 0.65 * pow(2.5 + ufl.exp((NV_Ith_S(y, 0) - (-10.0) + 72.0) / 17.0), (-1.0))
        yinf[4] = pow(1.0 + ufl.exp((NV_Ith_S(y, 0) - (-10.0) + 10.47) / (-17.54)), (-1.0))
        tau[4] = pow(AV_alpha_oa + AV_beta_oa, (-1.0)) / self.AC_K_Q10

        # /* transient_outward_K_current_oi_gate */
        AV_alpha_oi = pow(18.53 + 1.0 * ufl.exp((NV_Ith_S(y, 0) - (-10.0) + 103.7) / 10.95), (-1.0))
        AV_beta_oi = pow(35.56 + 1.0 * ufl.exp((NV_Ith_S(y, 0) - (-10.0) - 8.74) / (-7.44)), (-1.0))
        yinf[5] = pow(1.0 + ufl.exp((NV_Ith_S(y, 0) - (-10.0) + 33.1) / 5.3), (-1.0))
        tau[5] = pow(AV_alpha_oi + AV_beta_oi, (-1.0)) / self.AC_K_Q10

        # /* ultrarapid_delayed_rectifier_K_current_ua_gate */
        AV_alpha_ua = 0.65 * pow(ufl.exp((NV_Ith_S(y, 0) - (-10.0)) / (-8.5)) + ufl.exp((NV_Ith_S(y, 0) - (-10.0) - 40.0) / (-59.0)), (-1.0))
        AV_beta_ua = 0.65 * pow(2.5 + ufl.exp((NV_Ith_S(y, 0) - (-10.0) + 72.0) / 17.0), (-1.0))
        yinf[6] = pow(1.0 + ufl.exp((NV_Ith_S(y, 0) - (-10.0) + 20.3) / (-9.6)), (-1.0))
        tau[6] = pow(AV_alpha_ua + AV_beta_ua, (-1.0)) / self.AC_K_Q10

        # /* ultrarapid_delayed_rectifier_K_current_ui_gate */
        AV_alpha_ui = pow(21.0 + 1.0 * ufl.exp((NV_Ith_S(y, 0) - (-10.0) - 195.0) / (-28.0)), (-1.0))
        AV_beta_ui = 1.0 / ufl.exp((NV_Ith_S(y, 0) - (-10.0) - 168.0) / (-16.0))
        yinf[7] = pow(1.0 + ufl.exp((NV_Ith_S(y, 0) - (-10.0) - 109.45) / 27.48), (-1.0))
        tau[7] = pow(AV_alpha_ui + AV_beta_ui, (-1.0)) / self.AC_K_Q10

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
        AV_alpha_m = ufl.conditional(ufl.eq(NV_Ith_S(y, 0), (-47.13)), 3.2, 0.32 * (NV_Ith_S(y, 0) + 47.13) / (1.0 - ufl.exp((-0.1) * (NV_Ith_S(y, 0) + 47.13))))
        AV_beta_m = 0.08 * ufl.exp((-NV_Ith_S(y, 0)) / 11.0)
        AV_m_inf = AV_alpha_m / (AV_alpha_m + AV_beta_m)
        AV_tau_m = 1.0 / (AV_alpha_m + AV_beta_m)
        ydot[1] = (AV_m_inf - NV_Ith_S(y, 1)) / AV_tau_m

        return self.expression_list(ydot)

    @property
    def phi_f_stiff(self):
        y = self.y
        yinf = [None] * self.size
        tau = [None] * self.size

        # /* fast_sodium_current_m_gate */
        AV_alpha_m = ufl.conditional(ufl.eq(NV_Ith_S(y, 0), (-47.13)), 3.2, 0.32 * (NV_Ith_S(y, 0) + 47.13) / (1.0 - ufl.exp((-0.1) * (NV_Ith_S(y, 0) + 47.13))))
        AV_beta_m = 0.08 * ufl.exp((-NV_Ith_S(y, 0)) / 11.0)
        yinf[1] = AV_alpha_m / (AV_alpha_m + AV_beta_m)
        tau[1] = 1.0 / (AV_alpha_m + AV_beta_m)

        lmbda = [None] * self.size
        lmbda[1] = -1.0 / tau[1]

        return self.expression_list(self.apply_phi(lmbda, yinf))

    @property
    def f_nonstiff(self):
        y = self.y
        ydot = [None] * self.size

        # Linear (in the gating variables) terms

        # /* Ca_release_current_from_JSR_w_gate */
        AV_tau_w = ufl.conditional(
            ufl.lt(ufl.algebra.Abs(NV_Ith_S(y, 0) - 7.9), 1e-10),
            6.0 * 0.2 / 1.3,
            6.0 * (1.0 - ufl.exp((-(NV_Ith_S(y, 0) - 7.9)) / 5.0)) / ((1.0 + 0.3 * ufl.exp((-(NV_Ith_S(y, 0) - 7.9)) / 5.0)) * 1.0 * (NV_Ith_S(y, 0) - 7.9)),
        )
        AV_w_infinity = 1.0 - pow(1.0 + ufl.exp((-(NV_Ith_S(y, 0) - 40.0)) / 17.0), (-1.0))
        ydot[15] = (AV_w_infinity - NV_Ith_S(y, 15)) / AV_tau_w

        # /* L_type_Ca_channel_d_gate */
        AV_d_infinity = pow(1.0 + ufl.exp((NV_Ith_S(y, 0) + 10.0) / (-8.0)), (-1.0))
        AV_tau_d = ufl.conditional(
            ufl.lt(ufl.algebra.Abs(NV_Ith_S(y, 0) + 10.0), 1e-10),
            4.579 / (1.0 + ufl.exp((NV_Ith_S(y, 0) + 10.0) / (-6.24))),
            (1.0 - ufl.exp((NV_Ith_S(y, 0) + 10.0) / (-6.24))) / (0.035 * (NV_Ith_S(y, 0) + 10.0) * (1.0 + ufl.exp((NV_Ith_S(y, 0) + 10.0) / (-6.24)))),
        )
        ydot[10] = (AV_d_infinity - NV_Ith_S(y, 10)) / AV_tau_d

        # /* L_type_Ca_channel_f_gate */
        AV_f_infinity = ufl.exp((-(NV_Ith_S(y, 0) + 28.0)) / 6.9) / (1.0 + ufl.exp((-(NV_Ith_S(y, 0) + 28.0)) / 6.9))
        AV_tau_f = 9.0 * pow(0.0197 * ufl.exp((-pow(0.0337, 2.0)) * pow(NV_Ith_S(y, 0) + 10.0, 2.0)) + 0.02, (-1.0))
        ydot[11] = (AV_f_infinity - NV_Ith_S(y, 11)) / AV_tau_f

        # /* fast_sodium_current_h_gate */
        AV_alpha_h = ufl.conditional(ufl.lt(NV_Ith_S(y, 0), (-40.0)), 0.135 * ufl.exp((NV_Ith_S(y, 0) + 80.0) / (-6.8)), 0.0)
        AV_beta_h = ufl.conditional(
            ufl.lt(NV_Ith_S(y, 0), (-40.0)), 3.56 * ufl.exp(0.079 * NV_Ith_S(y, 0)) + 310000.0 * ufl.exp(0.35 * NV_Ith_S(y, 0)), 1.0 / (0.13 * (1.0 + ufl.exp((NV_Ith_S(y, 0) + 10.66) / (-11.1))))
        )
        AV_h_inf = AV_alpha_h / (AV_alpha_h + AV_beta_h)
        AV_tau_h = 1.0 / (AV_alpha_h + AV_beta_h)
        ydot[2] = (AV_h_inf - NV_Ith_S(y, 2)) / AV_tau_h

        # /* fast_sodium_current_j_gate */
        AV_alpha_j = ufl.conditional(
            ufl.lt(NV_Ith_S(y, 0), (-40.0)),
            ((-127140.0) * ufl.exp(0.2444 * NV_Ith_S(y, 0)) - 3.474e-05 * ufl.exp((-0.04391) * NV_Ith_S(y, 0))) * (NV_Ith_S(y, 0) + 37.78) / (1.0 + ufl.exp(0.311 * (NV_Ith_S(y, 0) + 79.23))),
            0.0,
        )
        AV_beta_j = ufl.conditional(
            ufl.lt(NV_Ith_S(y, 0), (-40.0)),
            0.1212 * ufl.exp((-0.01052) * NV_Ith_S(y, 0)) / (1.0 + ufl.exp((-0.1378) * (NV_Ith_S(y, 0) + 40.14))),
            0.3 * ufl.exp((-2.535e-07) * NV_Ith_S(y, 0)) / (1.0 + ufl.exp((-0.1) * (NV_Ith_S(y, 0) + 32.0))),
        )
        AV_j_inf = AV_alpha_j / (AV_alpha_j + AV_beta_j)
        AV_tau_j = 1.0 / (AV_alpha_j + AV_beta_j)
        ydot[3] = (AV_j_inf - NV_Ith_S(y, 3)) / AV_tau_j

        # /* fast_sodium_current_m_gate */
        # this is computed in self.f_stiff

        # /* rapid_delayed_rectifier_K_current_xr_gate */
        AV_alpha_xr = ufl.conditional(ufl.lt(ufl.algebra.Abs(NV_Ith_S(y, 0) + 14.1), 1e-10), 0.0015, 0.0003 * (NV_Ith_S(y, 0) + 14.1) / (1.0 - ufl.exp((NV_Ith_S(y, 0) + 14.1) / (-5.0))))
        AV_beta_xr = ufl.conditional(
            ufl.lt(ufl.algebra.Abs(NV_Ith_S(y, 0) - 3.3328), 1e-10), 3.78361180000000004e-04, 7.38980000000000030e-05 * (NV_Ith_S(y, 0) - 3.3328) / (ufl.exp((NV_Ith_S(y, 0) - 3.3328) / 5.1237) - 1.0)
        )
        AV_xr_infinity = pow(1.0 + ufl.exp((NV_Ith_S(y, 0) + 14.1) / (-6.5)), (-1.0))
        AV_tau_xr = pow(AV_alpha_xr + AV_beta_xr, (-1.0))
        ydot[8] = (AV_xr_infinity - NV_Ith_S(y, 8)) / AV_tau_xr

        # /* slow_delayed_rectifier_K_current_xs_gate */
        AV_alpha_xs = ufl.conditional(ufl.lt(ufl.algebra.Abs(NV_Ith_S(y, 0) - 19.9), 1e-10), 0.00068, 4e-05 * (NV_Ith_S(y, 0) - 19.9) / (1.0 - ufl.exp((NV_Ith_S(y, 0) - 19.9) / (-17.0))))
        AV_beta_xs = ufl.conditional(ufl.lt(ufl.algebra.Abs(NV_Ith_S(y, 0) - 19.9), 1e-10), 0.000315, 3.5e-05 * (NV_Ith_S(y, 0) - 19.9) / (ufl.exp((NV_Ith_S(y, 0) - 19.9) / 9.0) - 1.0))
        AV_xs_infinity = pow(1.0 + ufl.exp((NV_Ith_S(y, 0) - 19.9) / (-12.7)), (-0.5))
        AV_tau_xs = 0.5 * pow(AV_alpha_xs + AV_beta_xs, (-1.0))
        ydot[9] = (AV_xs_infinity - NV_Ith_S(y, 9)) / AV_tau_xs

        # /* transient_outward_K_current_oa_gate */
        AV_alpha_oa = 0.65 * pow(ufl.exp((NV_Ith_S(y, 0) - (-10.0)) / (-8.5)) + ufl.exp((NV_Ith_S(y, 0) - (-10.0) - 40.0) / (-59.0)), (-1.0))
        AV_beta_oa = 0.65 * pow(2.5 + ufl.exp((NV_Ith_S(y, 0) - (-10.0) + 72.0) / 17.0), (-1.0))
        AV_oa_infinity = pow(1.0 + ufl.exp((NV_Ith_S(y, 0) - (-10.0) + 10.47) / (-17.54)), (-1.0))
        AV_tau_oa = pow(AV_alpha_oa + AV_beta_oa, (-1.0)) / self.AC_K_Q10
        ydot[4] = (AV_oa_infinity - NV_Ith_S(y, 4)) / AV_tau_oa

        # /* transient_outward_K_current_oi_gate */
        AV_alpha_oi = pow(18.53 + 1.0 * ufl.exp((NV_Ith_S(y, 0) - (-10.0) + 103.7) / 10.95), (-1.0))
        AV_beta_oi = pow(35.56 + 1.0 * ufl.exp((NV_Ith_S(y, 0) - (-10.0) - 8.74) / (-7.44)), (-1.0))
        AV_oi_infinity = pow(1.0 + ufl.exp((NV_Ith_S(y, 0) - (-10.0) + 33.1) / 5.3), (-1.0))
        AV_tau_oi = pow(AV_alpha_oi + AV_beta_oi, (-1.0)) / self.AC_K_Q10
        ydot[5] = (AV_oi_infinity - NV_Ith_S(y, 5)) / AV_tau_oi

        # /* ultrarapid_delayed_rectifier_K_current_ua_gate */
        AV_alpha_ua = 0.65 * pow(ufl.exp((NV_Ith_S(y, 0) - (-10.0)) / (-8.5)) + ufl.exp((NV_Ith_S(y, 0) - (-10.0) - 40.0) / (-59.0)), (-1.0))
        AV_beta_ua = 0.65 * pow(2.5 + ufl.exp((NV_Ith_S(y, 0) - (-10.0) + 72.0) / 17.0), (-1.0))
        AV_ua_infinity = pow(1.0 + ufl.exp((NV_Ith_S(y, 0) - (-10.0) + 20.3) / (-9.6)), (-1.0))
        AV_tau_ua = pow(AV_alpha_ua + AV_beta_ua, (-1.0)) / self.AC_K_Q10
        ydot[6] = (AV_ua_infinity - NV_Ith_S(y, 6)) / AV_tau_ua

        # /* ultrarapid_delayed_rectifier_K_current_ui_gate */
        AV_alpha_ui = pow(21.0 + 1.0 * ufl.exp((NV_Ith_S(y, 0) - (-10.0) - 195.0) / (-28.0)), (-1.0))
        AV_beta_ui = 1.0 / ufl.exp((NV_Ith_S(y, 0) - (-10.0) - 168.0) / (-16.0))
        AV_ui_infinity = pow(1.0 + ufl.exp((NV_Ith_S(y, 0) - (-10.0) - 109.45) / 27.48), (-1.0))
        AV_tau_ui = pow(AV_alpha_ui + AV_beta_ui, (-1.0)) / self.AC_K_Q10
        ydot[7] = (AV_ui_infinity - NV_Ith_S(y, 7)) / AV_tau_ui

        # Non Linear (in the gating variables) terms

        # /* L_type_Ca_channel_f_Ca_gate */
        AV_f_Ca_infinity = pow(1.0 + NV_Ith_S(y, 17) / 0.00035, (-1.0))
        ydot[12] = (AV_f_Ca_infinity - NV_Ith_S(y, 12)) / self.AC_tau_f_Ca

        # /* transfer_current_from_NSR_to_JSR */
        AV_i_tr = (NV_Ith_S(y, 20) - NV_Ith_S(y, 19)) / self.AC_tau_tr

        # /* Ca_leak_current_by_the_NSR */
        AV_i_up_leak = self.AC_I_up_max * NV_Ith_S(y, 20) / self.AC_Ca_up_max

        # /* Ca_release_current_from_JSR */
        AV_i_rel = self.AC_K_rel * pow(NV_Ith_S(y, 13), 2.0) * NV_Ith_S(y, 14) * NV_Ith_S(y, 15) * (NV_Ith_S(y, 19) - NV_Ith_S(y, 17))

        # /* intracellular_ion_concentrations */
        ydot[19] = (AV_i_tr - AV_i_rel) * pow(1.0 + self.AC_CSQN_max * self.AC_Km_CSQN / pow(NV_Ith_S(y, 19) + self.AC_Km_CSQN, 2.0), (-1.0))

        # /* Ca_uptake_current_by_the_NSR */
        AV_i_up = self.AC_I_up_max / (1.0 + self.AC_K_up / NV_Ith_S(y, 17))
        ydot[20] = AV_i_up - (AV_i_up_leak + AV_i_tr * self.AC_V_rel / self.AC_V_up)

        # /* sarcolemmal_calcium_pump_current */
        AV_i_CaP = self.AC_Cm * self.AC_i_CaP_max * NV_Ith_S(y, 17) / (0.0005 + NV_Ith_S(y, 17))

        # /* sodium_potassium_pump */
        AV_f_NaK = pow(
            1.0 + 0.1245 * ufl.exp((-0.1) * self.AC_F * NV_Ith_S(y, 0) / (self.AC_R * self.AC_T)) + 0.0365 * self.AC_sigma * ufl.exp((-self.AC_F) * NV_Ith_S(y, 0) / (self.AC_R * self.AC_T)), (-1.0)
        )
        AV_i_NaK = self.AC_Cm * self.AC_i_NaK_max * AV_f_NaK * 1.0 / (1.0 + pow(self.AC_Km_Na_i / NV_Ith_S(y, 16), 1.5)) * self.AC_K_o / (self.AC_K_o + self.AC_Km_K_o)

        # /* time_independent_potassium_current */
        AV_E_K = self.AC_R * self.AC_T / self.AC_F * ufl.ln(self.AC_K_o / NV_Ith_S(y, 18))
        AV_i_K1 = self.AC_Cm * self.AC_g_K1 * (NV_Ith_S(y, 0) - AV_E_K) / (1.0 + ufl.exp(0.07 * (NV_Ith_S(y, 0) + 80.0)))

        # /* transient_outward_K_current */
        AV_i_to = self.AC_Cm * self.AC_g_to * pow(NV_Ith_S(y, 4), 3.0) * NV_Ith_S(y, 5) * (NV_Ith_S(y, 0) - AV_E_K)

        # /* ultrarapid_delayed_rectifier_K_current */
        AV_g_Kur = 0.005 + 0.05 / (1.0 + ufl.exp((NV_Ith_S(y, 0) - 15.0) / (-13.0)))
        AV_i_Kur = self.AC_Cm * AV_g_Kur * pow(NV_Ith_S(y, 6), 3.0) * NV_Ith_S(y, 7) * (NV_Ith_S(y, 0) - AV_E_K)

        # /* *remaining* */
        AV_i_Ca_L = self.AC_Cm * self.AC_g_Ca_L * NV_Ith_S(y, 10) * NV_Ith_S(y, 11) * NV_Ith_S(y, 12) * (NV_Ith_S(y, 0) - 65.0)
        AV_i_NaCa = (
            self.AC_Cm
            * self.AC_I_NaCa_max
            * (
                ufl.exp(self.AC_Na_Ca_exchanger_current_gamma * self.AC_F * NV_Ith_S(y, 0) / (self.AC_R * self.AC_T)) * pow(NV_Ith_S(y, 16), 3.0) * self.AC_Ca_o
                - ufl.exp((self.AC_Na_Ca_exchanger_current_gamma - 1.0) * self.AC_F * NV_Ith_S(y, 0) / (self.AC_R * self.AC_T)) * pow(self.AC_Na_o, 3.0) * NV_Ith_S(y, 17)
            )
            / (
                (pow(self.AC_K_mNa, 3.0) + pow(self.AC_Na_o, 3.0))
                * (self.AC_K_mCa + self.AC_Ca_o)
                * (1.0 + self.AC_K_sat * ufl.exp((self.AC_Na_Ca_exchanger_current_gamma - 1.0) * NV_Ith_S(y, 0) * self.AC_F / (self.AC_R * self.AC_T)))
            )
        )
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
        AV_u_infinity = pow(1.0 + ufl.exp((-(AV_Fn - 3.41749999999999983e-13)) / 1.367e-15), (-1.0))
        AV_tau_v = 1.91 + 2.09 * pow(1.0 + ufl.exp((-(AV_Fn - 3.41749999999999983e-13)) / 1.367e-15), (-1.0))
        AV_v_infinity = 1.0 - pow(1.0 + ufl.exp((-(AV_Fn - 6.835e-14)) / 1.367e-15), (-1.0))
        ydot[16] = ((-3.0) * AV_i_NaK - (3.0 * AV_i_NaCa + AV_i_B_Na + AV_i_Na)) / (self.AC_V_i * self.AC_F)

        ydot[0] = self.scale * (-(AV_i_Na + AV_i_K1 + AV_i_to + AV_i_Kur + AV_i_Kr + AV_i_Ks + AV_i_B_Na + AV_i_B_Ca + AV_i_NaK + AV_i_CaP + AV_i_NaCa + AV_i_Ca_L)) / self.AC_Cm
        ydot[13] = (AV_u_infinity - NV_Ith_S(y, 13)) / self.AC_tau_u
        ydot[14] = (AV_v_infinity - NV_Ith_S(y, 14)) / AV_tau_v

        AV_B1 = (2.0 * AV_i_NaCa - (AV_i_CaP + AV_i_Ca_L + AV_i_B_Ca)) / (2.0 * self.AC_V_i * self.AC_F) + (self.AC_V_up * (AV_i_up_leak - AV_i_up) + AV_i_rel * self.AC_V_rel) / self.AC_V_i
        AV_B2 = 1.0 + self.AC_TRPN_max * self.AC_Km_TRPN / pow(NV_Ith_S(y, 17) + self.AC_Km_TRPN, 2.0) + self.AC_CMDN_max * self.AC_Km_CMDN / pow(NV_Ith_S(y, 17) + self.AC_Km_CMDN, 2.0)
        ydot[17] = AV_B1 / AV_B2

        return self.expression_list(ydot)
