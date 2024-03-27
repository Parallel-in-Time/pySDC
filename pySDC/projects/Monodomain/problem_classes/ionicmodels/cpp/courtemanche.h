#include <cmath>
#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "ionicmodel.h"

#ifndef COURTEMANCHE
#define COURTEMANCHE

class Courtemanche1998 : public IonicModel
{
public:
    Courtemanche1998(const double scale_);
    ~Courtemanche1998(){};
    void f(py::array_t<double> &y, py::array_t<double> &fy);
    void f_expl(py::array_t<double> &y, py::array_t<double> &fy);
    void lmbda_exp(py::array_t<double> &y_list, py::array_t<double> &lmbda_list);
    void lmbda_yinf_exp(py::array_t<double> &y_list, py::array_t<double> &lmbda_list, py::array_t<double> &yinf_list);
    py::list initial_values();
    double rho_f_expl();

private:
    double AC_CMDN_max, AC_CSQN_max, AC_Km_CMDN, AC_Km_CSQN, AC_Km_TRPN, AC_TRPN_max, AC_I_up_max, AC_K_up, AC_tau_f_Ca, AC_Ca_o, AC_K_o, AC_Na_o;
    double AC_tau_tr, AC_Ca_up_max, AC_K_rel, AC_tau_u, AC_g_Ca_L, AC_I_NaCa_max, AC_K_mCa, AC_K_mNa, AC_K_sat, AC_Na_Ca_exchanger_current_gamma;
    double AC_g_B_Ca, AC_g_B_K, AC_g_B_Na, AC_g_Na, AC_V_cell, AC_V_i, AC_V_rel, AC_V_up, AC_Cm, AC_F, AC_R, AC_T, AC_g_Kr, AC_i_CaP_max, AC_g_Ks, AC_Km_K_o;
    double AC_Km_Na_i, AC_i_NaK_max, AC_sigma, AC_g_K1, AC_K_Q10, AC_g_to;
};

Courtemanche1998::Courtemanche1998(const double scale_)
    : IonicModel(scale_)
{
    size = 21;

    AC_CMDN_max = 0.05;
    AC_CSQN_max = 10.0;
    AC_Km_CMDN = 0.00238;
    AC_Km_CSQN = 0.8;
    AC_Km_TRPN = 0.0005;
    AC_TRPN_max = 0.07;
    AC_I_up_max = 0.005;
    AC_K_up = 0.00092;
    AC_tau_f_Ca = 2.0;
    AC_Ca_o = 1.8;
    AC_K_o = 5.4;
    AC_Na_o = 140.0;
    AC_tau_tr = 180.0;
    AC_Ca_up_max = 15.0;
    AC_K_rel = 30.0;
    AC_tau_u = 8.0;
    AC_g_Ca_L = 0.12375;
    AC_I_NaCa_max = 1600.0;
    AC_K_mCa = 1.38;
    AC_K_mNa = 87.5;
    AC_K_sat = 0.1;
    AC_Na_Ca_exchanger_current_gamma = 0.35;
    AC_g_B_Ca = 0.001131;
    AC_g_B_K = 0.0;
    AC_g_B_Na = 6.74437500000000015e-04;
    AC_g_Na = 7.8;
    AC_V_cell = 20100.0;
    AC_V_i = AC_V_cell * 0.68;
    AC_V_rel = 0.0048 * AC_V_cell;
    AC_V_up = 0.0552 * AC_V_cell;
    AC_Cm = 1.0; // 100.0;
    AC_F = 96.4867;
    AC_R = 8.3143;
    AC_T = 310.0;
    AC_g_Kr = 2.94117649999999994e-02;
    AC_i_CaP_max = 0.275;
    AC_g_Ks = 1.29411759999999987e-01;
    AC_Km_K_o = 1.5;
    AC_Km_Na_i = 10.0;
    AC_i_NaK_max = 5.99338739999999981e-01;
    AC_sigma = 1.0 / 7.0 * (exp(AC_Na_o / 67.3) - 1.0);
    AC_g_K1 = 0.09;
    AC_K_Q10 = 3.0;
    AC_g_to = 0.1652;

    assign(f_expl_args, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
    assign(f_exp_args, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15});
    assign(f_expl_indeces, {0, 12, 13, 14, 16, 17, 18, 19, 20});
    assign(f_exp_indeces, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15});
}

py::list Courtemanche1998::initial_values()
{
    py::list y0(size);
    y0[0] = -81.18;
    y0[1] = 0.002908;
    y0[2] = 0.9649;
    y0[3] = 0.9775;
    y0[4] = 0.03043;
    y0[5] = 0.9992;
    y0[6] = 0.004966;
    y0[7] = 0.9986;
    y0[8] = 3.296e-05;
    y0[9] = 0.01869;
    y0[10] = 0.0001367;
    y0[11] = 0.9996;
    y0[12] = 0.7755;
    y0[13] = 2.35e-112;
    y0[14] = 1.0;
    y0[15] = 0.9992;
    y0[16] = 11.17;
    y0[17] = 0.0001013;
    y0[18] = 139.0;
    y0[19] = 1.488;
    y0[20] = 1.488;

    return y0;
}

void Courtemanche1998::f(py::array_t<double> &y_list, py::array_t<double> &fy_list)
{
    double *y_ptrs[size];
    double *fy_ptrs[size];
    size_t N;
    size_t n_dofs;
    get_raw_data(y_list, y_ptrs, N, n_dofs);
    get_raw_data(fy_list, fy_ptrs, N, n_dofs);

    double y[size];
    // For linear in gating var terms
    double AV_tau_w, AV_w_infinity, AV_d_infinity, AV_tau_d, AV_f_infinity, AV_tau_f, AV_alpha_h, AV_beta_h, AV_h_inf, AV_tau_h, AV_alpha_j, AV_beta_j, AV_j_inf, AV_tau_j;
    double AV_alpha_m, AV_beta_m, AV_m_inf, AV_tau_m, AV_alpha_xr, AV_beta_xr, AV_xr_infinity, AV_tau_xr, AV_alpha_xs, AV_beta_xs, AV_xs_infinity;
    double AV_tau_xs, AV_alpha_oa, AV_beta_oa, AV_oa_infinity, AV_tau_oa, AV_oi_infinity;
    double AV_alpha_oi, AV_beta_oi, AV_tau_oi, AV_alpha_ua, AV_beta_ua, AV_ua_infinity, AV_tau_ua, AV_alpha_ui, AV_beta_ui, AV_ui_infinity, AV_tau_ui;
    // for nonlinear
    double AV_f_Ca_infinity, AV_i_tr, AV_i_up_leak, AV_i_rel, AV_i_up, AV_i_CaP, AV_f_NaK, AV_i_NaK, AV_E_K, AV_i_K1, AV_i_to, AV_g_Kur, AV_i_Kur;
    double AV_i_Ca_L, AV_i_NaCa, AV_E_Ca, AV_i_B_K, AV_E_Na, AV_i_Kr, AV_i_Ks, AV_Fn, AV_i_B_Ca, AV_i_B_Na, AV_i_Na, AV_u_infinity, AV_tau_v, AV_v_infinity, AV_B1, AV_B2;
    // Remember to scale the first variable!!!
    for (unsigned j = 0; j < n_dofs; j++)
    {
        for (unsigned i = 0; i < size; i++)
            y[i] = y_ptrs[i][j];

        // # Linear (in the gating variables) terms

        // #/* Ca_release_current_from_JSR_w_gate */

        AV_tau_w = abs(NV_Ith_S(y, 0) - 7.9) < 1e-10 ? 6.0 * 0.2 / 1.3 : 6.0 * (1.0 - exp((-(NV_Ith_S(y, 0) - 7.9)) / 5.0)) / ((1.0 + 0.3 * exp((-(NV_Ith_S(y, 0) - 7.9)) / 5.0)) * 1.0 * (NV_Ith_S(y, 0) - 7.9));
        AV_w_infinity = 1.0 - pow(1.0 + exp((-(NV_Ith_S(y, 0) - 40.0)) / 17.0), (-1.0));
        fy_ptrs[15][j] = (AV_w_infinity - NV_Ith_S(y, 15)) / AV_tau_w;

        // #/* L_type_Ca_channel_d_gate */
        AV_d_infinity = pow(1.0 + exp((NV_Ith_S(y, 0) + 10.0) / (-8.0)), (-1.0));
        AV_tau_d = abs(NV_Ith_S(y, 0) + 10.0) < 1e-10 ? 4.579 / (1.0 + exp((NV_Ith_S(y, 0) + 10.0) / (-6.24))) : (1.0 - exp((NV_Ith_S(y, 0) + 10.0) / (-6.24))) / (0.035 * (NV_Ith_S(y, 0) + 10.0) * (1.0 + exp((NV_Ith_S(y, 0) + 10.0) / (-6.24))));
        fy_ptrs[10][j] = (AV_d_infinity - NV_Ith_S(y, 10)) / AV_tau_d;

        // #/* L_type_Ca_channel_f_gate */
        AV_f_infinity = exp((-(NV_Ith_S(y, 0) + 28.0)) / 6.9) / (1.0 + exp((-(NV_Ith_S(y, 0) + 28.0)) / 6.9));
        AV_tau_f = 9.0 * pow(0.0197 * exp((-pow(0.0337, 2.0)) * pow(NV_Ith_S(y, 0) + 10.0, 2.0)) + 0.02, (-1.0));
        fy_ptrs[11][j] = (AV_f_infinity - NV_Ith_S(y, 11)) / AV_tau_f;

        // #/* fast_sodium_current_h_gate */
        AV_alpha_h = NV_Ith_S(y, 0) < (-40.0) ? 0.135 * exp((NV_Ith_S(y, 0) + 80.0) / (-6.8)) : 0.0;
        AV_beta_h = NV_Ith_S(y, 0) < (-40.0) ? 3.56 * exp(0.079 * NV_Ith_S(y, 0)) + 310000.0 * exp(0.35 * NV_Ith_S(y, 0)) : 1.0 / (0.13 * (1.0 + exp((NV_Ith_S(y, 0) + 10.66) / (-11.1))));
        AV_h_inf = AV_alpha_h / (AV_alpha_h + AV_beta_h);
        AV_tau_h = 1.0 / (AV_alpha_h + AV_beta_h);
        fy_ptrs[2][j] = (AV_h_inf - NV_Ith_S(y, 2)) / AV_tau_h;

        // #/* fast_sodium_current_j_gate */
        AV_alpha_j = NV_Ith_S(y, 0) < (-40.0) ? ((-127140.0) * exp(0.2444 * NV_Ith_S(y, 0)) - 3.474e-05 * exp((-0.04391) * NV_Ith_S(y, 0))) * (NV_Ith_S(y, 0) + 37.78) / (1.0 + exp(0.311 * (NV_Ith_S(y, 0) + 79.23))) : 0.0;
        AV_beta_j = NV_Ith_S(y, 0) < (-40.0) ? 0.1212 * exp((-0.01052) * NV_Ith_S(y, 0)) / (1.0 + exp((-0.1378) * (NV_Ith_S(y, 0) + 40.14))) : 0.3 * exp((-2.535e-07) * NV_Ith_S(y, 0)) / (1.0 + exp((-0.1) * (NV_Ith_S(y, 0) + 32.0)));
        AV_j_inf = AV_alpha_j / (AV_alpha_j + AV_beta_j);
        AV_tau_j = 1.0 / (AV_alpha_j + AV_beta_j);
        fy_ptrs[3][j] = (AV_j_inf - NV_Ith_S(y, 3)) / AV_tau_j;

        // #/* fast_sodium_current_m_gate */
        AV_alpha_m = NV_Ith_S(y, 0) == (-47.13) ? 3.2 : 0.32 * (NV_Ith_S(y, 0) + 47.13) / (1.0 - exp((-0.1) * (NV_Ith_S(y, 0) + 47.13)));
        AV_beta_m = 0.08 * exp((-NV_Ith_S(y, 0)) / 11.0);
        AV_m_inf = AV_alpha_m / (AV_alpha_m + AV_beta_m);
        AV_tau_m = 1.0 / (AV_alpha_m + AV_beta_m);
        fy_ptrs[1][j] = (AV_m_inf - NV_Ith_S(y, 1)) / AV_tau_m;

        // #/* rapid_delayed_rectifier_K_current_xr_gate */
        AV_alpha_xr = abs(NV_Ith_S(y, 0) + 14.1) < 1e-10 ? 0.0015 : 0.0003 * (NV_Ith_S(y, 0) + 14.1) / (1.0 - exp((NV_Ith_S(y, 0) + 14.1) / (-5.0)));
        AV_beta_xr = abs(NV_Ith_S(y, 0) - 3.3328) < 1e-10 ? 3.78361180000000004e-04 : 7.38980000000000030e-05 * (NV_Ith_S(y, 0) - 3.3328) / (exp((NV_Ith_S(y, 0) - 3.3328) / 5.1237) - 1.0);
        AV_xr_infinity = pow(1.0 + exp((NV_Ith_S(y, 0) + 14.1) / (-6.5)), (-1.0));
        AV_tau_xr = pow(AV_alpha_xr + AV_beta_xr, (-1.0));
        fy_ptrs[8][j] = (AV_xr_infinity - NV_Ith_S(y, 8)) / AV_tau_xr;

        // #/* slow_delayed_rectifier_K_current_xs_gate */
        AV_alpha_xs = abs(NV_Ith_S(y, 0) - 19.9) < 1e-10 ? 0.00068 : 4e-05 * (NV_Ith_S(y, 0) - 19.9) / (1.0 - exp((NV_Ith_S(y, 0) - 19.9) / (-17.0)));
        AV_beta_xs = abs(NV_Ith_S(y, 0) - 19.9) < 1e-10 ? 0.000315 : 3.5e-05 * (NV_Ith_S(y, 0) - 19.9) / (exp((NV_Ith_S(y, 0) - 19.9) / 9.0) - 1.0);
        AV_xs_infinity = pow(1.0 + exp((NV_Ith_S(y, 0) - 19.9) / (-12.7)), (-0.5));
        AV_tau_xs = 0.5 * pow(AV_alpha_xs + AV_beta_xs, (-1.0));
        fy_ptrs[9][j] = (AV_xs_infinity - NV_Ith_S(y, 9)) / AV_tau_xs;

        // #/* transient_outward_K_current_oa_gate */
        AV_alpha_oa = 0.65 * pow(exp((NV_Ith_S(y, 0) - (-10.0)) / (-8.5)) + exp((NV_Ith_S(y, 0) - (-10.0) - 40.0) / (-59.0)), (-1.0));
        AV_beta_oa = 0.65 * pow(2.5 + exp((NV_Ith_S(y, 0) - (-10.0) + 72.0) / 17.0), (-1.0));
        AV_oa_infinity = pow(1.0 + exp((NV_Ith_S(y, 0) - (-10.0) + 10.47) / (-17.54)), (-1.0));
        AV_tau_oa = pow(AV_alpha_oa + AV_beta_oa, (-1.0)) / AC_K_Q10;
        fy_ptrs[4][j] = (AV_oa_infinity - NV_Ith_S(y, 4)) / AV_tau_oa;

        // #/* transient_outward_K_current_oi_gate */
        AV_alpha_oi = pow(18.53 + 1.0 * exp((NV_Ith_S(y, 0) - (-10.0) + 103.7) / 10.95), (-1.0));
        AV_beta_oi = pow(35.56 + 1.0 * exp((NV_Ith_S(y, 0) - (-10.0) - 8.74) / (-7.44)), (-1.0));
        AV_oi_infinity = pow(1.0 + exp((NV_Ith_S(y, 0) - (-10.0) + 33.1) / 5.3), (-1.0));
        AV_tau_oi = pow(AV_alpha_oi + AV_beta_oi, (-1.0)) / AC_K_Q10;
        fy_ptrs[5][j] = (AV_oi_infinity - NV_Ith_S(y, 5)) / AV_tau_oi;

        // #/* ultrarapid_delayed_rectifier_K_current_ua_gate */
        AV_alpha_ua = 0.65 * pow(exp((NV_Ith_S(y, 0) - (-10.0)) / (-8.5)) + exp((NV_Ith_S(y, 0) - (-10.0) - 40.0) / (-59.0)), (-1.0));
        AV_beta_ua = 0.65 * pow(2.5 + exp((NV_Ith_S(y, 0) - (-10.0) + 72.0) / 17.0), (-1.0));
        AV_ua_infinity = pow(1.0 + exp((NV_Ith_S(y, 0) - (-10.0) + 20.3) / (-9.6)), (-1.0));
        AV_tau_ua = pow(AV_alpha_ua + AV_beta_ua, (-1.0)) / AC_K_Q10;
        fy_ptrs[6][j] = (AV_ua_infinity - NV_Ith_S(y, 6)) / AV_tau_ua;

        // #/* ultrarapid_delayed_rectifier_K_current_ui_gate */
        AV_alpha_ui = pow(21.0 + 1.0 * exp((NV_Ith_S(y, 0) - (-10.0) - 195.0) / (-28.0)), (-1.0));
        AV_beta_ui = 1.0 / exp((NV_Ith_S(y, 0) - (-10.0) - 168.0) / (-16.0));
        AV_ui_infinity = pow(1.0 + exp((NV_Ith_S(y, 0) - (-10.0) - 109.45) / 27.48), (-1.0));
        AV_tau_ui = pow(AV_alpha_ui + AV_beta_ui, (-1.0)) / AC_K_Q10;
        fy_ptrs[7][j] = (AV_ui_infinity - NV_Ith_S(y, 7)) / AV_tau_ui;

        // # Non Linear (in the gating variables) terms

        // #/* L_type_Ca_channel_f_Ca_gate */
        AV_f_Ca_infinity = pow(1.0 + NV_Ith_S(y, 17) / 0.00035, (-1.0));
        fy_ptrs[12][j] = (AV_f_Ca_infinity - NV_Ith_S(y, 12)) / AC_tau_f_Ca;

        // #/* transfer_current_from_NSR_to_JSR */
        AV_i_tr = (NV_Ith_S(y, 20) - NV_Ith_S(y, 19)) / AC_tau_tr;

        // #/* Ca_leak_current_by_the_NSR */
        AV_i_up_leak = AC_I_up_max * NV_Ith_S(y, 20) / AC_Ca_up_max;

        // #/* Ca_release_current_from_JSR */
        AV_i_rel = AC_K_rel * pow(NV_Ith_S(y, 13), 2.0) * NV_Ith_S(y, 14) * NV_Ith_S(y, 15) * (NV_Ith_S(y, 19) - NV_Ith_S(y, 17));

        // #/* intracellular_ion_concentrations */
        fy_ptrs[19][j] = (AV_i_tr - AV_i_rel) * pow(1.0 + AC_CSQN_max * AC_Km_CSQN / pow(NV_Ith_S(y, 19) + AC_Km_CSQN, 2.0), (-1.0));

        // #/* Ca_uptake_current_by_the_NSR */
        AV_i_up = AC_I_up_max / (1.0 + AC_K_up / NV_Ith_S(y, 17));
        fy_ptrs[20][j] = AV_i_up - (AV_i_up_leak + AV_i_tr * AC_V_rel / AC_V_up);

        // #/* sarcolemmal_calcium_pump_current */
        AV_i_CaP = AC_Cm * AC_i_CaP_max * NV_Ith_S(y, 17) / (0.0005 + NV_Ith_S(y, 17));

        // #/* sodium_potassium_pump */
        AV_f_NaK = pow(1.0 + 0.1245 * exp((-0.1) * AC_F * NV_Ith_S(y, 0) / (AC_R * AC_T)) + 0.0365 * AC_sigma * exp((-AC_F) * NV_Ith_S(y, 0) / (AC_R * AC_T)), (-1.0));
        AV_i_NaK = AC_Cm * AC_i_NaK_max * AV_f_NaK * 1.0 / (1.0 + pow(AC_Km_Na_i / NV_Ith_S(y, 16), 1.5)) * AC_K_o / (AC_K_o + AC_Km_K_o);

        // #/* time_independent_potassium_current */
        AV_E_K = AC_R * AC_T / AC_F * log(AC_K_o / NV_Ith_S(y, 18));
        AV_i_K1 = AC_Cm * AC_g_K1 * (NV_Ith_S(y, 0) - AV_E_K) / (1.0 + exp(0.07 * (NV_Ith_S(y, 0) + 80.0)));

        // #/* transient_outward_K_current */
        AV_i_to = AC_Cm * AC_g_to * pow(NV_Ith_S(y, 4), 3.0) * NV_Ith_S(y, 5) * (NV_Ith_S(y, 0) - AV_E_K);

        // #/* ultrarapid_delayed_rectifier_K_current */
        AV_g_Kur = 0.005 + 0.05 / (1.0 + exp((NV_Ith_S(y, 0) - 15.0) / (-13.0)));
        AV_i_Kur = AC_Cm * AV_g_Kur * pow(NV_Ith_S(y, 6), 3.0) * NV_Ith_S(y, 7) * (NV_Ith_S(y, 0) - AV_E_K);

        // #/* *remaining* */
        AV_i_Ca_L = AC_Cm * AC_g_Ca_L * NV_Ith_S(y, 10) * NV_Ith_S(y, 11) * NV_Ith_S(y, 12) * (NV_Ith_S(y, 0) - 65.0);
        AV_i_NaCa = AC_Cm * AC_I_NaCa_max * (exp(AC_Na_Ca_exchanger_current_gamma * AC_F * NV_Ith_S(y, 0) / (AC_R * AC_T)) * pow(NV_Ith_S(y, 16), 3.0) * AC_Ca_o - exp((AC_Na_Ca_exchanger_current_gamma - 1.0) * AC_F * NV_Ith_S(y, 0) / (AC_R * AC_T)) * pow(AC_Na_o, 3.0) * NV_Ith_S(y, 17)) / ((pow(AC_K_mNa, 3.0) + pow(AC_Na_o, 3.0)) * (AC_K_mCa + AC_Ca_o) * (1.0 + AC_K_sat * exp((AC_Na_Ca_exchanger_current_gamma - 1.0) * NV_Ith_S(y, 0) * AC_F / (AC_R * AC_T))));
        AV_E_Ca = AC_R * AC_T / (2.0 * AC_F) * log(AC_Ca_o / NV_Ith_S(y, 17));
        AV_i_B_K = AC_Cm * AC_g_B_K * (NV_Ith_S(y, 0) - AV_E_K);
        AV_E_Na = AC_R * AC_T / AC_F * log(AC_Na_o / NV_Ith_S(y, 16));
        AV_i_Kr = AC_Cm * AC_g_Kr * NV_Ith_S(y, 8) * (NV_Ith_S(y, 0) - AV_E_K) / (1.0 + exp((NV_Ith_S(y, 0) + 15.0) / 22.4));
        AV_i_Ks = AC_Cm * AC_g_Ks * pow(NV_Ith_S(y, 9), 2.0) * (NV_Ith_S(y, 0) - AV_E_K);
        AV_Fn = 1000.0 * (1e-15 * AC_V_rel * AV_i_rel - 1e-15 / (2.0 * AC_F) * (0.5 * AV_i_Ca_L - 0.2 * AV_i_NaCa));
        AV_i_B_Ca = AC_Cm * AC_g_B_Ca * (NV_Ith_S(y, 0) - AV_E_Ca);
        AV_i_B_Na = AC_Cm * AC_g_B_Na * (NV_Ith_S(y, 0) - AV_E_Na);
        AV_i_Na = AC_Cm * AC_g_Na * pow(NV_Ith_S(y, 1), 3.0) * NV_Ith_S(y, 2) * NV_Ith_S(y, 3) * (NV_Ith_S(y, 0) - AV_E_Na);
        fy_ptrs[18][j] = (2.0 * AV_i_NaK - (AV_i_K1 + AV_i_to + AV_i_Kur + AV_i_Kr + AV_i_Ks + AV_i_B_K)) / (AC_V_i * AC_F);
        AV_u_infinity = pow(1.0 + exp((-(AV_Fn - 3.41749999999999983e-13)) / 1.367e-15), (-1.0));
        AV_tau_v = 1.91 + 2.09 * pow(1.0 + exp((-(AV_Fn - 3.41749999999999983e-13)) / 1.367e-15), (-1.0));
        AV_v_infinity = 1.0 - pow(1.0 + exp((-(AV_Fn - 6.835e-14)) / 1.367e-15), (-1.0));
        fy_ptrs[16][j] = ((-3.0) * AV_i_NaK - (3.0 * AV_i_NaCa + AV_i_B_Na + AV_i_Na)) / (AC_V_i * AC_F);

        fy_ptrs[0][j] = scale * (-(AV_i_Na + AV_i_K1 + AV_i_to + AV_i_Kur + AV_i_Kr + AV_i_Ks + AV_i_B_Na + AV_i_B_Ca + AV_i_NaK + AV_i_CaP + AV_i_NaCa + AV_i_Ca_L)) / AC_Cm;
        fy_ptrs[13][j] = (AV_u_infinity - NV_Ith_S(y, 13)) / AC_tau_u;
        fy_ptrs[14][j] = (AV_v_infinity - NV_Ith_S(y, 14)) / AV_tau_v;

        AV_B1 = (2.0 * AV_i_NaCa - (AV_i_CaP + AV_i_Ca_L + AV_i_B_Ca)) / (2.0 * AC_V_i * AC_F) + (AC_V_up * (AV_i_up_leak - AV_i_up) + AV_i_rel * AC_V_rel) / AC_V_i;
        AV_B2 = 1.0 + AC_TRPN_max * AC_Km_TRPN / pow(NV_Ith_S(y, 17) + AC_Km_TRPN, 2.0) + AC_CMDN_max * AC_Km_CMDN / pow(NV_Ith_S(y, 17) + AC_Km_CMDN, 2.0);
        fy_ptrs[17][j] = AV_B1 / AV_B2;
    }
}

void Courtemanche1998::f_expl(py::array_t<double> &y_list, py::array_t<double> &fy_list)
{
    double *y_ptrs[size];
    double *fy_ptrs[size];
    size_t N;
    size_t n_dofs;
    get_raw_data(y_list, y_ptrs, N, n_dofs);
    get_raw_data(fy_list, fy_ptrs, N, n_dofs);

    double y[size];
    // for nonlinear
    double AV_f_Ca_infinity, AV_i_tr, AV_i_up_leak, AV_i_rel, AV_i_up, AV_i_CaP, AV_f_NaK, AV_i_NaK, AV_E_K, AV_i_K1, AV_i_to, AV_g_Kur, AV_i_Kur;
    double AV_i_Ca_L, AV_i_NaCa, AV_E_Ca, AV_i_B_K, AV_E_Na, AV_i_Kr, AV_i_Ks, AV_Fn, AV_i_B_Ca, AV_i_B_Na, AV_i_Na, AV_u_infinity, AV_tau_v, AV_v_infinity, AV_B1, AV_B2;
    // Remember to scale the first variable!!!
    for (unsigned j = 0; j < n_dofs; j++)
    {
        for (unsigned i = 0; i < size; i++)
            y[i] = y_ptrs[i][j];

        // #/* L_type_Ca_channel_f_Ca_gate */
        AV_f_Ca_infinity = pow(1.0 + NV_Ith_S(y, 17) / 0.00035, (-1.0));
        fy_ptrs[12][j] = (AV_f_Ca_infinity - NV_Ith_S(y, 12)) / AC_tau_f_Ca;

        // #/* transfer_current_from_NSR_to_JSR */
        AV_i_tr = (NV_Ith_S(y, 20) - NV_Ith_S(y, 19)) / AC_tau_tr;

        // #/* Ca_leak_current_by_the_NSR */
        AV_i_up_leak = AC_I_up_max * NV_Ith_S(y, 20) / AC_Ca_up_max;

        // #/* Ca_release_current_from_JSR */
        AV_i_rel = AC_K_rel * pow(NV_Ith_S(y, 13), 2.0) * NV_Ith_S(y, 14) * NV_Ith_S(y, 15) * (NV_Ith_S(y, 19) - NV_Ith_S(y, 17));

        // #/* intracellular_ion_concentrations */
        fy_ptrs[19][j] = (AV_i_tr - AV_i_rel) * pow(1.0 + AC_CSQN_max * AC_Km_CSQN / pow(NV_Ith_S(y, 19) + AC_Km_CSQN, 2.0), (-1.0));

        // #/* Ca_uptake_current_by_the_NSR */
        AV_i_up = AC_I_up_max / (1.0 + AC_K_up / NV_Ith_S(y, 17));
        fy_ptrs[20][j] = AV_i_up - (AV_i_up_leak + AV_i_tr * AC_V_rel / AC_V_up);

        // #/* sarcolemmal_calcium_pump_current */
        AV_i_CaP = AC_Cm * AC_i_CaP_max * NV_Ith_S(y, 17) / (0.0005 + NV_Ith_S(y, 17));

        // #/* sodium_potassium_pump */
        AV_f_NaK = pow(1.0 + 0.1245 * exp((-0.1) * AC_F * NV_Ith_S(y, 0) / (AC_R * AC_T)) + 0.0365 * AC_sigma * exp((-AC_F) * NV_Ith_S(y, 0) / (AC_R * AC_T)), (-1.0));
        AV_i_NaK = AC_Cm * AC_i_NaK_max * AV_f_NaK * 1.0 / (1.0 + pow(AC_Km_Na_i / NV_Ith_S(y, 16), 1.5)) * AC_K_o / (AC_K_o + AC_Km_K_o);

        // #/* time_independent_potassium_current */
        AV_E_K = AC_R * AC_T / AC_F * log(AC_K_o / NV_Ith_S(y, 18));
        AV_i_K1 = AC_Cm * AC_g_K1 * (NV_Ith_S(y, 0) - AV_E_K) / (1.0 + exp(0.07 * (NV_Ith_S(y, 0) + 80.0)));

        // #/* transient_outward_K_current */
        AV_i_to = AC_Cm * AC_g_to * pow(NV_Ith_S(y, 4), 3.0) * NV_Ith_S(y, 5) * (NV_Ith_S(y, 0) - AV_E_K);

        // #/* ultrarapid_delayed_rectifier_K_current */
        AV_g_Kur = 0.005 + 0.05 / (1.0 + exp((NV_Ith_S(y, 0) - 15.0) / (-13.0)));
        AV_i_Kur = AC_Cm * AV_g_Kur * pow(NV_Ith_S(y, 6), 3.0) * NV_Ith_S(y, 7) * (NV_Ith_S(y, 0) - AV_E_K);

        // #/* *remaining* */
        AV_i_Ca_L = AC_Cm * AC_g_Ca_L * NV_Ith_S(y, 10) * NV_Ith_S(y, 11) * NV_Ith_S(y, 12) * (NV_Ith_S(y, 0) - 65.0);
        AV_i_NaCa = AC_Cm * AC_I_NaCa_max * (exp(AC_Na_Ca_exchanger_current_gamma * AC_F * NV_Ith_S(y, 0) / (AC_R * AC_T)) * pow(NV_Ith_S(y, 16), 3.0) * AC_Ca_o - exp((AC_Na_Ca_exchanger_current_gamma - 1.0) * AC_F * NV_Ith_S(y, 0) / (AC_R * AC_T)) * pow(AC_Na_o, 3.0) * NV_Ith_S(y, 17)) / ((pow(AC_K_mNa, 3.0) + pow(AC_Na_o, 3.0)) * (AC_K_mCa + AC_Ca_o) * (1.0 + AC_K_sat * exp((AC_Na_Ca_exchanger_current_gamma - 1.0) * NV_Ith_S(y, 0) * AC_F / (AC_R * AC_T))));
        AV_E_Ca = AC_R * AC_T / (2.0 * AC_F) * log(AC_Ca_o / NV_Ith_S(y, 17));
        AV_i_B_K = AC_Cm * AC_g_B_K * (NV_Ith_S(y, 0) - AV_E_K);
        AV_E_Na = AC_R * AC_T / AC_F * log(AC_Na_o / NV_Ith_S(y, 16));
        AV_i_Kr = AC_Cm * AC_g_Kr * NV_Ith_S(y, 8) * (NV_Ith_S(y, 0) - AV_E_K) / (1.0 + exp((NV_Ith_S(y, 0) + 15.0) / 22.4));
        AV_i_Ks = AC_Cm * AC_g_Ks * pow(NV_Ith_S(y, 9), 2.0) * (NV_Ith_S(y, 0) - AV_E_K);
        AV_Fn = 1000.0 * (1e-15 * AC_V_rel * AV_i_rel - 1e-15 / (2.0 * AC_F) * (0.5 * AV_i_Ca_L - 0.2 * AV_i_NaCa));
        AV_i_B_Ca = AC_Cm * AC_g_B_Ca * (NV_Ith_S(y, 0) - AV_E_Ca);
        AV_i_B_Na = AC_Cm * AC_g_B_Na * (NV_Ith_S(y, 0) - AV_E_Na);
        AV_i_Na = AC_Cm * AC_g_Na * pow(NV_Ith_S(y, 1), 3.0) * NV_Ith_S(y, 2) * NV_Ith_S(y, 3) * (NV_Ith_S(y, 0) - AV_E_Na);
        fy_ptrs[18][j] = (2.0 * AV_i_NaK - (AV_i_K1 + AV_i_to + AV_i_Kur + AV_i_Kr + AV_i_Ks + AV_i_B_K)) / (AC_V_i * AC_F);
        AV_u_infinity = pow(1.0 + exp((-(AV_Fn - 3.41749999999999983e-13)) / 1.367e-15), (-1.0));
        AV_tau_v = 1.91 + 2.09 * pow(1.0 + exp((-(AV_Fn - 3.41749999999999983e-13)) / 1.367e-15), (-1.0));
        AV_v_infinity = 1.0 - pow(1.0 + exp((-(AV_Fn - 6.835e-14)) / 1.367e-15), (-1.0));
        fy_ptrs[16][j] = ((-3.0) * AV_i_NaK - (3.0 * AV_i_NaCa + AV_i_B_Na + AV_i_Na)) / (AC_V_i * AC_F);

        fy_ptrs[0][j] = scale * (-(AV_i_Na + AV_i_K1 + AV_i_to + AV_i_Kur + AV_i_Kr + AV_i_Ks + AV_i_B_Na + AV_i_B_Ca + AV_i_NaK + AV_i_CaP + AV_i_NaCa + AV_i_Ca_L)) / AC_Cm;
        fy_ptrs[13][j] = (AV_u_infinity - NV_Ith_S(y, 13)) / AC_tau_u;
        fy_ptrs[14][j] = (AV_v_infinity - NV_Ith_S(y, 14)) / AV_tau_v;

        AV_B1 = (2.0 * AV_i_NaCa - (AV_i_CaP + AV_i_Ca_L + AV_i_B_Ca)) / (2.0 * AC_V_i * AC_F) + (AC_V_up * (AV_i_up_leak - AV_i_up) + AV_i_rel * AC_V_rel) / AC_V_i;
        AV_B2 = 1.0 + AC_TRPN_max * AC_Km_TRPN / pow(NV_Ith_S(y, 17) + AC_Km_TRPN, 2.0) + AC_CMDN_max * AC_Km_CMDN / pow(NV_Ith_S(y, 17) + AC_Km_CMDN, 2.0);
        fy_ptrs[17][j] = AV_B1 / AV_B2;
    }
}

void Courtemanche1998::lmbda_yinf_exp(py::array_t<double> &y_list, py::array_t<double> &lmbda_list, py::array_t<double> &yinf_list)
{
    double *y_ptrs[size];
    double *lmbda_ptrs[size];
    double *yinf_ptrs[size];
    size_t N;
    size_t n_dofs;
    get_raw_data(y_list, y_ptrs, N, n_dofs);
    get_raw_data(lmbda_list, lmbda_ptrs, N, n_dofs);
    get_raw_data(yinf_list, yinf_ptrs, N, n_dofs);

    double y[size];
    double AV_tau_w, AV_tau_d, AV_tau_f, AV_alpha_h, AV_beta_h, AV_tau_h, AV_alpha_j, AV_beta_j, AV_tau_j;
    double AV_alpha_m, AV_beta_m, AV_tau_m, AV_alpha_xr, AV_beta_xr, AV_tau_xr, AV_alpha_xs, AV_beta_xs;
    double AV_tau_xs, AV_alpha_oa, AV_beta_oa, AV_tau_oa;
    double AV_alpha_oi, AV_beta_oi, AV_tau_oi, AV_alpha_ua, AV_beta_ua, AV_tau_ua, AV_alpha_ui, AV_beta_ui, AV_tau_ui;
    // Remember to scale the first variable!!!
    for (unsigned j = 0; j < n_dofs; j++)
    {
        for (unsigned i = 0; i < size; i++)
            y[i] = y_ptrs[i][j];

        // # Linear (in the gating variables) terms

        // #/* Ca_release_current_from_JSR_w_gate */

        AV_tau_w = abs(NV_Ith_S(y, 0) - 7.9) < 1e-10 ? 6.0 * 0.2 / 1.3 : 6.0 * (1.0 - exp((-(NV_Ith_S(y, 0) - 7.9)) / 5.0)) / ((1.0 + 0.3 * exp((-(NV_Ith_S(y, 0) - 7.9)) / 5.0)) * 1.0 * (NV_Ith_S(y, 0) - 7.9));
        lmbda_ptrs[15][j] = -1. / AV_tau_w;
        yinf_ptrs[15][j] = 1.0 - pow(1.0 + exp((-(NV_Ith_S(y, 0) - 40.0)) / 17.0), (-1.0));

        // #/* L_type_Ca_channel_d_gate */
        yinf_ptrs[10][j] = pow(1.0 + exp((NV_Ith_S(y, 0) + 10.0) / (-8.0)), (-1.0));
        AV_tau_d = abs(NV_Ith_S(y, 0) + 10.0) < 1e-10 ? 4.579 / (1.0 + exp((NV_Ith_S(y, 0) + 10.0) / (-6.24))) : (1.0 - exp((NV_Ith_S(y, 0) + 10.0) / (-6.24))) / (0.035 * (NV_Ith_S(y, 0) + 10.0) * (1.0 + exp((NV_Ith_S(y, 0) + 10.0) / (-6.24))));
        lmbda_ptrs[10][j] = -1. / AV_tau_d;

        // #/* L_type_Ca_channel_f_gate */
        yinf_ptrs[11][j] = exp((-(NV_Ith_S(y, 0) + 28.0)) / 6.9) / (1.0 + exp((-(NV_Ith_S(y, 0) + 28.0)) / 6.9));
        AV_tau_f = 9.0 * pow(0.0197 * exp((-pow(0.0337, 2.0)) * pow(NV_Ith_S(y, 0) + 10.0, 2.0)) + 0.02, (-1.0));
        lmbda_ptrs[11][j] = -1. / AV_tau_f;

        // #/* fast_sodium_current_h_gate */
        AV_alpha_h = NV_Ith_S(y, 0) < (-40.0) ? 0.135 * exp((NV_Ith_S(y, 0) + 80.0) / (-6.8)) : 0.0;
        AV_beta_h = NV_Ith_S(y, 0) < (-40.0) ? 3.56 * exp(0.079 * NV_Ith_S(y, 0)) + 310000.0 * exp(0.35 * NV_Ith_S(y, 0)) : 1.0 / (0.13 * (1.0 + exp((NV_Ith_S(y, 0) + 10.66) / (-11.1))));
        yinf_ptrs[2][j] = AV_alpha_h / (AV_alpha_h + AV_beta_h);
        AV_tau_h = 1.0 / (AV_alpha_h + AV_beta_h);
        lmbda_ptrs[2][j] = -1. / AV_tau_h;

        // #/* fast_sodium_current_j_gate */
        AV_alpha_j = NV_Ith_S(y, 0) < (-40.0) ? ((-127140.0) * exp(0.2444 * NV_Ith_S(y, 0)) - 3.474e-05 * exp((-0.04391) * NV_Ith_S(y, 0))) * (NV_Ith_S(y, 0) + 37.78) / (1.0 + exp(0.311 * (NV_Ith_S(y, 0) + 79.23))) : 0.0;
        AV_beta_j = NV_Ith_S(y, 0) < (-40.0) ? 0.1212 * exp((-0.01052) * NV_Ith_S(y, 0)) / (1.0 + exp((-0.1378) * (NV_Ith_S(y, 0) + 40.14))) : 0.3 * exp((-2.535e-07) * NV_Ith_S(y, 0)) / (1.0 + exp((-0.1) * (NV_Ith_S(y, 0) + 32.0)));
        yinf_ptrs[3][j] = AV_alpha_j / (AV_alpha_j + AV_beta_j);
        AV_tau_j = 1.0 / (AV_alpha_j + AV_beta_j);
        lmbda_ptrs[3][j] = -1. / AV_tau_j;

        // #/* fast_sodium_current_m_gate */
        AV_alpha_m = NV_Ith_S(y, 0) == (-47.13) ? 3.2 : 0.32 * (NV_Ith_S(y, 0) + 47.13) / (1.0 - exp((-0.1) * (NV_Ith_S(y, 0) + 47.13)));
        AV_beta_m = 0.08 * exp((-NV_Ith_S(y, 0)) / 11.0);
        yinf_ptrs[1][j] = AV_alpha_m / (AV_alpha_m + AV_beta_m);
        AV_tau_m = 1.0 / (AV_alpha_m + AV_beta_m);
        lmbda_ptrs[1][j] = -1. / AV_tau_m;

        // #/* rapid_delayed_rectifier_K_current_xr_gate */
        AV_alpha_xr = abs(NV_Ith_S(y, 0) + 14.1) < 1e-10 ? 0.0015 : 0.0003 * (NV_Ith_S(y, 0) + 14.1) / (1.0 - exp((NV_Ith_S(y, 0) + 14.1) / (-5.0)));
        AV_beta_xr = abs(NV_Ith_S(y, 0) - 3.3328) < 1e-10 ? 3.78361180000000004e-04 : 7.38980000000000030e-05 * (NV_Ith_S(y, 0) - 3.3328) / (exp((NV_Ith_S(y, 0) - 3.3328) / 5.1237) - 1.0);
        yinf_ptrs[8][j] = pow(1.0 + exp((NV_Ith_S(y, 0) + 14.1) / (-6.5)), (-1.0));
        AV_tau_xr = pow(AV_alpha_xr + AV_beta_xr, (-1.0));
        lmbda_ptrs[8][j] = -1. / AV_tau_xr;

        // #/* slow_delayed_rectifier_K_current_xs_gate */
        AV_alpha_xs = abs(NV_Ith_S(y, 0) - 19.9) < 1e-10 ? 0.00068 : 4e-05 * (NV_Ith_S(y, 0) - 19.9) / (1.0 - exp((NV_Ith_S(y, 0) - 19.9) / (-17.0)));
        AV_beta_xs = abs(NV_Ith_S(y, 0) - 19.9) < 1e-10 ? 0.000315 : 3.5e-05 * (NV_Ith_S(y, 0) - 19.9) / (exp((NV_Ith_S(y, 0) - 19.9) / 9.0) - 1.0);
        yinf_ptrs[9][j] = pow(1.0 + exp((NV_Ith_S(y, 0) - 19.9) / (-12.7)), (-0.5));
        AV_tau_xs = 0.5 * pow(AV_alpha_xs + AV_beta_xs, (-1.0));
        lmbda_ptrs[9][j] = -1. / AV_tau_xs;

        // #/* transient_outward_K_current_oa_gate */
        AV_alpha_oa = 0.65 * pow(exp((NV_Ith_S(y, 0) - (-10.0)) / (-8.5)) + exp((NV_Ith_S(y, 0) - (-10.0) - 40.0) / (-59.0)), (-1.0));
        AV_beta_oa = 0.65 * pow(2.5 + exp((NV_Ith_S(y, 0) - (-10.0) + 72.0) / 17.0), (-1.0));
        yinf_ptrs[4][j] = pow(1.0 + exp((NV_Ith_S(y, 0) - (-10.0) + 10.47) / (-17.54)), (-1.0));
        AV_tau_oa = pow(AV_alpha_oa + AV_beta_oa, (-1.0)) / AC_K_Q10;
        lmbda_ptrs[4][j] = -1. / AV_tau_oa;

        // #/* transient_outward_K_current_oi_gate */
        AV_alpha_oi = pow(18.53 + 1.0 * exp((NV_Ith_S(y, 0) - (-10.0) + 103.7) / 10.95), (-1.0));
        AV_beta_oi = pow(35.56 + 1.0 * exp((NV_Ith_S(y, 0) - (-10.0) - 8.74) / (-7.44)), (-1.0));
        yinf_ptrs[5][j] = pow(1.0 + exp((NV_Ith_S(y, 0) - (-10.0) + 33.1) / 5.3), (-1.0));
        AV_tau_oi = pow(AV_alpha_oi + AV_beta_oi, (-1.0)) / AC_K_Q10;
        lmbda_ptrs[5][j] = -1. / AV_tau_oi;

        // #/* ultrarapid_delayed_rectifier_K_current_ua_gate */
        AV_alpha_ua = 0.65 * pow(exp((NV_Ith_S(y, 0) - (-10.0)) / (-8.5)) + exp((NV_Ith_S(y, 0) - (-10.0) - 40.0) / (-59.0)), (-1.0));
        AV_beta_ua = 0.65 * pow(2.5 + exp((NV_Ith_S(y, 0) - (-10.0) + 72.0) / 17.0), (-1.0));
        yinf_ptrs[6][j] = pow(1.0 + exp((NV_Ith_S(y, 0) - (-10.0) + 20.3) / (-9.6)), (-1.0));
        AV_tau_ua = pow(AV_alpha_ua + AV_beta_ua, (-1.0)) / AC_K_Q10;
        lmbda_ptrs[6][j] = -1. / AV_tau_ua;

        // #/* ultrarapid_delayed_rectifier_K_current_ui_gate */
        AV_alpha_ui = pow(21.0 + 1.0 * exp((NV_Ith_S(y, 0) - (-10.0) - 195.0) / (-28.0)), (-1.0));
        AV_beta_ui = 1.0 / exp((NV_Ith_S(y, 0) - (-10.0) - 168.0) / (-16.0));
        yinf_ptrs[7][j] = pow(1.0 + exp((NV_Ith_S(y, 0) - (-10.0) - 109.45) / 27.48), (-1.0));
        AV_tau_ui = pow(AV_alpha_ui + AV_beta_ui, (-1.0)) / AC_K_Q10;
        lmbda_ptrs[7][j] = -1. / AV_tau_ui;
    }
}

void Courtemanche1998::lmbda_exp(py::array_t<double> &y_list, py::array_t<double> &lmbda_list)
{
    double *y_ptrs[size];
    double *lmbda_ptrs[size];
    size_t N;
    size_t n_dofs;
    get_raw_data(y_list, y_ptrs, N, n_dofs);
    get_raw_data(lmbda_list, lmbda_ptrs, N, n_dofs);

    double y[size];
    double AV_tau_w, AV_tau_d, AV_tau_f, AV_alpha_h, AV_beta_h, AV_tau_h, AV_alpha_j, AV_beta_j, AV_tau_j;
    double AV_alpha_m, AV_beta_m, AV_tau_m, AV_alpha_xr, AV_beta_xr, AV_tau_xr, AV_alpha_xs, AV_beta_xs;
    double AV_tau_xs, AV_alpha_oa, AV_beta_oa, AV_tau_oa;
    double AV_alpha_oi, AV_beta_oi, AV_tau_oi, AV_alpha_ua, AV_beta_ua, AV_tau_ua, AV_alpha_ui, AV_beta_ui, AV_tau_ui;
    // Remember to scale the first variable!!!
    for (unsigned j = 0; j < n_dofs; j++)
    {
        for (unsigned i = 0; i < size; i++)
            y[i] = y_ptrs[i][j];

        // # Linear (in the gating variables) terms

        // #/* Ca_release_current_from_JSR_w_gate */

        AV_tau_w = abs(NV_Ith_S(y, 0) - 7.9) < 1e-10 ? 6.0 * 0.2 / 1.3 : 6.0 * (1.0 - exp((-(NV_Ith_S(y, 0) - 7.9)) / 5.0)) / ((1.0 + 0.3 * exp((-(NV_Ith_S(y, 0) - 7.9)) / 5.0)) * 1.0 * (NV_Ith_S(y, 0) - 7.9));
        lmbda_ptrs[15][j] = -1. / AV_tau_w;

        // #/* L_type_Ca_channel_d_gate */
        AV_tau_d = abs(NV_Ith_S(y, 0) + 10.0) < 1e-10 ? 4.579 / (1.0 + exp((NV_Ith_S(y, 0) + 10.0) / (-6.24))) : (1.0 - exp((NV_Ith_S(y, 0) + 10.0) / (-6.24))) / (0.035 * (NV_Ith_S(y, 0) + 10.0) * (1.0 + exp((NV_Ith_S(y, 0) + 10.0) / (-6.24))));
        lmbda_ptrs[10][j] = -1. / AV_tau_d;

        // #/* L_type_Ca_channel_f_gate */
        AV_tau_f = 9.0 * pow(0.0197 * exp((-pow(0.0337, 2.0)) * pow(NV_Ith_S(y, 0) + 10.0, 2.0)) + 0.02, (-1.0));
        lmbda_ptrs[11][j] = -1. / AV_tau_f;

        // #/* fast_sodium_current_h_gate */
        AV_alpha_h = NV_Ith_S(y, 0) < (-40.0) ? 0.135 * exp((NV_Ith_S(y, 0) + 80.0) / (-6.8)) : 0.0;
        AV_beta_h = NV_Ith_S(y, 0) < (-40.0) ? 3.56 * exp(0.079 * NV_Ith_S(y, 0)) + 310000.0 * exp(0.35 * NV_Ith_S(y, 0)) : 1.0 / (0.13 * (1.0 + exp((NV_Ith_S(y, 0) + 10.66) / (-11.1))));
        AV_tau_h = 1.0 / (AV_alpha_h + AV_beta_h);
        lmbda_ptrs[2][j] = -1. / AV_tau_h;

        // #/* fast_sodium_current_j_gate */
        AV_alpha_j = NV_Ith_S(y, 0) < (-40.0) ? ((-127140.0) * exp(0.2444 * NV_Ith_S(y, 0)) - 3.474e-05 * exp((-0.04391) * NV_Ith_S(y, 0))) * (NV_Ith_S(y, 0) + 37.78) / (1.0 + exp(0.311 * (NV_Ith_S(y, 0) + 79.23))) : 0.0;
        AV_beta_j = NV_Ith_S(y, 0) < (-40.0) ? 0.1212 * exp((-0.01052) * NV_Ith_S(y, 0)) / (1.0 + exp((-0.1378) * (NV_Ith_S(y, 0) + 40.14))) : 0.3 * exp((-2.535e-07) * NV_Ith_S(y, 0)) / (1.0 + exp((-0.1) * (NV_Ith_S(y, 0) + 32.0)));
        AV_tau_j = 1.0 / (AV_alpha_j + AV_beta_j);
        lmbda_ptrs[3][j] = -1. / AV_tau_j;

        // #/* fast_sodium_current_m_gate */
        AV_alpha_m = NV_Ith_S(y, 0) == (-47.13) ? 3.2 : 0.32 * (NV_Ith_S(y, 0) + 47.13) / (1.0 - exp((-0.1) * (NV_Ith_S(y, 0) + 47.13)));
        AV_beta_m = 0.08 * exp((-NV_Ith_S(y, 0)) / 11.0);
        AV_tau_m = 1.0 / (AV_alpha_m + AV_beta_m);
        lmbda_ptrs[1][j] = -1. / AV_tau_m;

        // #/* rapid_delayed_rectifier_K_current_xr_gate */
        AV_alpha_xr = abs(NV_Ith_S(y, 0) + 14.1) < 1e-10 ? 0.0015 : 0.0003 * (NV_Ith_S(y, 0) + 14.1) / (1.0 - exp((NV_Ith_S(y, 0) + 14.1) / (-5.0)));
        AV_beta_xr = abs(NV_Ith_S(y, 0) - 3.3328) < 1e-10 ? 3.78361180000000004e-04 : 7.38980000000000030e-05 * (NV_Ith_S(y, 0) - 3.3328) / (exp((NV_Ith_S(y, 0) - 3.3328) / 5.1237) - 1.0);
        AV_tau_xr = pow(AV_alpha_xr + AV_beta_xr, (-1.0));
        lmbda_ptrs[8][j] = -1. / AV_tau_xr;

        // #/* slow_delayed_rectifier_K_current_xs_gate */
        AV_alpha_xs = abs(NV_Ith_S(y, 0) - 19.9) < 1e-10 ? 0.00068 : 4e-05 * (NV_Ith_S(y, 0) - 19.9) / (1.0 - exp((NV_Ith_S(y, 0) - 19.9) / (-17.0)));
        AV_beta_xs = abs(NV_Ith_S(y, 0) - 19.9) < 1e-10 ? 0.000315 : 3.5e-05 * (NV_Ith_S(y, 0) - 19.9) / (exp((NV_Ith_S(y, 0) - 19.9) / 9.0) - 1.0);
        AV_tau_xs = 0.5 * pow(AV_alpha_xs + AV_beta_xs, (-1.0));
        lmbda_ptrs[9][j] = -1. / AV_tau_xs;

        // #/* transient_outward_K_current_oa_gate */
        AV_alpha_oa = 0.65 * pow(exp((NV_Ith_S(y, 0) - (-10.0)) / (-8.5)) + exp((NV_Ith_S(y, 0) - (-10.0) - 40.0) / (-59.0)), (-1.0));
        AV_beta_oa = 0.65 * pow(2.5 + exp((NV_Ith_S(y, 0) - (-10.0) + 72.0) / 17.0), (-1.0));
        AV_tau_oa = pow(AV_alpha_oa + AV_beta_oa, (-1.0)) / AC_K_Q10;
        lmbda_ptrs[4][j] = -1. / AV_tau_oa;

        // #/* transient_outward_K_current_oi_gate */
        AV_alpha_oi = pow(18.53 + 1.0 * exp((NV_Ith_S(y, 0) - (-10.0) + 103.7) / 10.95), (-1.0));
        AV_beta_oi = pow(35.56 + 1.0 * exp((NV_Ith_S(y, 0) - (-10.0) - 8.74) / (-7.44)), (-1.0));
        AV_tau_oi = pow(AV_alpha_oi + AV_beta_oi, (-1.0)) / AC_K_Q10;
        lmbda_ptrs[5][j] = -1. / AV_tau_oi;

        // #/* ultrarapid_delayed_rectifier_K_current_ua_gate */
        AV_alpha_ua = 0.65 * pow(exp((NV_Ith_S(y, 0) - (-10.0)) / (-8.5)) + exp((NV_Ith_S(y, 0) - (-10.0) - 40.0) / (-59.0)), (-1.0));
        AV_beta_ua = 0.65 * pow(2.5 + exp((NV_Ith_S(y, 0) - (-10.0) + 72.0) / 17.0), (-1.0));
        AV_tau_ua = pow(AV_alpha_ua + AV_beta_ua, (-1.0)) / AC_K_Q10;
        lmbda_ptrs[6][j] = -1. / AV_tau_ua;

        // #/* ultrarapid_delayed_rectifier_K_current_ui_gate */
        AV_alpha_ui = pow(21.0 + 1.0 * exp((NV_Ith_S(y, 0) - (-10.0) - 195.0) / (-28.0)), (-1.0));
        AV_beta_ui = 1.0 / exp((NV_Ith_S(y, 0) - (-10.0) - 168.0) / (-16.0));
        AV_tau_ui = pow(AV_alpha_ui + AV_beta_ui, (-1.0)) / AC_K_Q10;
        lmbda_ptrs[7][j] = -1. / AV_tau_ui;
    }
}

double Courtemanche1998::rho_f_expl()
{
    return 7.5;
}

#endif