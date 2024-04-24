#include <cmath>
#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "ionicmodel.h"

#ifndef TENTUSSCHER_SMOOTH
#define TENTUSSCHER_SMOOTH

/*
The original TenTusscher2006_epi model has if clauses in the right hand side, which makes it non-smooth.
This is the original TenTusscher2006_epi model, but where if clauses are removed in order to obtain a smooth right hand side.
This model is used only for convergence tests with high order methods, since with the original one the relative error (typically) stagnates at 1e-8.
*/

class TenTusscher2006_epi_smooth : public IonicModel
{
public:
    TenTusscher2006_epi_smooth(const double scale_);
    ~TenTusscher2006_epi_smooth(){};
    void f(py::array_t<double> &y, py::array_t<double> &fy);
    void f_expl(py::array_t<double> &y, py::array_t<double> &fy);
    void lmbda_exp(py::array_t<double> &y_list, py::array_t<double> &lmbda_list);
    void lmbda_yinf_exp(py::array_t<double> &y_list, py::array_t<double> &lmbda_list, py::array_t<double> &yinf_list);
    py::list initial_values();
    double rho_f_expl();

private:
    double AC_Cm, AC_K_pCa, AC_g_pCa, AC_g_CaL, AC_g_bca, AC_Buf_c, AC_Buf_sr, AC_Buf_ss, AC_Ca_o, AC_EC, AC_K_buf_c, AC_K_buf_sr, AC_K_buf_ss, AC_K_up, AC_V_leak, AC_V_rel, AC_V_sr, AC_V_ss, AC_V_xfer, AC_Vmax_up, AC_k1_prime, AC_k2_prime, AC_k3, AC_k4, AC_max_sr, AC_min_sr, AC_g_Na, AC_g_K1, AC_F, AC_R, AC_T, AC_V_c, AC_stim_amplitude, AC_K_o, AC_g_pK, AC_g_Kr, AC_P_kna, AC_g_Ks, AC_g_bna, AC_K_NaCa, AC_K_sat, AC_Km_Ca, AC_Km_Nai, AC_alpha, AC_sodium_calcium_exchanger_current_gamma, AC_Na_o, AC_K_mNa, AC_K_mk, AC_P_NaK, AC_g_to;
};

TenTusscher2006_epi_smooth::TenTusscher2006_epi_smooth(const double scale_)
    : IonicModel(scale_)
{
    size = 19;

    AC_Cm = 1.0; // 185.0;
    AC_K_pCa = 0.0005;
    AC_g_pCa = 0.1238;
    AC_g_CaL = 0.0398;
    AC_g_bca = 0.000592;
    AC_Buf_c = 0.2;
    AC_Buf_sr = 10.0;
    AC_Buf_ss = 0.4;
    AC_Ca_o = 2.0;
    AC_EC = 1.5;
    AC_K_buf_c = 0.001;
    AC_K_buf_sr = 0.3;
    AC_K_buf_ss = 0.00025;
    AC_K_up = 0.00025;
    AC_V_leak = 0.00036;
    AC_V_rel = 0.102;
    AC_V_sr = 1094.0;
    AC_V_ss = 54.68;
    AC_V_xfer = 0.0038;
    AC_Vmax_up = 0.006375;
    AC_k1_prime = 0.15;
    AC_k2_prime = 0.045;
    AC_k3 = 0.06;
    AC_k4 = 0.005;
    AC_max_sr = 2.5;
    AC_min_sr = 1.0;
    AC_g_Na = 14.838;
    AC_g_K1 = 5.405;
    AC_F = 96.485;
    AC_R = 8.314;
    AC_T = 310.0;
    AC_V_c = 16404.0;
    AC_stim_amplitude = (-52.0);
    AC_K_o = 5.4;
    AC_g_pK = 0.0146;
    AC_g_Kr = 0.153;
    AC_P_kna = 0.03;
    AC_g_Ks = 0.392;
    AC_g_bna = 0.00029;
    AC_K_NaCa = 1000.0;
    AC_K_sat = 0.1;
    AC_Km_Ca = 1.38;
    AC_Km_Nai = 87.5;
    AC_alpha = 2.5;
    AC_sodium_calcium_exchanger_current_gamma = 0.35;
    AC_Na_o = 140.0;
    AC_K_mNa = 40.0;
    AC_K_mk = 1.0;
    AC_P_NaK = 2.724;
    AC_g_to = 0.294;

    assign(f_expl_args, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18});
    assign(f_exp_args, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15});
    assign(f_expl_indeces, {0, 13, 14, 15, 16, 17, 18});
    assign(f_exp_indeces, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
}

py::list TenTusscher2006_epi_smooth::initial_values()
{
    py::list y0(size);
    y0[0] = -85.23;
    y0[1] = 0.00621;
    y0[2] = 0.4712;
    y0[3] = 0.0095;
    y0[4] = 0.00172;
    y0[5] = 0.7444;
    y0[6] = 0.7045;
    y0[7] = 3.373e-05;
    y0[8] = 0.7888;
    y0[9] = 0.9755;
    y0[10] = 0.9953;
    y0[11] = 0.999998;
    y0[12] = 2.42e-08;
    y0[13] = 0.000126;
    y0[14] = 3.64;
    y0[15] = 0.00036;
    y0[16] = 0.9073;
    y0[17] = 8.604;
    y0[18] = 136.89;

    return y0;
}

void TenTusscher2006_epi_smooth::f(py::array_t<double> &y_list, py::array_t<double> &fy_list)
{
    double *y_ptrs[size];
    double *fy_ptrs[size];
    size_t N;
    size_t n_dofs;
    get_raw_data(y_list, y_ptrs, N, n_dofs);
    get_raw_data(fy_list, fy_ptrs, N, n_dofs);

    double y[size];
    // # needed for linear in gating variables
    double AV_alpha_d, AV_beta_d, AV_d_inf, AV_gamma_d, AV_tau_d, AV_f2_inf, AV_tau_f2, AV_fCass_inf, AV_tau_fCass, AV_f_inf, AV_tau_f;
    double AV_alpha_h, AV_beta_h, AV_h_inf, AV_tau_h, AV_alpha_j, AV_beta_j, AV_j_inf, AV_tau_j, AV_alpha_m, AV_beta_m, AV_m_inf, AV_tau_m;
    double AV_alpha_xr1, AV_beta_xr1, AV_xr1_inf, AV_tau_xr1, AV_alpha_xr2, AV_beta_xr2, AV_xr2_inf, AV_tau_xr2, AV_alpha_xs, AV_beta_xs, AV_xs_inf, AV_tau_xs;
    double AV_r_inf, AV_tau_r, AV_s_inf, AV_tau_s;
    // # needed for nonlinear in gating variables
    double AV_f_JCa_i_free, AV_f_JCa_sr_free, AV_f_JCa_ss_free, AV_i_leak, AV_i_up, AV_i_xfer, AV_kcasr, AV_k1, AV_k2, AV_O, AV_i_rel, AV_ddt_Ca_sr_total;
    double AV_E_Ca, AV_E_K, AV_i_NaK, AV_i_to, AV_i_p_Ca, AV_i_CaL, AV_i_b_Ca, AV_alpha_K1, AV_beta_K1, AV_i_p_K, AV_i_Kr, AV_E_Ks, AV_E_Na, AV_i_NaCa;
    double AV_ddt_Ca_i_total, AV_ddt_Ca_ss_total, AV_i_Na, AV_i_K1, AV_xK1_inf, AV_i_Ks, AV_i_b_Na;
    // Remember to scale the first variable!!!
    for (unsigned j = 0; j < n_dofs; j++)
    {
        for (unsigned i = 0; i < size; i++)
            y[i] = y_ptrs[i][j];

        // # Linear in gating variables

        // # /* L_type_Ca_current_d_gate */
        AV_alpha_d = 1.4 / (1.0 + exp(((-35.0) - NV_Ith_S(y, 0)) / 13.0)) + 0.25;
        AV_beta_d = 1.4 / (1.0 + exp((NV_Ith_S(y, 0) + 5.0) / 5.0));
        AV_d_inf = 1.0 / (1.0 + exp(((-8.0) - NV_Ith_S(y, 0)) / 7.5));
        AV_gamma_d = 1.0 / (1.0 + exp((50.0 - NV_Ith_S(y, 0)) / 20.0));
        AV_tau_d = 1.0 * AV_alpha_d * AV_beta_d + AV_gamma_d;
        fy_ptrs[7][j] = (AV_d_inf - NV_Ith_S(y, 7)) / AV_tau_d;

        // # /* L_type_Ca_current_f2_gate */
        AV_f2_inf = 0.67 / (1.0 + exp((NV_Ith_S(y, 0) + 35.0) / 7.0)) + 0.33;
        AV_tau_f2 = 562.0 * exp((-pow(NV_Ith_S(y, 0) + 27.0, 2.0)) / 240.0) + 31.0 / (1.0 + exp((25.0 - NV_Ith_S(y, 0)) / 10.0)) + 80.0 / (1.0 + exp((NV_Ith_S(y, 0) + 30.0) / 10.0));
        fy_ptrs[9][j] = (AV_f2_inf - NV_Ith_S(y, 9)) / AV_tau_f2;

        // # /* L_type_Ca_current_fCass_gate */
        AV_fCass_inf = 0.6 / (1.0 + pow(NV_Ith_S(y, 15) / 0.05, 2.0)) + 0.4;
        AV_tau_fCass = 80.0 / (1.0 + pow(NV_Ith_S(y, 15) / 0.05, 2.0)) + 2.0;
        fy_ptrs[10][j] = (AV_fCass_inf - NV_Ith_S(y, 10)) / AV_tau_fCass;

        // # /* L_type_Ca_current_f_gate */
        AV_f_inf = 1.0 / (1.0 + exp((NV_Ith_S(y, 0) + 20.0) / 7.0));
        AV_tau_f = 1102.5 * exp((-pow(NV_Ith_S(y, 0) + 27.0, 2.0)) / 225.0) + 200.0 / (1.0 + exp((13.0 - NV_Ith_S(y, 0)) / 10.0)) + 180.0 / (1.0 + exp((NV_Ith_S(y, 0) + 30.0) / 10.0)) + 20.0;
        fy_ptrs[8][j] = (AV_f_inf - NV_Ith_S(y, 8)) / AV_tau_f;

        // # /* fast_sodium_current_h_gate */
        AV_alpha_h = 0.0;
        AV_beta_h = 0.77 / (0.13 * (1.0 + exp((NV_Ith_S(y, 0) + 10.66) / (-11.1))));
        AV_h_inf = 1.0 / pow(1.0 + exp((NV_Ith_S(y, 0) + 71.55) / 7.43), 2.0);
        AV_tau_h = 1.0 / (AV_alpha_h + AV_beta_h);
        fy_ptrs[5][j] = (AV_h_inf - NV_Ith_S(y, 5)) / AV_tau_h;

        // # /* fast_sodium_current_j_gate */
        AV_alpha_j = 0.0;
        AV_beta_j = 0.6 * exp(0.057 * NV_Ith_S(y, 0)) / (1.0 + exp((-0.1) * (NV_Ith_S(y, 0) + 32.0)));
        AV_j_inf = 1.0 / pow(1.0 + exp((NV_Ith_S(y, 0) + 71.55) / 7.43), 2.0);
        AV_tau_j = 1.0 / (AV_alpha_j + AV_beta_j);
        fy_ptrs[6][j] = (AV_j_inf - NV_Ith_S(y, 6)) / AV_tau_j;

        // # /* fast_sodium_current_m_gate */
        AV_alpha_m = 1.0 / (1.0 + exp(((-60.0) - NV_Ith_S(y, 0)) / 5.0));
        AV_beta_m = 0.1 / (1.0 + exp((NV_Ith_S(y, 0) + 35.0) / 5.0)) + 0.1 / (1.0 + exp((NV_Ith_S(y, 0) - 50.0) / 200.0));
        AV_m_inf = 1.0 / pow(1.0 + exp(((-56.86) - NV_Ith_S(y, 0)) / 9.03), 2.0);
        AV_tau_m = 1.0 * AV_alpha_m * AV_beta_m;
        fy_ptrs[4][j] = (AV_m_inf - NV_Ith_S(y, 4)) / AV_tau_m;

        // # /* rapid_time_dependent_potassium_current_Xr1_gate */
        AV_alpha_xr1 = 450.0 / (1.0 + exp(((-45.0) - NV_Ith_S(y, 0)) / 10.0));
        AV_beta_xr1 = 6.0 / (1.0 + exp((NV_Ith_S(y, 0) + 30.0) / 11.5));
        AV_xr1_inf = 1.0 / (1.0 + exp(((-26.0) - NV_Ith_S(y, 0)) / 7.0));
        AV_tau_xr1 = 1.0 * AV_alpha_xr1 * AV_beta_xr1;
        fy_ptrs[1][j] = (AV_xr1_inf - NV_Ith_S(y, 1)) / AV_tau_xr1;

        // # /* rapid_time_dependent_potassium_current_Xr2_gate */
        AV_alpha_xr2 = 3.0 / (1.0 + exp(((-60.0) - NV_Ith_S(y, 0)) / 20.0));
        AV_beta_xr2 = 1.12 / (1.0 + exp((NV_Ith_S(y, 0) - 60.0) / 20.0));
        AV_xr2_inf = 1.0 / (1.0 + exp((NV_Ith_S(y, 0) + 88.0) / 24.0));
        AV_tau_xr2 = 1.0 * AV_alpha_xr2 * AV_beta_xr2;
        fy_ptrs[2][j] = (AV_xr2_inf - NV_Ith_S(y, 2)) / AV_tau_xr2;

        // # /* slow_time_dependent_potassium_current_Xs_gate */
        AV_alpha_xs = 1400.0 / sqrt(1.0 + exp((5.0 - NV_Ith_S(y, 0)) / 6.0));
        AV_beta_xs = 1.0 / (1.0 + exp((NV_Ith_S(y, 0) - 35.0) / 15.0));
        AV_xs_inf = 1.0 / (1.0 + exp(((-5.0) - NV_Ith_S(y, 0)) / 14.0));
        AV_tau_xs = 1.0 * AV_alpha_xs * AV_beta_xs + 80.0;
        fy_ptrs[3][j] = (AV_xs_inf - NV_Ith_S(y, 3)) / AV_tau_xs;

        // # /* transient_outward_current_r_gate */
        AV_r_inf = 1.0 / (1.0 + exp((20.0 - NV_Ith_S(y, 0)) / 6.0));
        AV_tau_r = 9.5 * exp((-pow(NV_Ith_S(y, 0) + 40.0, 2.0)) / 1800.0) + 0.8;
        fy_ptrs[12][j] = (AV_r_inf - NV_Ith_S(y, 12)) / AV_tau_r;

        // # /* transient_outward_current_s_gate */
        AV_s_inf = 1.0 / (1.0 + exp((NV_Ith_S(y, 0) + 20.0) / 5.0));
        AV_tau_s = 85.0 * exp((-pow(NV_Ith_S(y, 0) + 45.0, 2.0)) / 320.0) + 5.0 / (1.0 + exp((NV_Ith_S(y, 0) - 20.0) / 5.0)) + 3.0;
        fy_ptrs[11][j] = (AV_s_inf - NV_Ith_S(y, 11)) / AV_tau_s;

        // # Non linear in gating variables

        // # /* calcium_dynamics */
        AV_f_JCa_i_free = 1.0 / (1.0 + AC_Buf_c * AC_K_buf_c / pow(NV_Ith_S(y, 13) + AC_K_buf_c, 2.0));
        AV_f_JCa_sr_free = 1.0 / (1.0 + AC_Buf_sr * AC_K_buf_sr / pow(NV_Ith_S(y, 14) + AC_K_buf_sr, 2.0));
        AV_f_JCa_ss_free = 1.0 / (1.0 + AC_Buf_ss * AC_K_buf_ss / pow(NV_Ith_S(y, 15) + AC_K_buf_ss, 2.0));
        AV_i_leak = AC_V_leak * (NV_Ith_S(y, 14) - NV_Ith_S(y, 13));
        AV_i_up = AC_Vmax_up / (1.0 + pow(AC_K_up, 2.0) / pow(NV_Ith_S(y, 13), 2.0));
        AV_i_xfer = AC_V_xfer * (NV_Ith_S(y, 15) - NV_Ith_S(y, 13));
        AV_kcasr = AC_max_sr - (AC_max_sr - AC_min_sr) / (1.0 + pow(AC_EC / NV_Ith_S(y, 14), 2.0));
        AV_k1 = AC_k1_prime / AV_kcasr;
        AV_k2 = AC_k2_prime * AV_kcasr;
        AV_O = AV_k1 * pow(NV_Ith_S(y, 15), 2.0) * NV_Ith_S(y, 16) / (AC_k3 + AV_k1 * pow(NV_Ith_S(y, 15), 2.0));
        fy_ptrs[16][j] = (-AV_k2) * NV_Ith_S(y, 15) * NV_Ith_S(y, 16) + AC_k4 * (1.0 - NV_Ith_S(y, 16));
        AV_i_rel = AC_V_rel * AV_O * (NV_Ith_S(y, 14) - NV_Ith_S(y, 15));
        AV_ddt_Ca_sr_total = AV_i_up - (AV_i_rel + AV_i_leak);
        fy_ptrs[14][j] = AV_ddt_Ca_sr_total * AV_f_JCa_sr_free;

        // # /* reversal_potentials */
        AV_E_Ca = 0.5 * AC_R * AC_T / AC_F * log(AC_Ca_o / NV_Ith_S(y, 13));
        AV_E_K = AC_R * AC_T / AC_F * log(AC_K_o / NV_Ith_S(y, 18));

        // # /* sodium_potassium_pump_current */
        AV_i_NaK = AC_P_NaK * AC_K_o / (AC_K_o + AC_K_mk) * NV_Ith_S(y, 17) / (NV_Ith_S(y, 17) + AC_K_mNa) / (1.0 + 0.1245 * exp((-0.1) * NV_Ith_S(y, 0) * AC_F / (AC_R * AC_T)) + 0.0353 * exp((-NV_Ith_S(y, 0)) * AC_F / (AC_R * AC_T)));

        // # /* transient_outward_current */
        AV_i_to = AC_g_to * NV_Ith_S(y, 12) * NV_Ith_S(y, 11) * (NV_Ith_S(y, 0) - AV_E_K);

        // # /* calcium_pump_current */
        AV_i_p_Ca = AC_g_pCa * NV_Ith_S(y, 13) / (NV_Ith_S(y, 13) + AC_K_pCa);

        // # /* *remaining* */
        AV_i_CaL = AC_g_CaL * NV_Ith_S(y, 7) * NV_Ith_S(y, 8) * NV_Ith_S(y, 9) * NV_Ith_S(y, 10) * 4.0 * (NV_Ith_S(y, 0) - 15.0) * pow(AC_F, 2.0) / (AC_R * AC_T) * (0.25 * NV_Ith_S(y, 15) * exp(2.0 * (NV_Ith_S(y, 0) - 15.0) * AC_F / (AC_R * AC_T)) - AC_Ca_o) / (exp(2.0 * (NV_Ith_S(y, 0) - 15.0) * AC_F / (AC_R * AC_T)) - 1.0);
        AV_i_b_Ca = AC_g_bca * (NV_Ith_S(y, 0) - AV_E_Ca);
        AV_alpha_K1 = 0.1 / (1.0 + exp(0.06 * (NV_Ith_S(y, 0) - AV_E_K - 200.0)));
        AV_beta_K1 = (3.0 * exp(0.0002 * (NV_Ith_S(y, 0) - AV_E_K + 100.0)) + exp(0.1 * (NV_Ith_S(y, 0) - AV_E_K - 10.0))) / (1.0 + exp((-0.5) * (NV_Ith_S(y, 0) - AV_E_K)));
        AV_i_p_K = AC_g_pK * (NV_Ith_S(y, 0) - AV_E_K) / (1.0 + exp((25.0 - NV_Ith_S(y, 0)) / 5.98));
        AV_i_Kr = AC_g_Kr * sqrt(AC_K_o / 5.4) * NV_Ith_S(y, 1) * NV_Ith_S(y, 2) * (NV_Ith_S(y, 0) - AV_E_K);
        AV_E_Ks = AC_R * AC_T / AC_F * log((AC_K_o + AC_P_kna * AC_Na_o) / (NV_Ith_S(y, 18) + AC_P_kna * NV_Ith_S(y, 17)));
        AV_E_Na = AC_R * AC_T / AC_F * log(AC_Na_o / NV_Ith_S(y, 17));
        AV_i_NaCa = AC_K_NaCa * (exp(AC_sodium_calcium_exchanger_current_gamma * NV_Ith_S(y, 0) * AC_F / (AC_R * AC_T)) * pow(NV_Ith_S(y, 17), 3.0) * AC_Ca_o - exp((AC_sodium_calcium_exchanger_current_gamma - 1.0) * NV_Ith_S(y, 0) * AC_F / (AC_R * AC_T)) * pow(AC_Na_o, 3.0) * NV_Ith_S(y, 13) * AC_alpha) / ((pow(AC_Km_Nai, 3.0) + pow(AC_Na_o, 3.0)) * (AC_Km_Ca + AC_Ca_o) * (1.0 + AC_K_sat * exp((AC_sodium_calcium_exchanger_current_gamma - 1.0) * NV_Ith_S(y, 0) * AC_F / (AC_R * AC_T))));
        AV_ddt_Ca_i_total = (-(AV_i_b_Ca + AV_i_p_Ca - 2.0 * AV_i_NaCa)) * AC_Cm / (2.0 * AC_V_c * AC_F) + (AV_i_leak - AV_i_up) * AC_V_sr / AC_V_c + AV_i_xfer;
        AV_ddt_Ca_ss_total = (-AV_i_CaL) * AC_Cm / (2.0 * AC_V_ss * AC_F) + AV_i_rel * AC_V_sr / AC_V_ss - AV_i_xfer * AC_V_c / AC_V_ss;
        AV_i_Na = AC_g_Na * pow(NV_Ith_S(y, 4), 3.0) * NV_Ith_S(y, 5) * NV_Ith_S(y, 6) * (NV_Ith_S(y, 0) - AV_E_Na);
        AV_xK1_inf = AV_alpha_K1 / (AV_alpha_K1 + AV_beta_K1);
        AV_i_Ks = AC_g_Ks * pow(NV_Ith_S(y, 3), 2.0) * (NV_Ith_S(y, 0) - AV_E_Ks);
        AV_i_b_Na = AC_g_bna * (NV_Ith_S(y, 0) - AV_E_Na);
        fy_ptrs[13][j] = AV_ddt_Ca_i_total * AV_f_JCa_i_free;
        fy_ptrs[15][j] = AV_ddt_Ca_ss_total * AV_f_JCa_ss_free;
        AV_i_K1 = AC_g_K1 * AV_xK1_inf * sqrt(AC_K_o / 5.4) * (NV_Ith_S(y, 0) - AV_E_K);
        fy_ptrs[17][j] = (-(AV_i_Na + AV_i_b_Na + 3.0 * AV_i_NaK + 3.0 * AV_i_NaCa)) / (AC_V_c * AC_F) * AC_Cm;
        fy_ptrs[0][j] = scale * (-(AV_i_K1 + AV_i_to + AV_i_Kr + AV_i_Ks + AV_i_CaL + AV_i_NaK + AV_i_Na + AV_i_b_Na + AV_i_NaCa + AV_i_b_Ca + AV_i_p_K + AV_i_p_Ca));
        fy_ptrs[18][j] = (-(AV_i_K1 + AV_i_to + AV_i_Kr + AV_i_Ks + AV_i_p_K - 2.0 * AV_i_NaK)) / (AC_V_c * AC_F) * AC_Cm;
    }
}

void TenTusscher2006_epi_smooth::f_expl(py::array_t<double> &y_list, py::array_t<double> &fy_list)
{
    double *y_ptrs[size];
    double *fy_ptrs[size];
    size_t N;
    size_t n_dofs;
    get_raw_data(y_list, y_ptrs, N, n_dofs);
    get_raw_data(fy_list, fy_ptrs, N, n_dofs);

    double y[size];
    // # needed for nonlinear in gating variables
    double AV_f_JCa_i_free, AV_f_JCa_sr_free, AV_f_JCa_ss_free, AV_i_leak, AV_i_up, AV_i_xfer, AV_kcasr, AV_k1, AV_k2, AV_O, AV_i_rel, AV_ddt_Ca_sr_total;
    double AV_E_Ca, AV_E_K, AV_i_NaK, AV_i_to, AV_i_p_Ca, AV_i_CaL, AV_i_b_Ca, AV_alpha_K1, AV_beta_K1, AV_i_p_K, AV_i_Kr, AV_E_Ks, AV_E_Na, AV_i_NaCa;
    double AV_ddt_Ca_i_total, AV_ddt_Ca_ss_total, AV_i_Na, AV_i_K1, AV_xK1_inf, AV_i_Ks, AV_i_b_Na;
    // Remember to scale the first variable!!!
    for (unsigned j = 0; j < n_dofs; j++)
    {
        for (unsigned i = 0; i < size; i++)
            y[i] = y_ptrs[i][j];

        // # Non linear in gating variables

        // # /* calcium_dynamics */
        AV_f_JCa_i_free = 1.0 / (1.0 + AC_Buf_c * AC_K_buf_c / pow(NV_Ith_S(y, 13) + AC_K_buf_c, 2.0));
        AV_f_JCa_sr_free = 1.0 / (1.0 + AC_Buf_sr * AC_K_buf_sr / pow(NV_Ith_S(y, 14) + AC_K_buf_sr, 2.0));
        AV_f_JCa_ss_free = 1.0 / (1.0 + AC_Buf_ss * AC_K_buf_ss / pow(NV_Ith_S(y, 15) + AC_K_buf_ss, 2.0));
        AV_i_leak = AC_V_leak * (NV_Ith_S(y, 14) - NV_Ith_S(y, 13));
        AV_i_up = AC_Vmax_up / (1.0 + pow(AC_K_up, 2.0) / pow(NV_Ith_S(y, 13), 2.0));
        AV_i_xfer = AC_V_xfer * (NV_Ith_S(y, 15) - NV_Ith_S(y, 13));
        AV_kcasr = AC_max_sr - (AC_max_sr - AC_min_sr) / (1.0 + pow(AC_EC / NV_Ith_S(y, 14), 2.0));
        AV_k1 = AC_k1_prime / AV_kcasr;
        AV_k2 = AC_k2_prime * AV_kcasr;
        AV_O = AV_k1 * pow(NV_Ith_S(y, 15), 2.0) * NV_Ith_S(y, 16) / (AC_k3 + AV_k1 * pow(NV_Ith_S(y, 15), 2.0));
        fy_ptrs[16][j] = (-AV_k2) * NV_Ith_S(y, 15) * NV_Ith_S(y, 16) + AC_k4 * (1.0 - NV_Ith_S(y, 16));
        AV_i_rel = AC_V_rel * AV_O * (NV_Ith_S(y, 14) - NV_Ith_S(y, 15));
        AV_ddt_Ca_sr_total = AV_i_up - (AV_i_rel + AV_i_leak);
        fy_ptrs[14][j] = AV_ddt_Ca_sr_total * AV_f_JCa_sr_free;

        // # /* reversal_potentials */
        AV_E_Ca = 0.5 * AC_R * AC_T / AC_F * log(AC_Ca_o / NV_Ith_S(y, 13));
        AV_E_K = AC_R * AC_T / AC_F * log(AC_K_o / NV_Ith_S(y, 18));

        // # /* sodium_potassium_pump_current */
        AV_i_NaK = AC_P_NaK * AC_K_o / (AC_K_o + AC_K_mk) * NV_Ith_S(y, 17) / (NV_Ith_S(y, 17) + AC_K_mNa) / (1.0 + 0.1245 * exp((-0.1) * NV_Ith_S(y, 0) * AC_F / (AC_R * AC_T)) + 0.0353 * exp((-NV_Ith_S(y, 0)) * AC_F / (AC_R * AC_T)));

        // # /* transient_outward_current */
        AV_i_to = AC_g_to * NV_Ith_S(y, 12) * NV_Ith_S(y, 11) * (NV_Ith_S(y, 0) - AV_E_K);

        // # /* calcium_pump_current */
        AV_i_p_Ca = AC_g_pCa * NV_Ith_S(y, 13) / (NV_Ith_S(y, 13) + AC_K_pCa);

        // # /* *remaining* */
        AV_i_CaL = AC_g_CaL * NV_Ith_S(y, 7) * NV_Ith_S(y, 8) * NV_Ith_S(y, 9) * NV_Ith_S(y, 10) * 4.0 * (NV_Ith_S(y, 0) - 15.0) * pow(AC_F, 2.0) / (AC_R * AC_T) * (0.25 * NV_Ith_S(y, 15) * exp(2.0 * (NV_Ith_S(y, 0) - 15.0) * AC_F / (AC_R * AC_T)) - AC_Ca_o) / (exp(2.0 * (NV_Ith_S(y, 0) - 15.0) * AC_F / (AC_R * AC_T)) - 1.0);
        AV_i_b_Ca = AC_g_bca * (NV_Ith_S(y, 0) - AV_E_Ca);
        AV_alpha_K1 = 0.1 / (1.0 + exp(0.06 * (NV_Ith_S(y, 0) - AV_E_K - 200.0)));
        AV_beta_K1 = (3.0 * exp(0.0002 * (NV_Ith_S(y, 0) - AV_E_K + 100.0)) + exp(0.1 * (NV_Ith_S(y, 0) - AV_E_K - 10.0))) / (1.0 + exp((-0.5) * (NV_Ith_S(y, 0) - AV_E_K)));
        AV_i_p_K = AC_g_pK * (NV_Ith_S(y, 0) - AV_E_K) / (1.0 + exp((25.0 - NV_Ith_S(y, 0)) / 5.98));
        AV_i_Kr = AC_g_Kr * sqrt(AC_K_o / 5.4) * NV_Ith_S(y, 1) * NV_Ith_S(y, 2) * (NV_Ith_S(y, 0) - AV_E_K);
        AV_E_Ks = AC_R * AC_T / AC_F * log((AC_K_o + AC_P_kna * AC_Na_o) / (NV_Ith_S(y, 18) + AC_P_kna * NV_Ith_S(y, 17)));
        AV_E_Na = AC_R * AC_T / AC_F * log(AC_Na_o / NV_Ith_S(y, 17));
        AV_i_NaCa = AC_K_NaCa * (exp(AC_sodium_calcium_exchanger_current_gamma * NV_Ith_S(y, 0) * AC_F / (AC_R * AC_T)) * pow(NV_Ith_S(y, 17), 3.0) * AC_Ca_o - exp((AC_sodium_calcium_exchanger_current_gamma - 1.0) * NV_Ith_S(y, 0) * AC_F / (AC_R * AC_T)) * pow(AC_Na_o, 3.0) * NV_Ith_S(y, 13) * AC_alpha) / ((pow(AC_Km_Nai, 3.0) + pow(AC_Na_o, 3.0)) * (AC_Km_Ca + AC_Ca_o) * (1.0 + AC_K_sat * exp((AC_sodium_calcium_exchanger_current_gamma - 1.0) * NV_Ith_S(y, 0) * AC_F / (AC_R * AC_T))));
        AV_ddt_Ca_i_total = (-(AV_i_b_Ca + AV_i_p_Ca - 2.0 * AV_i_NaCa)) * AC_Cm / (2.0 * AC_V_c * AC_F) + (AV_i_leak - AV_i_up) * AC_V_sr / AC_V_c + AV_i_xfer;
        AV_ddt_Ca_ss_total = (-AV_i_CaL) * AC_Cm / (2.0 * AC_V_ss * AC_F) + AV_i_rel * AC_V_sr / AC_V_ss - AV_i_xfer * AC_V_c / AC_V_ss;
        AV_i_Na = AC_g_Na * pow(NV_Ith_S(y, 4), 3.0) * NV_Ith_S(y, 5) * NV_Ith_S(y, 6) * (NV_Ith_S(y, 0) - AV_E_Na);
        AV_xK1_inf = AV_alpha_K1 / (AV_alpha_K1 + AV_beta_K1);
        AV_i_Ks = AC_g_Ks * pow(NV_Ith_S(y, 3), 2.0) * (NV_Ith_S(y, 0) - AV_E_Ks);
        AV_i_b_Na = AC_g_bna * (NV_Ith_S(y, 0) - AV_E_Na);
        fy_ptrs[13][j] = AV_ddt_Ca_i_total * AV_f_JCa_i_free;
        fy_ptrs[15][j] = AV_ddt_Ca_ss_total * AV_f_JCa_ss_free;
        AV_i_K1 = AC_g_K1 * AV_xK1_inf * sqrt(AC_K_o / 5.4) * (NV_Ith_S(y, 0) - AV_E_K);
        fy_ptrs[17][j] = (-(AV_i_Na + AV_i_b_Na + 3.0 * AV_i_NaK + 3.0 * AV_i_NaCa)) / (AC_V_c * AC_F) * AC_Cm;
        fy_ptrs[0][j] = scale * (-(AV_i_K1 + AV_i_to + AV_i_Kr + AV_i_Ks + AV_i_CaL + AV_i_NaK + AV_i_Na + AV_i_b_Na + AV_i_NaCa + AV_i_b_Ca + AV_i_p_K + AV_i_p_Ca));
        fy_ptrs[18][j] = (-(AV_i_K1 + AV_i_to + AV_i_Kr + AV_i_Ks + AV_i_p_K - 2.0 * AV_i_NaK)) / (AC_V_c * AC_F) * AC_Cm;
    }
}

void TenTusscher2006_epi_smooth::lmbda_yinf_exp(py::array_t<double> &y_list, py::array_t<double> &lmbda_list, py::array_t<double> &yinf_list)
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
    double AV_alpha_d, AV_beta_d, AV_gamma_d, AV_tau_d, AV_tau_f2, AV_tau_fCass, AV_tau_f;
    double AV_alpha_h, AV_beta_h, AV_tau_h, AV_alpha_j, AV_beta_j, AV_tau_j, AV_alpha_m, AV_beta_m, AV_tau_m;
    double AV_alpha_xr1, AV_beta_xr1, AV_tau_xr1, AV_alpha_xr2, AV_beta_xr2, AV_tau_xr2, AV_alpha_xs, AV_beta_xs, AV_tau_xs;
    double AV_tau_r, AV_tau_s;
    // Remember to scale the first variable!!!
    for (unsigned j = 0; j < n_dofs; j++)
    {
        for (unsigned i = 0; i < size; i++)
            y[i] = y_ptrs[i][j];

        // # Linear in gating variables

        // # /* L_type_Ca_current_d_gate */
        AV_alpha_d = 1.4 / (1.0 + exp(((-35.0) - NV_Ith_S(y, 0)) / 13.0)) + 0.25;
        AV_beta_d = 1.4 / (1.0 + exp((NV_Ith_S(y, 0) + 5.0) / 5.0));
        yinf_ptrs[7][j] = 1.0 / (1.0 + exp(((-8.0) - NV_Ith_S(y, 0)) / 7.5));
        AV_gamma_d = 1.0 / (1.0 + exp((50.0 - NV_Ith_S(y, 0)) / 20.0));
        AV_tau_d = 1.0 * AV_alpha_d * AV_beta_d + AV_gamma_d;
        lmbda_ptrs[7][j] = -1. / AV_tau_d;

        // # /* L_type_Ca_current_f2_gate */
        yinf_ptrs[9][j] = 0.67 / (1.0 + exp((NV_Ith_S(y, 0) + 35.0) / 7.0)) + 0.33;
        AV_tau_f2 = 562.0 * exp((-pow(NV_Ith_S(y, 0) + 27.0, 2.0)) / 240.0) + 31.0 / (1.0 + exp((25.0 - NV_Ith_S(y, 0)) / 10.0)) + 80.0 / (1.0 + exp((NV_Ith_S(y, 0) + 30.0) / 10.0));
        lmbda_ptrs[9][j] = -1. / AV_tau_f2;

        // # /* L_type_Ca_current_fCass_gate */
        yinf_ptrs[10][j] = 0.6 / (1.0 + pow(NV_Ith_S(y, 15) / 0.05, 2.0)) + 0.4;
        AV_tau_fCass = 80.0 / (1.0 + pow(NV_Ith_S(y, 15) / 0.05, 2.0)) + 2.0;
        lmbda_ptrs[10][j] = -1. / AV_tau_fCass;

        // # /* L_type_Ca_current_f_gate */
        yinf_ptrs[8][j] = 1.0 / (1.0 + exp((NV_Ith_S(y, 0) + 20.0) / 7.0));
        AV_tau_f = 1102.5 * exp((-pow(NV_Ith_S(y, 0) + 27.0, 2.0)) / 225.0) + 200.0 / (1.0 + exp((13.0 - NV_Ith_S(y, 0)) / 10.0)) + 180.0 / (1.0 + exp((NV_Ith_S(y, 0) + 30.0) / 10.0)) + 20.0;
        lmbda_ptrs[8][j] = -1. / AV_tau_f;

        // # /* fast_sodium_current_h_gate */
        AV_alpha_h = 0.0;
        AV_beta_h = 0.77 / (0.13 * (1.0 + exp((NV_Ith_S(y, 0) + 10.66) / (-11.1))));
        yinf_ptrs[5][j] = 1.0 / pow(1.0 + exp((NV_Ith_S(y, 0) + 71.55) / 7.43), 2.0);
        AV_tau_h = 1.0 / (AV_alpha_h + AV_beta_h);
        lmbda_ptrs[5][j] = -1. / AV_tau_h;

        // # /* fast_sodium_current_j_gate */
        AV_alpha_j = 0.0;
        AV_beta_j = 0.6 * exp(0.057 * NV_Ith_S(y, 0)) / (1.0 + exp((-0.1) * (NV_Ith_S(y, 0) + 32.0)));
        yinf_ptrs[6][j] = 1.0 / pow(1.0 + exp((NV_Ith_S(y, 0) + 71.55) / 7.43), 2.0);
        AV_tau_j = 1.0 / (AV_alpha_j + AV_beta_j);
        lmbda_ptrs[6][j] = -1. / AV_tau_j;

        // # /* fast_sodium_current_m_gate */
        AV_alpha_m = 1.0 / (1.0 + exp(((-60.0) - NV_Ith_S(y, 0)) / 5.0));
        AV_beta_m = 0.1 / (1.0 + exp((NV_Ith_S(y, 0) + 35.0) / 5.0)) + 0.1 / (1.0 + exp((NV_Ith_S(y, 0) - 50.0) / 200.0));
        yinf_ptrs[4][j] = 1.0 / pow(1.0 + exp(((-56.86) - NV_Ith_S(y, 0)) / 9.03), 2.0);
        AV_tau_m = 1.0 * AV_alpha_m * AV_beta_m;
        lmbda_ptrs[4][j] = -1. / AV_tau_m;

        // # /* rapid_time_dependent_potassium_current_Xr1_gate */
        AV_alpha_xr1 = 450.0 / (1.0 + exp(((-45.0) - NV_Ith_S(y, 0)) / 10.0));
        AV_beta_xr1 = 6.0 / (1.0 + exp((NV_Ith_S(y, 0) + 30.0) / 11.5));
        yinf_ptrs[1][j] = 1.0 / (1.0 + exp(((-26.0) - NV_Ith_S(y, 0)) / 7.0));
        AV_tau_xr1 = 1.0 * AV_alpha_xr1 * AV_beta_xr1;
        lmbda_ptrs[1][j] = -1. / AV_tau_xr1;

        // # /* rapid_time_dependent_potassium_current_Xr2_gate */
        AV_alpha_xr2 = 3.0 / (1.0 + exp(((-60.0) - NV_Ith_S(y, 0)) / 20.0));
        AV_beta_xr2 = 1.12 / (1.0 + exp((NV_Ith_S(y, 0) - 60.0) / 20.0));
        yinf_ptrs[2][j] = 1.0 / (1.0 + exp((NV_Ith_S(y, 0) + 88.0) / 24.0));
        AV_tau_xr2 = 1.0 * AV_alpha_xr2 * AV_beta_xr2;
        lmbda_ptrs[2][j] = -1. / AV_tau_xr2;

        // # /* slow_time_dependent_potassium_current_Xs_gate */
        AV_alpha_xs = 1400.0 / sqrt(1.0 + exp((5.0 - NV_Ith_S(y, 0)) / 6.0));
        AV_beta_xs = 1.0 / (1.0 + exp((NV_Ith_S(y, 0) - 35.0) / 15.0));
        yinf_ptrs[3][j] = 1.0 / (1.0 + exp(((-5.0) - NV_Ith_S(y, 0)) / 14.0));
        AV_tau_xs = 1.0 * AV_alpha_xs * AV_beta_xs + 80.0;
        lmbda_ptrs[3][j] = -1. / AV_tau_xs;

        // # /* transient_outward_current_r_gate */
        yinf_ptrs[12][j] = 1.0 / (1.0 + exp((20.0 - NV_Ith_S(y, 0)) / 6.0));
        AV_tau_r = 9.5 * exp((-pow(NV_Ith_S(y, 0) + 40.0, 2.0)) / 1800.0) + 0.8;
        lmbda_ptrs[12][j] = -1. / AV_tau_r;

        // # /* transient_outward_current_s_gate */
        yinf_ptrs[11][j] = 1.0 / (1.0 + exp((NV_Ith_S(y, 0) + 20.0) / 5.0));
        AV_tau_s = 85.0 * exp((-pow(NV_Ith_S(y, 0) + 45.0, 2.0)) / 320.0) + 5.0 / (1.0 + exp((NV_Ith_S(y, 0) - 20.0) / 5.0)) + 3.0;
        lmbda_ptrs[11][j] = -1. / AV_tau_s;
    }
}

void TenTusscher2006_epi_smooth::lmbda_exp(py::array_t<double> &y_list, py::array_t<double> &lmbda_list)
{
    double *y_ptrs[size];
    double *lmbda_ptrs[size];
    size_t N;
    size_t n_dofs;
    get_raw_data(y_list, y_ptrs, N, n_dofs);
    get_raw_data(lmbda_list, lmbda_ptrs, N, n_dofs);

    double y[size];
    double AV_alpha_d, AV_beta_d, AV_gamma_d, AV_tau_d, AV_tau_f2, AV_tau_fCass, AV_tau_f;
    double AV_alpha_h, AV_beta_h, AV_tau_h, AV_alpha_j, AV_beta_j, AV_tau_j, AV_alpha_m, AV_beta_m, AV_tau_m;
    double AV_alpha_xr1, AV_beta_xr1, AV_tau_xr1, AV_alpha_xr2, AV_beta_xr2, AV_tau_xr2, AV_alpha_xs, AV_beta_xs, AV_tau_xs;
    double AV_tau_r, AV_tau_s;
    // Remember to scale the first variable!!!
    for (unsigned j = 0; j < n_dofs; j++)
    {
        for (unsigned i = 0; i < size; i++)
            y[i] = y_ptrs[i][j];

        // # Linear in gating variables

        // # /* L_type_Ca_current_d_gate */
        AV_alpha_d = 1.4 / (1.0 + exp(((-35.0) - NV_Ith_S(y, 0)) / 13.0)) + 0.25;
        AV_beta_d = 1.4 / (1.0 + exp((NV_Ith_S(y, 0) + 5.0) / 5.0));
        AV_gamma_d = 1.0 / (1.0 + exp((50.0 - NV_Ith_S(y, 0)) / 20.0));
        AV_tau_d = 1.0 * AV_alpha_d * AV_beta_d + AV_gamma_d;
        lmbda_ptrs[7][j] = -1. / AV_tau_d;

        // # /* L_type_Ca_current_f2_gate */
        AV_tau_f2 = 562.0 * exp((-pow(NV_Ith_S(y, 0) + 27.0, 2.0)) / 240.0) + 31.0 / (1.0 + exp((25.0 - NV_Ith_S(y, 0)) / 10.0)) + 80.0 / (1.0 + exp((NV_Ith_S(y, 0) + 30.0) / 10.0));
        lmbda_ptrs[9][j] = -1. / AV_tau_f2;

        // # /* L_type_Ca_current_fCass_gate */
        AV_tau_fCass = 80.0 / (1.0 + pow(NV_Ith_S(y, 15) / 0.05, 2.0)) + 2.0;
        lmbda_ptrs[10][j] = -1. / AV_tau_fCass;

        // # /* L_type_Ca_current_f_gate */
        AV_tau_f = 1102.5 * exp((-pow(NV_Ith_S(y, 0) + 27.0, 2.0)) / 225.0) + 200.0 / (1.0 + exp((13.0 - NV_Ith_S(y, 0)) / 10.0)) + 180.0 / (1.0 + exp((NV_Ith_S(y, 0) + 30.0) / 10.0)) + 20.0;
        lmbda_ptrs[8][j] = -1. / AV_tau_f;

        // # /* fast_sodium_current_h_gate */
        AV_alpha_h = 0.0;
        AV_beta_h = 0.77 / (0.13 * (1.0 + exp((NV_Ith_S(y, 0) + 10.66) / (-11.1))));
        AV_tau_h = 1.0 / (AV_alpha_h + AV_beta_h);
        lmbda_ptrs[5][j] = -1. / AV_tau_h;

        // # /* fast_sodium_current_j_gate */
        AV_alpha_j = 0.0;
        AV_beta_j = 0.6 * exp(0.057 * NV_Ith_S(y, 0)) / (1.0 + exp((-0.1) * (NV_Ith_S(y, 0) + 32.0)));
        AV_tau_j = 1.0 / (AV_alpha_j + AV_beta_j);
        lmbda_ptrs[6][j] = -1. / AV_tau_j;

        // # /* fast_sodium_current_m_gate */
        AV_alpha_m = 1.0 / (1.0 + exp(((-60.0) - NV_Ith_S(y, 0)) / 5.0));
        AV_beta_m = 0.1 / (1.0 + exp((NV_Ith_S(y, 0) + 35.0) / 5.0)) + 0.1 / (1.0 + exp((NV_Ith_S(y, 0) - 50.0) / 200.0));
        AV_tau_m = 1.0 * AV_alpha_m * AV_beta_m;
        lmbda_ptrs[4][j] = -1. / AV_tau_m;

        // # /* rapid_time_dependent_potassium_current_Xr1_gate */
        AV_alpha_xr1 = 450.0 / (1.0 + exp(((-45.0) - NV_Ith_S(y, 0)) / 10.0));
        AV_beta_xr1 = 6.0 / (1.0 + exp((NV_Ith_S(y, 0) + 30.0) / 11.5));
        AV_tau_xr1 = 1.0 * AV_alpha_xr1 * AV_beta_xr1;
        lmbda_ptrs[1][j] = -1. / AV_tau_xr1;

        // # /* rapid_time_dependent_potassium_current_Xr2_gate */
        AV_alpha_xr2 = 3.0 / (1.0 + exp(((-60.0) - NV_Ith_S(y, 0)) / 20.0));
        AV_beta_xr2 = 1.12 / (1.0 + exp((NV_Ith_S(y, 0) - 60.0) / 20.0));
        AV_tau_xr2 = 1.0 * AV_alpha_xr2 * AV_beta_xr2;
        lmbda_ptrs[2][j] = -1. / AV_tau_xr2;

        // # /* slow_time_dependent_potassium_current_Xs_gate */
        AV_alpha_xs = 1400.0 / sqrt(1.0 + exp((5.0 - NV_Ith_S(y, 0)) / 6.0));
        AV_beta_xs = 1.0 / (1.0 + exp((NV_Ith_S(y, 0) - 35.0) / 15.0));
        AV_tau_xs = 1.0 * AV_alpha_xs * AV_beta_xs + 80.0;
        lmbda_ptrs[3][j] = -1. / AV_tau_xs;

        // # /* transient_outward_current_r_gate */
        AV_tau_r = 9.5 * exp((-pow(NV_Ith_S(y, 0) + 40.0, 2.0)) / 1800.0) + 0.8;
        lmbda_ptrs[12][j] = -1. / AV_tau_r;

        // # /* transient_outward_current_s_gate */
        AV_tau_s = 85.0 * exp((-pow(NV_Ith_S(y, 0) + 45.0, 2.0)) / 320.0) + 5.0 / (1.0 + exp((NV_Ith_S(y, 0) - 20.0) / 5.0)) + 3.0;
        lmbda_ptrs[11][j] = -1. / AV_tau_s;
    }
}

double TenTusscher2006_epi_smooth::rho_f_expl()
{
    return 6.5;
}

#endif