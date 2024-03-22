#include <cmath>
#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "ionicmodel.h"

#ifndef HODGKINHUXLEY
#define HODGKINHUXLEY

class HodgkinHuxley : public IonicModel
{
public:
    HodgkinHuxley(const double scale_);
    ~HodgkinHuxley(){};
    void f(py::array_t<double> &y, py::array_t<double> &fy);
    void f_expl(py::array_t<double> &y, py::array_t<double> &fy);
    void lmbda_exp(py::array_t<double> &y_list, py::array_t<double> &lmbda_list);
    void lmbda_yinf_exp(py::array_t<double> &y_list, py::array_t<double> &lmbda_list, py::array_t<double> &yinf_list);
    py::list initial_values();
    double rho_f_expl();

private:
    double AC_g_L, AC_Cm, AC_E_R, AC_E_K, AC_g_K, AC_E_Na, AC_g_Na, AC_E_L;
};

HodgkinHuxley::HodgkinHuxley(const double scale_)
    : IonicModel(scale_)
{
    size = 4;

    // Set values of constants
    AC_g_L = 0.3;
    AC_Cm = 1.0;
    AC_E_R = -75.0;
    AC_E_K = AC_E_R - 12.0;
    AC_g_K = 36.0;
    AC_E_Na = AC_E_R + 115.0;
    AC_g_Na = 120.0;
    AC_E_L = AC_E_R + 10.613;

    assign(f_expl_args, {0, 1, 2, 3});
    assign(f_exp_args, {0, 1, 2, 3});
    assign(f_expl_indeces, {0});
    assign(f_exp_indeces, {1, 2, 3});
}

py::list HodgkinHuxley::initial_values()
{
    py::list y0(size);
    y0[0] = -75.0;
    y0[1] = 0.05;
    y0[2] = 0.595;
    y0[3] = 0.317;

    return y0;
}

void HodgkinHuxley::f(py::array_t<double> &y_list, py::array_t<double> &fy_list)
{
    double *y_ptrs[size];
    double *fy_ptrs[size];
    size_t N;
    size_t n_dofs;
    get_raw_data(y_list, y_ptrs, N, n_dofs);
    get_raw_data(fy_list, fy_ptrs, N, n_dofs);

    double AV_alpha_n, AV_beta_n, AV_alpha_h, AV_beta_h, AV_alpha_m, AV_beta_m, AV_i_K, AV_i_Na, AV_i_L;
    // Remember to scale the first variable!!!
    for (unsigned j = 0; j < n_dofs; j++)
    {
        double y[4] = {y_ptrs[0][j], y_ptrs[1][j], y_ptrs[2][j], y_ptrs[3][j]};

        AV_alpha_n = (-0.01) * (y[0] + 65.0) / (exp((-(y[0] + 65.0)) / 10.0) - 1.0);
        AV_beta_n = 0.125 * exp((y[0] + 75.0) / 80.0);
        fy_ptrs[3][j] = AV_alpha_n * (1.0 - y[3]) - AV_beta_n * y[3];

        AV_alpha_h = 0.07 * exp((-(y[0] + 75.0)) / 20.0);
        AV_beta_h = 1.0 / (exp((-(y[0] + 45.0)) / 10.0) + 1.0);
        fy_ptrs[2][j] = AV_alpha_h * (1.0 - y[2]) - AV_beta_h * y[2];

        AV_alpha_m = (-0.1) * (y[0] + 50.0) / (exp((-(y[0] + 50.0)) / 10.0) - 1.0);
        AV_beta_m = 4.0 * exp((-(y[0] + 75.0)) / 18.0);
        fy_ptrs[1][j] = AV_alpha_m * (1.0 - y[1]) - AV_beta_m * y[1];

        AV_i_K = AC_g_K * pow(y[3], 4.0) * (y[0] - AC_E_K);
        AV_i_Na = AC_g_Na * pow(y[1], 3.0) * y[2] * (y[0] - AC_E_Na);
        AV_i_L = AC_g_L * (y[0] - AC_E_L);
        fy_ptrs[0][j] = -scale * (AV_i_Na + AV_i_K + AV_i_L);
    }
}

void HodgkinHuxley::f_expl(py::array_t<double> &y_list, py::array_t<double> &fy_list)
{
    double *y_ptrs[size];
    double *fy_ptrs[size];
    size_t N;
    size_t n_dofs;
    get_raw_data(y_list, y_ptrs, N, n_dofs);
    get_raw_data(fy_list, fy_ptrs, N, n_dofs);

    double AV_i_K, AV_i_Na, AV_i_L;
    // Remember to scale the first variable!!!
    for (unsigned j = 0; j < n_dofs; j++)
    {
        AV_i_K = AC_g_K * pow(y_ptrs[3][j], 4.0) * (y_ptrs[0][j] - AC_E_K);
        AV_i_Na = AC_g_Na * pow(y_ptrs[1][j], 3.0) * y_ptrs[2][j] * (y_ptrs[0][j] - AC_E_Na);
        AV_i_L = AC_g_L * (y_ptrs[0][j] - AC_E_L);
        fy_ptrs[0][j] = -scale * (AV_i_Na + AV_i_K + AV_i_L);
    }
}

void HodgkinHuxley::lmbda_exp(py::array_t<double> &y_list, py::array_t<double> &lmbda_list)
{
    double *y_ptrs[size];
    double *lmbda_ptrs[size];
    size_t N;
    size_t n_dofs;
    get_raw_data(y_list, y_ptrs, N, n_dofs);
    get_raw_data(lmbda_list, lmbda_ptrs, N, n_dofs);

    double AV_alpha_n, AV_beta_n, AV_alpha_h, AV_beta_h, AV_alpha_m, AV_beta_m;
    for (unsigned j = 0; j < n_dofs; j++)
    {
        AV_alpha_n = (-0.01) * (y_ptrs[0][j] + 65.0) / (exp((-(y_ptrs[0][j] + 65.0)) / 10.0) - 1.0);
        AV_beta_n = 0.125 * exp((y_ptrs[0][j] + 75.0) / 80.0);
        lmbda_ptrs[3][j] = -(AV_alpha_n + AV_beta_n);

        AV_alpha_h = 0.07 * exp((-(y_ptrs[0][j] + 75.0)) / 20.0);
        AV_beta_h = 1.0 / (exp((-(y_ptrs[0][j] + 45.0)) / 10.0) + 1.0);
        lmbda_ptrs[2][j] = -(AV_alpha_h + AV_beta_h);

        AV_alpha_m = (-0.1) * (y_ptrs[0][j] + 50.0) / (exp((-(y_ptrs[0][j] + 50.0)) / 10.0) - 1.0);
        AV_beta_m = 4.0 * exp((-(y_ptrs[0][j] + 75.0)) / 18.0);
        lmbda_ptrs[1][j] = -(AV_alpha_m + AV_beta_m);
    }
}

void HodgkinHuxley::lmbda_yinf_exp(py::array_t<double> &y_list, py::array_t<double> &lmbda_list, py::array_t<double> &yinf_list)
{
    double *y_ptrs[size];
    double *lmbda_ptrs[size];
    double *yinf_ptrs[size];
    size_t N;
    size_t n_dofs;
    get_raw_data(y_list, y_ptrs, N, n_dofs);
    get_raw_data(lmbda_list, lmbda_ptrs, N, n_dofs);
    get_raw_data(yinf_list, yinf_ptrs, N, n_dofs);

    double AV_alpha_n, AV_beta_n, AV_alpha_h, AV_beta_h, AV_alpha_m, AV_beta_m;
    for (unsigned j = 0; j < n_dofs; j++)
    {
        AV_alpha_n = (-0.01) * (y_ptrs[0][j] + 65.0) / (exp((-(y_ptrs[0][j] + 65.0)) / 10.0) - 1.0);
        AV_beta_n = 0.125 * exp((y_ptrs[0][j] + 75.0) / 80.0);
        lmbda_ptrs[3][j] = -(AV_alpha_n + AV_beta_n);
        yinf_ptrs[3][j] = -AV_alpha_n / lmbda_ptrs[3][j];

        AV_alpha_h = 0.07 * exp((-(y_ptrs[0][j] + 75.0)) / 20.0);
        AV_beta_h = 1.0 / (exp((-(y_ptrs[0][j] + 45.0)) / 10.0) + 1.0);
        lmbda_ptrs[2][j] = -(AV_alpha_h + AV_beta_h);
        yinf_ptrs[2][j] = -AV_alpha_h / lmbda_ptrs[2][j];

        AV_alpha_m = (-0.1) * (y_ptrs[0][j] + 50.0) / (exp((-(y_ptrs[0][j] + 50.0)) / 10.0) - 1.0);
        AV_beta_m = 4.0 * exp((-(y_ptrs[0][j] + 75.0)) / 18.0);
        lmbda_ptrs[1][j] = -(AV_alpha_m + AV_beta_m);
        yinf_ptrs[1][j] = -AV_alpha_m / lmbda_ptrs[1][j];
    }
}

double HodgkinHuxley::rho_f_expl()
{
    return 40.;
}

#endif