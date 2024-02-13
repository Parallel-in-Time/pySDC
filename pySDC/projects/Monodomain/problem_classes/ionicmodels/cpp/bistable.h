#include <cmath>
#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "ionicmodel.h"

#ifndef BISTABLE
#define BISTABLE

class BiStable : public IonicModel
{
public:
    BiStable(const double scale_);
    ~BiStable(){};
    void f(py::list y, py::list fy);
    void f_expl(py::list y, py::list fy);
    void lmbda_exp(py::list y_list, py::list lmbda_list);
    void lmbda_yinf_exp(py::list y_list, py::list lmbda_list, py::list yinf_list);
    void f_nonstiff(py::list y, py::list fy);
    void f_stiff(py::list y, py::list fy);
    py::list initial_values();
    double rho_f_expl();
    double rho_f_nonstiff();

private:
    double V_th, V_depol, V_rest, a;
};

BiStable::BiStable(const double scale_)
    : IonicModel(scale_)
{
    size = 1;

    // Set values of constants
    V_th = -57.6;
    V_depol = 30.;
    V_rest = -85.;
    a = 1.4e-3;

    assign(f_nonstiff_args, {0});
    assign(f_stiff_args, {});
    assign(f_nonstiff_indeces, {0});
    assign(f_stiff_indeces, {});
    assign(f_expl_args, {0});
    assign(f_exp_args, {});
    assign(f_expl_indeces, {0});
    assign(f_exp_indeces, {});
}

py::list BiStable::initial_values()
{
    py::list y0(size);
    y0[0] = -85.0;

    return y0;
}

void BiStable::f(py::list y_list, py::list fy_list)
{
    double *y_ptrs[size];
    double *fy_ptrs[size];
    size_t N;
    size_t n_dofs;
    get_raw_data(y_list, y_ptrs, N, n_dofs);
    get_raw_data(fy_list, fy_ptrs, N, n_dofs);

    // Remember to scale the first variable!!!
    for (unsigned j = 0; j < n_dofs; j++)
        fy_ptrs[0][j] = -scale * a * (y_ptrs[0][j] - V_th) * (y_ptrs[0][j] - V_depol) * (y_ptrs[0][j] - V_rest);
}

void BiStable::f_expl(py::list y_list, py::list fy_list)
{
    this->f(y_list, fy_list);
}

void BiStable::lmbda_exp(py::list y_list, py::list lmbda_list)
{
    return;
}

void BiStable::lmbda_yinf_exp(py::list y_list, py::list lmbda_list, py::list yinf_list)
{
    return;
}

void BiStable::f_nonstiff(py::list y_list, py::list fy_list)
{
    this->f(y_list, fy_list);
}

void BiStable::f_stiff(py::list y_list, py::list fy_list)
{
    return;
}

double BiStable::rho_f_expl()
{
    return 20.;
}

double BiStable::rho_f_nonstiff()
{
    return 20.;
}

#endif