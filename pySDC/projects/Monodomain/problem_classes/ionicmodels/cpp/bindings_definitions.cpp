#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include "ionicmodel.h"
#include "hodgkinhuxley.h"
#include "courtemanche.h"
#include "tentusscher.h"
#include "tentusscher_smooth.h"
#include "bistable.h"

PYBIND11_MODULE(ionicmodels, m)
{
    m.doc() = "";

    // A class to represent the ionic models.
    py::class_<IonicModel> IonicModelPy(m, "IonicModel");
    IonicModelPy.def(py::init<const double>());
    IonicModelPy.def_property_readonly("f_expl_args", &IonicModel::get_f_expl_args, py::return_value_policy::copy);
    IonicModelPy.def_property_readonly("f_exp_args", &IonicModel::get_f_exp_args, py::return_value_policy::copy);
    IonicModelPy.def_property_readonly("f_expl_indeces", &IonicModel::get_f_expl_indeces, py::return_value_policy::copy);
    IonicModelPy.def_property_readonly("f_exp_indeces", &IonicModel::get_f_exp_indeces, py::return_value_policy::copy);
    IonicModelPy.def_property_readonly("size", &IonicModel::get_size, py::return_value_policy::copy);

    // A very simple ionic model with one variable one. It is used for testing purposes. With this model the
    // monodomain equation reduces to a reaction-diffusion equation with one variable.
    py::class_<BiStable, IonicModel> BiStablePy(m, "BiStable");
    BiStablePy.def(py::init<const double>());
    BiStablePy.def("initial_values", &BiStable::initial_values, py::return_value_policy::copy);
    BiStablePy.def("f", &BiStable::f);
    BiStablePy.def("f_expl", &BiStable::f_expl);
    BiStablePy.def("lmbda_exp", &BiStable::lmbda_exp);
    BiStablePy.def("lmbda_yinf_exp", &BiStable::lmbda_yinf_exp);
    BiStablePy.def("rho_f_expl", &BiStable::rho_f_expl);

    // The Hodgkin-Huxley ionic model. A model with 4 variables, smooth, nonstiff. Still an academic model.
    py::class_<HodgkinHuxley, IonicModel> HodgkinHuxleyPy(m, "HodgkinHuxley");
    HodgkinHuxleyPy.def(py::init<const double>());
    HodgkinHuxleyPy.def("initial_values", &HodgkinHuxley::initial_values, py::return_value_policy::copy);
    HodgkinHuxleyPy.def("f", &HodgkinHuxley::f);
    HodgkinHuxleyPy.def("f_expl", &HodgkinHuxley::f_expl);
    HodgkinHuxleyPy.def("lmbda_exp", &HodgkinHuxley::lmbda_exp);
    HodgkinHuxleyPy.def("lmbda_yinf_exp", &HodgkinHuxley::lmbda_yinf_exp);
    HodgkinHuxleyPy.def("rho_f_expl", &HodgkinHuxley::rho_f_expl);

    // The Courtemanche ionic model. A model with 21 variables, mildly stiff. It is a realistic model for the human atrial cells.
    py::class_<Courtemanche1998, IonicModel> Courtemanche1998Py(m, "Courtemanche1998");
    Courtemanche1998Py.def(py::init<const double>());
    Courtemanche1998Py.def("initial_values", &Courtemanche1998::initial_values, py::return_value_policy::copy);
    Courtemanche1998Py.def("f", &Courtemanche1998::f);
    Courtemanche1998Py.def("f_expl", &Courtemanche1998::f_expl);
    Courtemanche1998Py.def("lmbda_exp", &Courtemanche1998::lmbda_exp);
    Courtemanche1998Py.def("lmbda_yinf_exp", &Courtemanche1998::lmbda_yinf_exp);
    Courtemanche1998Py.def("rho_f_expl", &Courtemanche1998::rho_f_expl);

    // The TenTusscher ionic model. A model with 20 variables, very stiff. It is a realistic model for the human ventricular cells.
    py::class_<TenTusscher2006_epi, IonicModel> TenTusscher2006_epiPy(m, "TenTusscher2006_epi");
    TenTusscher2006_epiPy.def(py::init<const double>());
    TenTusscher2006_epiPy.def("initial_values", &TenTusscher2006_epi::initial_values, py::return_value_policy::copy);
    TenTusscher2006_epiPy.def("f", &TenTusscher2006_epi::f);
    TenTusscher2006_epiPy.def("f_expl", &TenTusscher2006_epi::f_expl);
    TenTusscher2006_epiPy.def("lmbda_exp", &TenTusscher2006_epi::lmbda_exp);
    TenTusscher2006_epiPy.def("lmbda_yinf_exp", &TenTusscher2006_epi::lmbda_yinf_exp);
    TenTusscher2006_epiPy.def("rho_f_expl", &TenTusscher2006_epi::rho_f_expl);

    // A smoothed version TenTusscher ionic model. Indeed, in the right-hand side of the original model there are if-else clauses which are not differentiable.
    // This model is a smoothed version of the original model, where the if-else clauses are removed by keeping the 'else' part of the clauses.
    // The model is no more exact, from a physiological viewpoint, but the qualitative behavior is preserved. For instance it remains very stiff and
    // action potentials are still propagated. Moreover, it is now differentiable.
    // We use this model for convergence experiments only.
    py::class_<TenTusscher2006_epi_smooth, IonicModel> TenTusscher2006_epi_smoothPy(m, "TenTusscher2006_epi_smooth");
    TenTusscher2006_epi_smoothPy.def(py::init<const double>());
    TenTusscher2006_epi_smoothPy.def("initial_values", &TenTusscher2006_epi_smooth::initial_values, py::return_value_policy::copy);
    TenTusscher2006_epi_smoothPy.def("f", &TenTusscher2006_epi_smooth::f);
    TenTusscher2006_epi_smoothPy.def("f_expl", &TenTusscher2006_epi_smooth::f_expl);
    TenTusscher2006_epi_smoothPy.def("lmbda_exp", &TenTusscher2006_epi_smooth::lmbda_exp);
    TenTusscher2006_epi_smoothPy.def("lmbda_yinf_exp", &TenTusscher2006_epi_smooth::lmbda_yinf_exp);
    TenTusscher2006_epi_smoothPy.def("rho_f_expl", &TenTusscher2006_epi_smooth::rho_f_expl);
}