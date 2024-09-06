import logging
import numpy as np
from pySDC.core.problem import Problem
from pySDC.core.common import RegisterParams
from pySDC.implementations.datatype_classes.mesh import mesh
from pySDC.projects.Monodomain.datatype_classes.my_mesh import imexexp_mesh


"""
Here we define the problems classes for the multirate Dahlquist test equation y'=lambda_I*y + lambda_E*y + lambda_e*y
Things are done so that it is compatible witht the sweepers.
"""


class Parabolic(RegisterParams):
    def __init__(self, **problem_params):
        self._makeAttributeAndRegister(*problem_params.keys(), localVars=problem_params, readOnly=True)
        self.shape = (1,)
        self.init = ((1,), None, np.dtype("float64"))


class TestODE(Problem):
    def __init__(self, **problem_params):
        self.logger = logging.getLogger("step")

        self.parabolic = Parabolic(**problem_params)
        self.size = 1  # one state variable
        self.init = ((self.size, *self.parabolic.init[0]), self.parabolic.init[1], self.parabolic.init[2])

        # invoke super init
        super(TestODE, self).__init__(self.init)
        # store all problem params dictionary values as attributes
        self._makeAttributeAndRegister(*problem_params.keys(), localVars=problem_params, readOnly=True)

        # initial and end time
        self.t0 = 0.0
        self.Tend = 1.0 if self.end_time < 0.0 else self.end_time

        # set lambdas, if not provided by user
        if not hasattr(self, 'lmbda_laplacian'):
            self.lmbda_laplacian = -5.0
        if not hasattr(self, 'lmbda_gating'):
            self.lmbda_gating = -10.0
        if not hasattr(self, 'lmbda_others'):
            self.lmbda_others = -1.0

        self.dtype_u = mesh
        self.dtype_f = mesh

    def init_exp_extruded(self, new_dim_shape):
        return ((*new_dim_shape, 1, self.init[0][1]), self.init[1], self.init[2])

    def initial_value(self):
        u0 = self.dtype_u(self.init, val=1.0)

        return u0

    def eval_f(self, u, t, fh=None):
        if fh is None:
            fh = self.dtype_f(init=self.init, val=0.0)

        fh[0] = (self.lmbda_laplacian + self.lmbda_gating + self.lmbda_others) * u[0]

        return fh


class MultiscaleTestODE(TestODE):
    def __init__(self, **problem_params):
        super(MultiscaleTestODE, self).__init__(**problem_params)

        self.dtype_f = imexexp_mesh

        self.rhs_stiff_indeces = [0]
        self.rhs_stiff_args = [0]
        self.rhs_nonstiff_indeces = [0]
        self.rhs_nonstiff_args = [0]
        self.rhs_exp_args = [0]
        self.rhs_exp_indeces = [0]
        self.rhs_non_exp_indeces = []

        self.constant_lambda_and_phi = True

        self.one = self.dtype_u(init=self.init, val=1.0)

    def solve_system(self, rhs, factor, u0, t, u_sol=None):
        if u_sol is None:
            u_sol = self.dtype_u(init=self.init, val=0.0)

        u_sol[0] = rhs[0] / (1 - factor * self.lmbda_laplacian)

        return u_sol

    def eval_f(self, u, t, eval_impl=True, eval_expl=True, eval_exp=True, fh=None, zero_untouched_indeces=True):

        if fh is None:
            fh = self.dtype_f(init=self.init, val=0.0)

        if eval_expl:
            fh.expl[0] = self.lmbda_others * u[0]

        if eval_impl:
            fh.impl[0] = self.lmbda_laplacian * u[0]

        if eval_exp:
            fh.exp[0] = self.lmbda_gating * u[0]

        return fh

    def eval_lmbda_yinf_exp(self, u, lmbda, yinf):
        lmbda[0] = self.lmbda_gating
        yinf[0] = 0.0

    def lmbda_eval(self, u, t, lmbda=None):
        if lmbda is None:
            lmbda = self.dtype_u(init=self.init, val=0.0)

        lmbda[0] = self.lmbda_gating

        return lmbda
