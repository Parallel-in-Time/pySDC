import logging
from pySDC.core.Problem import ptype
from pySDC.core.Common import RegisterParams
from pySDC.projects.Monodomain.datatype_classes.DCT_Vector import DCT_Vector
from pySDC.projects.Monodomain.datatype_classes.VectorOfVectors import VectorOfVectors, IMEXEXP_VectorOfVectors


class Parabolic(RegisterParams):
    def __init__(self, **problem_params):
        self._makeAttributeAndRegister(*problem_params.keys(), localVars=problem_params, readOnly=True)
        self.vector_type = DCT_Vector
        self.shape = (1,)


class TestODE(ptype):
    def __init__(self, **problem_params):
        self.logger = logging.getLogger("step")

        # the class for the spatial discretization of the parabolic part of monodomain
        self.parabolic = Parabolic(**problem_params)
        self.init = 1  # one dof
        self.size = 1  # one state variable

        # invoke super init
        super(TestODE, self).__init__(self.init)
        # store all problem params dictionary values as attributes
        self._makeAttributeAndRegister(*problem_params.keys(), localVars=problem_params, readOnly=True)

        # initial and end time
        self.t0 = 0.0
        self.Tend = 1.0 if self.end_time < 0.0 else self.end_time

        if not hasattr(self, 'lmbda_laplacian'):
            self.lmbda_laplacian = -5.0
        if not hasattr(self, 'lmbda_gating'):
            self.lmbda_gating = -10.0
        if not hasattr(self, 'lmbda_others'):
            self.lmbda_others = -1.0

        # dtype_u and dtype_f are super vectors of vector_type
        self.vector_type = self.parabolic.vector_type

        def dtype_u(init=None, val=0.0):
            return VectorOfVectors(init, val, self.vector_type, self.size)

        def dtype_f(init=None, val=0.0):
            return VectorOfVectors(init, val, self.vector_type, self.size)

        self.dtype_u = dtype_u
        self.dtype_f = dtype_f

    def initial_value(self):
        u0 = self.dtype_u(self.init, val=1.0)

        return u0

    def eval_f(self, u, t, fh=None):
        if fh is None:
            fh = self.dtype_f(init=self.init, val=0.0)

        fh.val_list[0].values[0] = (self.lmbda_laplacian + self.lmbda_gating + self.lmbda_others) * u[0].values[0]

        return fh


class MultiscaleTestODE(TestODE):
    def __init__(self, **problem_params):
        super(MultiscaleTestODE, self).__init__(**problem_params)

        def dtype_f(init=None, val=0.0):
            return IMEXEXP_VectorOfVectors(init, val, self.vector_type, self.size)

        self.dtype_f = dtype_f

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

        u_sol[0].values[0] = rhs[0].values[0] / (1 - factor * self.lmbda_laplacian)

        return u_sol

    def eval_f(self, u, t, eval_impl=True, eval_expl=True, eval_exp=True, fh=None, zero_untouched_indeces=True):
        """
        Evaluates F(u,t) = M^-1*( A*u + f(u,t) )

        Returns:
            dtype_u: solution as mesh
        """

        if fh is None:
            fh = self.dtype_f(init=self.init, val=0.0)

        # evaluate explicit (non stiff) part lambda_others*u
        if eval_expl:
            fh.expl.val_list[0].values[0] = self.lmbda_others * u[0].values[0]

        # evaluate implicit (stiff) part lambda_laplacian*u
        if eval_impl:
            fh.impl.val_list[0].values[0] = self.lmbda_laplacian * u[0].values[0]

        # evaluate exponential part lambda_gating*u
        if eval_exp:
            fh.exp.val_list[0].values[0] = self.lmbda_gating * u[0].values[0]

        return fh

    # def eval_phi_f_exp(self, u, factor, t, phi_f_exp=None, zero_untouched_indeces=True):
    #     if phi_f_exp is None:
    #         phi_f_exp = self.dtype_u(init=self.init, val=0.0)

    #     phi_f_exp.val_list[0].values[0] = (
    #         (np.exp(factor * self.lmbda_gating) - 1.0) / (factor) * u.val_list[0].values[0]
    #     )

    #     return phi_f_exp

    def eval_lmbda_yinf_exp(self, u, lmbda, yinf):
        lmbda.val_list[0].values[0] = self.lmbda_gating
        yinf.val_list[0].values[0] = 0.0

    def lmbda_eval(self, u, t, lmbda=None):
        if lmbda is None:
            lmbda = self.dtype_u(init=self.init, val=0.0)

        lmbda.val_list[0].values[0] = self.lmbda_gating

        return lmbda
