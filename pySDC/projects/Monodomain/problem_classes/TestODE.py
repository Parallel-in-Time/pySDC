import logging
import numpy as np
from pySDC.core.Problem import ptype
from pySDC.core.Common import RegisterParams
from pySDC.projects.Monodomain.datatype_classes.myfloat import myfloat, IMEXEXP_myfloat
from pySDC.core.Collocation import CollBase
import scipy


class Parabolic(RegisterParams):
    def __init__(self, **problem_params):
        self._makeAttributeAndRegister(*problem_params.keys(), localVars=problem_params, readOnly=True)


class TestODE(ptype):
    def __init__(self, **problem_params):
        self.logger = logging.getLogger("step")

        # the class for the spatial discretization of the parabolic part of monodomain
        self.parabolic = Parabolic(**problem_params)

        # invoke super init
        self.init = 1
        self.size = 1
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

        def dtype_u(init, val=0.0):
            return myfloat(init, val)

        def dtype_f(init, val=0.0):
            return myfloat(init, val)

        self.dtype_u = dtype_u
        self.dtype_f = dtype_f

    def initial_value(self):
        u0 = self.dtype_u(self.init, val=1.0)

        return u0

    def eval_f(self, u, t, fh=None):
        if fh is None:
            fh = self.dtype_f(init=self.init, val=0.0)

        # apply stimulus
        fh.values = (self.lmbda_laplacian + self.lmbda_gating + self.lmbda_others) * u.values

        return fh


class MultiscaleTestODE(TestODE):
    def __init__(self, **problem_params):
        super(MultiscaleTestODE, self).__init__(**problem_params)

        def dtype_f(init=None, val=0.0):
            return IMEXEXP_myfloat(init, val)

        self.dtype_f = dtype_f

        self.rhs_stiff_indeces = [0]
        self.rhs_stiff_args = [0]
        self.rhs_nonstiff_indeces = [0]
        self.rhs_nonstiff_args = [0]
        self.rhs_exp_args = [0]
        self.rhs_exp_indeces = [0]

        self.constant_lambda_and_phi = True

        self.num_coll_nodes = 10
        self.coll = CollBase(num_nodes=self.num_coll_nodes, tleft=0, tright=1, node_type='LEGENDRE', quad_type='GAUSS')

    def solve_system(self, rhs, factor, u0, t, u_sol=None):
        if u_sol is None:
            u_sol = self.dtype_u(init=self.init, val=0.0)

        u_sol.values = rhs.values / (1 - factor * self.lmbda_laplacian)

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
            fh.expl.values = self.lmbda_others * u.values

        # evaluate implicit (stiff) part lambda_laplacian*u
        if eval_impl:
            fh.impl.values = self.lmbda_laplacian * u.values

        # evaluate exponential part lambda_gating*u
        if eval_exp:
            fh.exp.values = self.lmbda_gating * u.values

        return fh

    def eval_phi_f_exp(self, u, factor, t, phi_f_exp=None, zero_untouched_indeces=True):
        if phi_f_exp is None:
            phi_f_exp = self.dtype_u(init=self.init, val=0.0)

        phi_f_exp.values = (np.exp(factor * self.lmbda_gating) - 1.0) / (factor) * u.values

        return phi_f_exp

    def phi_eval_lists(self, u, factors, t, indeces, phi=None, lmbda=None, update_non_exp_indeces=True):
        # compute phi[k][i] = phi_{k}(factor_i*lmbda), factor_i in factors, k in indeces

        N_fac = len(factors)
        N_ind = len(indeces)

        if phi is None:
            phi = [[self.dtype_u(init=self.init, val=0.0) for i in range(N_fac)] for j in range(N_ind)]
        else:
            for n in range(N_fac):
                for m in range(N_ind):
                    phi[m][n].zero()

        factorials = scipy.special.factorial(np.array(indeces) - 1)
        c = self.coll.nodes
        b = self.coll.weights
        for n in range(N_fac):
            factor = factors[n]
            exp_terms = [np.exp(((1.0 - c[j]) * factor) * self.lmbda_gating) for j in range(self.num_coll_nodes)]
            for m in range(N_ind):
                k = indeces[m]
                km1_fac = factorials[m]
                if k == 0:
                    phi[m][n].values = np.exp(factor * self.lmbda_gating)
                else:
                    for j in range(self.num_coll_nodes):
                        phi[m][n].values += ((b[j] * c[j] ** (k - 1)) / km1_fac) * exp_terms[j]

        return phi

    def lmbda_eval(self, u, t, lmbda=None):
        if lmbda is None:
            lmbda = self.dtype_u(init=self.init, val=0.0)

        lmbda.values = self.lmbda_gating

        return lmbda
