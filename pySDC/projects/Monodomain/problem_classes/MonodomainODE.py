from pathlib import Path
import logging
import numpy as np
from pySDC.core.Problem import ptype
from pySDC.projects.Monodomain.datatype_classes.VectorOfVectors import VectorOfVectors, IMEXEXP_VectorOfVectors
import pySDC.projects.Monodomain.problem_classes.ionicmodels.cpp as ionicmodels
from pySDC.core.Collocation import CollBase
import scipy


class MonodomainODE(ptype):
    def __init__(self, **problem_params):
        self.logger = logging.getLogger("step")

        # the class for the spatial discretization of the parabolic part of monodomain
        parabolic_class = problem_params['parabolic_class']
        del problem_params['parabolic_class']
        self.parabolic = parabolic_class(**problem_params)
        self.init = self.parabolic.init

        # invoke super init
        super(MonodomainODE, self).__init__(self.init)
        # store all problem params dictionary values as attributes
        self._makeAttributeAndRegister(*problem_params.keys(), localVars=problem_params, readOnly=True)

        self.define_ionic_model()

        # initial and end time
        self.t0 = 0.0
        self.Tend = 50.0 if self.end_time < 0.0 else self.end_time

        # dtype_u and dtype_f are super vectors of vector_type
        self.vector_type = self.parabolic.vector_type

        def dtype_u(init=None, val=0.0):
            return VectorOfVectors(init, val, self.vector_type, self.size)

        def dtype_f(init=None, val=0.0):
            return VectorOfVectors(init, val, self.vector_type, self.size)

        self.dtype_u = dtype_u
        self.dtype_f = dtype_f

        # init output stuff
        self.output_folder = Path(self.output_root) / Path(self.parabolic.domain_name) / Path(self.parabolic.mesh_name) / Path(self.ionic_model_name)
        self.parabolic.init_output(self.output_folder)

    def write_solution(self, uh, t):
        # write solution to file, only the potential V=uh[0], not the ionic model variables
        self.parabolic.write_solution(uh[0], t)

    def write_reference_solution(self, uh, all=False):
        # write solution to file, only the potential V=uh[0] or all variables if all=True
        self.parabolic.write_reference_solution(uh, list(range(uh.size)) if all else [0])

    def read_reference_solution(self, uh, ref_file_name, all=False):
        # read solution from file, only the potential V=uh[0] or all variables if all=True
        # returns true if read was successful, false else
        return self.parabolic.read_reference_solution(uh, list(range(uh.size)) if all else [0], ref_file_name)

    def initial_value(self):
        # create initial value (as vector of vectors). Every variable is constant in space
        u0 = self.dtype_u(self.init, val=self.ionic_model.initial_values())

        # overwwrite the initial value with solution form file if desired
        if self.read_init_val:
            read_ok = self.read_reference_solution(u0, self.init_val_name, True)
            assert read_ok, "ERROR: Could not read initial value from file"

        return u0

    def compute_errors(self, uh):
        """
        Compute L2 error of uh[0] (potential V)
        Args:
            uh (VectorOfVectors): solution as vector of vectors

        Returns:
            computed (bool): if error computation was successful
            error (float): L2 error
            rel_error (float): relative L2 error
        """
        ref_sol_V = self.vector_type(init=self.init, val=0.0)
        read_ok = self.read_reference_solution([ref_sol_V], self.ref_sol, False)
        if read_ok:
            error_L2, rel_error_L2 = self.parabolic.compute_errors(uh[0], ref_sol_V)

            if self.comm.rank == 0:
                print(f"L2-errors: {error_L2}")
                print(f"Relative L2-errors: {rel_error_L2}")

            return True, error_L2, rel_error_L2
        else:
            return False, 0.0, 0.0

    def getSize(self):
        # return number of dofs in the mesh
        return self.parabolic.getSize()

    def eval_on_points(self, u):
        # evaluate the solution on a set of points (points are already defined in self.parabolic)
        return self.parabolic.eval_on_points(u)

    def define_ionic_model(self):
        self.scale_Iion = 0.01  # used to convert currents in uA/cm^2 to uA/mm^2
        # scale_im is applied to the rhs of the ionic model, so that the rhs is in units of mV/ms
        self.scale_im = self.scale_Iion / self.parabolic.Cm

        if self.ionic_model_name in ["HodgkinHuxley", "HH"]:
            self.ionic_model = ionicmodels.HodgkinHuxley(self.scale_im)
        elif self.ionic_model_name in ["Courtemanche1998", "CRN"]:
            self.ionic_model = ionicmodels.Courtemanche1998(self.scale_im)
        elif self.ionic_model_name in ["TenTusscher2006_epi", "TTP"]:
            self.ionic_model = ionicmodels.TenTusscher2006_epi(self.scale_im)
        else:
            raise Exception("Unknown ionic model.")

        self.size = self.ionic_model.size

    def eval_f(self, u, t, fh=None):
        if fh is None:
            fh = self.dtype_f(init=self.init, val=0.0)

        # self.update_u_and_uh(u, False)

        # eval ionic model rhs on u and put result in fh. All indices of the super vector fh mush be computed (list(range(self.size))
        self.eval_expr(self.ionic_model.f, u, fh, list(range(self.size)), False)
        # apply stimulus
        fh.val_list[0] += self.parabolic.Istim(t)  # Istim is defined in parabolic because it is a function of space too

        # eval diffusion
        self.parabolic.add_disc_laplacian(u[0], fh[0])

        return fh

    def eval_expr(self, expr, u, fh, indeces, zero_untouched_indeces=True):
        if expr is not None:
            expr(u.np_list, fh.np_list)

        if zero_untouched_indeces:
            non_indeces = [i for i in range(self.size) if i not in indeces]
            for i in non_indeces:
                fh[i].zero()


class MultiscaleMonodomainODE(MonodomainODE):
    def __init__(self, **problem_params):
        super(MultiscaleMonodomainODE, self).__init__(**problem_params)

        def dtype_f(init=None, val=0.0):
            return IMEXEXP_VectorOfVectors(init, val, self.vector_type, self.size)

        self.dtype_f = dtype_f

        self.define_splittings()
        self.parabolic.define_solver()

    def rho_nonstiff(self, y, t, fy=None):
        return self.rho_nonstiff_cte

    def define_splittings(self):
        # Here we define different splittings of the rhs into stiff, nonstiff and exponential terms

        if self.splitting == "stiff_nonstiff":
            # SPLITTING stiff_nonstiff
            # this is a splitting to be used in multirate explicit stabilized methods. W euse it for the mES schemes.
            # define nonstiff
            self.im_f_nonstiff = self.ionic_model.f_nonstiff
            self.im_nonstiff_args = self.ionic_model.f_nonstiff_args
            self.im_nonstiff_indeces = self.ionic_model.f_nonstiff_indeces
            # define stiff
            self.im_f_stiff = self.ionic_model.f_stiff
            self.im_stiff_args = self.ionic_model.f_stiff_args
            self.im_stiff_indeces = self.ionic_model.f_stiff_indeces
            # define exp
            self.im_lmbda_exp = None
            self.im_lmbda_yinf_exp = None
            self.im_exp_args = []
            self.im_exp_indeces = []

            self.rho_nonstiff_cte = self.ionic_model.rho_f_nonstiff()

        elif self.splitting == "exp_nonstiff":
            # SPLITTING exp_nonstiff
            # this is the standard splitting used in Rush-Larsen methods. We use it for the IMEXEXP (IMEX+RL) and exp_mES schemes.
            # define nonstiff.
            self.im_f_nonstiff = self.ionic_model.f_expl
            self.im_nonstiff_args = self.ionic_model.f_expl_args
            self.im_nonstiff_indeces = self.ionic_model.f_expl_indeces
            # define stiff
            self.im_f_stiff = None  # no stiff part coming from ionic model
            self.im_stiff_args = []
            self.im_stiff_indeces = []
            # define exp
            self.im_lmbda_exp = self.ionic_model.lmbda_exp
            self.im_lmbda_yinf_exp = self.ionic_model.lmbda_yinf_exp
            self.im_exp_args = self.ionic_model.f_exp_args
            self.im_exp_indeces = self.ionic_model.f_exp_indeces

            self.rho_nonstiff_cte = self.ionic_model.rho_f_expl()

        else:
            raise Exception("Unknown splitting.")

        self.one = self.dtype_u(init=self.init, val=1.0)

        self.im_non_exp_indeces = [i for i in range(self.size) if i not in self.im_exp_indeces]
        self.rhs_stiff_args = self.im_stiff_args
        self.rhs_stiff_indeces = self.im_stiff_indeces
        if 0 not in self.rhs_stiff_args:
            self.rhs_stiff_args = [0] + self.rhs_stiff_args
        self.rhs_nonstiff_args = self.im_nonstiff_args
        if 0 not in self.rhs_nonstiff_args:
            self.rhs_nonstiff_args = [0] + self.rhs_nonstiff_args
        self.rhs_exp_args = self.im_exp_args
        self.rhs_exp_indeces = self.im_exp_indeces

        self.lmbda = self.dtype_u(init=self.init, val=0.0)
        self.yinf = self.dtype_u(init=self.init, val=0.0)

    def eval_lmbda_yinf_exp(self, u, lmbda, yinf):
        self.im_lmbda_yinf_exp(u.np_list, lmbda.np_list, yinf.np_list)

    def eval_lmbda_exp(self, u, lmbda):
        self.im_lmbda_exp(u.np_list, lmbda.np_list)

    def solve_system(self, rhs, factor, u0, t, u_sol=None):
        if u_sol is None:
            u_sol = self.dtype_u(init=self.init, val=0.0)

        self.parabolic.solve_system(rhs[0], factor, u0[0], t, u_sol[0])

        if rhs is not u_sol:
            for i in range(1, self.size):
                u_sol[i].copy(rhs[i])

        return u_sol

    def eval_f(self, u, t, eval_impl=True, eval_expl=True, eval_exp=True, fh=None, zero_untouched_indeces=True):
        """
        Evaluates F(u,t) = M^-1*( A*u + f(u,t) )

        Returns:
            dtype_u: solution as mesh
        """

        if fh is None:
            fh = self.dtype_f(init=self.init, val=0.0)

        # evaluate explicit (non stiff) part M^-1*f_nonstiff(u,t)
        if eval_expl:
            fh.expl = self.eval_f_nonstiff(u, t, fh.expl, zero_untouched_indeces)

        # evaluate implicit (stiff) part M^1*A*u+M^-1*f_stiff(u,t)
        if eval_impl:
            fh.impl = self.eval_f_stiff(u, t, fh.impl, zero_untouched_indeces)

        # evaluate exponential part
        if eval_exp:
            fh.exp = self.eval_f_exp(u, t, fh.exp, zero_untouched_indeces)

        return fh

    def eval_f_nonstiff(self, u, t, fh_nonstiff, zero_untouched_indeces=True):
        # eval ionic model nonstiff terms
        self.eval_expr(self.im_f_nonstiff, u, fh_nonstiff, self.im_nonstiff_indeces, zero_untouched_indeces)

        if not zero_untouched_indeces and 0 not in self.im_nonstiff_indeces:
            fh_nonstiff[0].zero()

        # apply stimulus
        fh_nonstiff.val_list[0] += self.parabolic.Istim(t)

        return fh_nonstiff

    def eval_f_stiff(self, u, t, fh_stiff, zero_untouched_indeces=True):
        # eval ionic model stiff terms
        self.eval_expr(self.im_f_stiff, u, fh_stiff, self.im_stiff_indeces, zero_untouched_indeces)

        if not zero_untouched_indeces and 0 not in self.im_stiff_indeces:
            fh_stiff[0].zero()

        # eval diffusion
        self.parabolic.add_disc_laplacian(u[0], fh_stiff[0])

        return fh_stiff

    def eval_f_exp(self, u, t, fh_exp, zero_untouched_indeces=True):
        self.eval_lmbda_yinf_exp(u, self.lmbda, self.yinf)
        for i in self.im_exp_indeces:
            fh_exp.np_list[i][:] = self.lmbda.np_list[i] * (u.np_list[i] - self.yinf.np_list[i])

        if zero_untouched_indeces:
            fh_exp.zero_sub(self.im_non_exp_indeces)

        return fh_exp

    def eval_phi_f_exp(self, u, factor, t, phi_f_exp=None, zero_untouched_indeces=True):
        if phi_f_exp is None:
            phi_f_exp = self.dtype_u(init=self.init, val=0.0)

        self.eval_lmbda_yinf_exp(u, self.lmbda, self.yinf)
        for i in self.im_exp_indeces:
            phi_f_exp.np_list[i][:] = (np.exp(factor * self.lmbda.np_list[i]) - 1.0) / (factor) * (u.np_list[i] - self.yinf.np_list[i])

        if zero_untouched_indeces:
            phi_f_exp.zero_sub(self.im_non_exp_indeces)

        return phi_f_exp

    def phi_eval(self, u, factor, t, k, phi=None):
        if phi is None:
            phi = self.dtype_u(init=self.init, val=0.0)

        self.eval_lmbda_exp(u, self.lmbda)
        self.lmbda *= factor

        if k == 0:
            for i in self.im_exp_indeces:  # phi_0
                phi.np_list[i][:] = np.exp(self.lmbda.np_list[i])
        else:
            num_nodes = 10
            self.coll = CollBase(num_nodes=num_nodes, tleft=0, tright=1, node_type='LEGENDRE', quad_type='GAUSS')
            c = self.coll.nodes
            b = self.coll.weights

            km1_fac = scipy.special.factorial(k - 1)
            for i in self.im_exp_indeces:
                phi.np_list[i][:] = (b[0] * c[0] ** (k - 1)) * np.exp((1.0 - c[0]) * self.lmbda.np_list[i])
                for j in range(1, num_nodes):
                    phi.np_list[i][:] += (b[j] * c[j] ** (k - 1)) * np.exp((1.0 - c[j]) * self.lmbda.np_list[i])
                phi.np_list[i][:] /= km1_fac

        phi.copy_sub(self.one, self.im_non_exp_indeces)
        if k > 1:
            k_fac = km1_fac * k
            phi.imul_sub(1.0 / k_fac, self.im_non_exp_indeces)

        return phi

    def lmbda_eval(self, u, t, lmbda=None):
        if lmbda is None:
            lmbda = self.dtype_u(init=self.init, val=0.0)

        self.eval_lmbda_exp(u, lmbda)

        lmbda.zero_sub(self.im_non_exp_indeces)

        return lmbda
