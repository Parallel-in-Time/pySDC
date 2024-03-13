from pathlib import Path
import logging
import numpy as np
from pySDC.core.Problem import ptype
from pySDC.projects.Monodomain.datatype_classes.VectorOfVectors import VectorOfVectors, IMEXEXP_VectorOfVectors
from pySDC.projects.Monodomain.problem_classes.space_discretizazions.Parabolic_DCT import Parabolic_DCT
import pySDC.projects.Monodomain.problem_classes.ionicmodels.cpp as ionicmodels
from pySDC.core.Collocation import CollBase


class MonodomainODE(ptype):
    def __init__(self, **problem_params):
        self.logger = logging.getLogger("step")

        self.parabolic = Parabolic_DCT(**problem_params)

        self.init = self.parabolic.init

        # invoke super init
        super(MonodomainODE, self).__init__(self.init)
        # store all problem params dictionary values as attributes
        self._makeAttributeAndRegister(*problem_params.keys(), localVars=problem_params, readOnly=True)

        self.define_ionic_model()
        self.define_stimulus()

        # initial and end time
        self.t0 = 0.0
        self.Tend = 50.0 if self.end_time < 0.0 else self.end_time

        # dtype_u and dtype_f are super vectors of vector_type
        self.vector_type = self.parabolic.vector_type

        def dtype_u(init, val=None):
            return VectorOfVectors(init, val, self.vector_type, self.size)

        def dtype_f(init, val=None):
            return VectorOfVectors(init, val, self.vector_type, self.size)

        self.dtype_u = dtype_u
        self.dtype_f = dtype_f

        # init output stuff
        self.output_folder = (
            Path(self.output_root)
            / Path(self.parabolic.domain_name)
            / Path(self.parabolic.mesh_name)
            / Path(self.ionic_model_name)
        )
        self.parabolic.init_output(self.output_folder)

    def write_solution(self, uh, t):
        # write solution to file, only the potential V=uh[0], not the ionic model variables
        self.parabolic.write_solution(uh, t, not self.output_V_only)

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
            assert read_ok, "ERROR: Could not read initial value from file."

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
        elif self.ionic_model_name in ["TTP_S", "TTP_SMOOTH"]:
            self.ionic_model = ionicmodels.TenTusscher2006_epi_smooth(self.scale_im)
        elif self.ionic_model_name in ["BiStable", "BS"]:
            self.ionic_model = ionicmodels.BiStable(self.scale_im)
        else:
            raise Exception("Unknown ionic model.")

        self.size = self.ionic_model.size

    def define_stimulus(self):
        self.scale_Iion = 0.01  # used to convert currents in uA/cm^2 to uA/mm^2
        # scale_im is applied to the rhs of the ionic model, so that the rhs is in units of mV/ms
        self.scale_im = self.scale_Iion / self.parabolic.Cm

        stim_dur = 2.0
        if "cuboid" in self.parabolic.domain_name:
            self.stim_protocol = [[0.0, stim_dur]]  # list of stim_time, sitm_dur values
            self.stim_intensities = [50.0]
            self.stim_centers = [[0.0, 0.0, 0.0]]
            r = 1.5
            self.stim_radii = [[r, r, r]] * len(self.stim_protocol)
        elif "cube" in self.parabolic.domain_name:
            self.stim_protocol = [[0.0, 2.0], [1000.0, 10.0]]
            self.stim_intensities = [50.0, 80.0]
            centers = [[0.0, 50.0, 50.0], [58.5, 0.0, 50.0]]
            self.stim_centers = [centers[i] for i in range(len(self.stim_protocol))]
            self.stim_radii = [[1.0, 50.0, 50.0], [1.5, 60.0, 50.0]]
        else:
            raise Exception("Unknown domain name.")

        self.stim_protocol = np.array(self.stim_protocol)
        self.stim_protocol[:, 0] -= self.init_time

        self.last_stim_index = -1

    def eval_f(self, u, t, fh=None):
        if fh is None:
            fh = self.dtype_f(init=self.init, val=0.0)

        # self.update_u_and_uh(u, False)

        # eval ionic model rhs on u and put result in fh. All indices of the super vector fh mush be computed (list(range(self.size))
        self.eval_expr(self.ionic_model.f, u, fh, list(range(self.size)), False)
        # apply stimulus
        fh.val_list[0] += self.Istim(t)

        # eval diffusion
        self.parabolic.add_disc_laplacian(u[0], fh[0])

        return fh

    def Istim(self, t):
        tol = 1e-8
        for i, (stim_time, stim_dur) in enumerate(self.stim_protocol):
            if (t + stim_dur * tol >= stim_time) and (t + stim_dur * tol < stim_time + stim_dur):
                if i != self.last_stim_index:
                    self.last_stim_index = i
                    self.space_stim = self.parabolic.stim_region(self.stim_centers[i], self.stim_radii[i])
                    self.space_stim *= self.scale_im * self.stim_intensities[i]
                return self.space_stim

        return self.parabolic.zero_stim_vec

    def eval_expr(self, expr, u, fh, indeces, zero_untouched_indeces=True):
        if expr is not None:
            expr(u.np_list, fh.np_list)

        if zero_untouched_indeces:
            non_indeces = [i for i in range(self.size) if i not in indeces]
            for i in non_indeces:
                fh[i].zero()

    def apply_mass_matrix(self, x, y=None):
        # computes y = M x on parabolic part and not on ionic model part
        if y is None:
            y = x.copy()
        else:
            y.copy(x)

        if self.mass_rhs == "one":
            self.parabolic.apply_mass_matrix(x.val_list[0], y.val_list[0])
        elif self.mass_rhs == "all":
            for i in range(self.size):
                self.parabolic.apply_mass_matrix(x.val_list[i], y.val_list[i])

        return y


class MultiscaleMonodomainODE(MonodomainODE):
    def __init__(self, **problem_params):
        super(MultiscaleMonodomainODE, self).__init__(**problem_params)

        def dtype_f(init, val=None):
            return IMEXEXP_VectorOfVectors(init, val, self.vector_type, self.size)

        self.dtype_f = dtype_f

        self.define_splittings()
        self.parabolic.define_solver()

        self.constant_lambda_and_phi = False

    def rho_nonstiff(self, y, t, fy=None):
        return self.rho_nonstiff_cte

    def define_splittings(self):
        """
        This function defines the splittings used in the problem.
        The self.im_* variables are meant for internal use, the self.rhs_* for external use (i.e. in the sweeper).
        The *_args and *_indeces are list of integers.
        The args are list of variables that are needed to evaluate a function plus the variables that are modified by the function
        The indeces are the list of variables that are modified by the function (subset of args).
        Example: for f(x_0,x_1,x_2,x_3,x_4)=f(x_0,x_2,x_4)=(y_0,y_1,0,0,y_4) we have
        f_args=[0,1,2,4]=([0,2,4] union [0,1,4]) since f needs x_0,x_2,x_4 and y_0,y_1,y_4 are effective outputs of the function (others are zero).
        f_indeces=[0,1,4] since only y_0,y_1,y_4 are outputs of the function, y_2,y_3 are zero

        The ionic model has many variables (say M) and each variable has the same number of dofs as the mesh (say N).
        Therefore the problem has size N*M and quickly becomes very large. Thanks to args and indeces we can:
        - avoid to copy the whole vector M*N of variables when we only need a subset, for instance 2*N
        - avoid unnecessary operations on the whole vector, for instance update only the variables that are effective outputs of a function (indeces),
        and so on.

        Yeah, it's a bit a mess, but helpful.
        """
        # this is the standard splitting used in Rush-Larsen methods.
        # define nonstiff term (explicit part)
        self.im_f_nonstiff = self.ionic_model.f_expl
        self.im_nonstiff_args = self.ionic_model.f_expl_args
        self.im_nonstiff_indeces = self.ionic_model.f_expl_indeces
        # define stiff term (implicit part)
        self.im_f_stiff = None  # no stiff part coming from ionic model (everything stiff is in the exponential part)
        self.im_stiff_args = []
        self.im_stiff_indeces = []
        # define exp term (eponential part)
        self.im_lmbda_exp = self.ionic_model.lmbda_exp
        self.im_lmbda_yinf_exp = self.ionic_model.lmbda_yinf_exp
        self.im_exp_args = self.ionic_model.f_exp_args
        self.im_exp_indeces = self.ionic_model.f_exp_indeces

        self.rho_nonstiff_cte = self.ionic_model.rho_f_expl()

        self.rhs_stiff_args = self.im_stiff_args
        self.rhs_stiff_indeces = self.im_stiff_indeces
        if 0 not in self.rhs_stiff_args:
            self.rhs_stiff_args = [0] + self.rhs_stiff_args
        if 0 not in self.rhs_stiff_indeces:
            self.rhs_stiff_indeces = [0] + self.rhs_stiff_indeces

        self.rhs_nonstiff_args = self.im_nonstiff_args
        self.rhs_nonstiff_indeces = self.im_nonstiff_indeces
        if 0 not in self.rhs_nonstiff_indeces:
            self.rhs_nonstiff_indeces = [0] + self.rhs_nonstiff_indeces

        self.im_non_exp_indeces = [i for i in range(self.size) if i not in self.im_exp_indeces]

        self.rhs_exp_args = self.im_exp_args
        self.rhs_exp_indeces = self.im_exp_indeces

        self.rhs_non_exp_indeces = self.im_non_exp_indeces

        self.one = self.dtype_u(init=self.init, val=1.0)

        self.lmbda = self.dtype_u(init=self.init, val=0.0)
        self.yinf = self.dtype_u(init=self.init, val=0.0)

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

        u.ghostUpdate(addv="insert", mode="forward")

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
        fh_nonstiff.val_list[0] += self.Istim(t)

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
            fh_exp.np_array(i)[:] = self.lmbda.np_array(i) * (u.np_array(i) - self.yinf.np_array(i))

        if zero_untouched_indeces:
            fh_exp.zero_sub(self.im_non_exp_indeces)

        return fh_exp

    def lmbda_eval(self, u, t, lmbda=None):
        if lmbda is None:
            lmbda = self.dtype_u(init=self.init, val=0.0)

        self.eval_lmbda_exp(u, lmbda)

        lmbda.zero_sub(self.im_non_exp_indeces)

        return lmbda

    def eval_lmbda_yinf_exp(self, u, lmbda, yinf):
        self.im_lmbda_yinf_exp(u.np_list, lmbda.np_list, yinf.np_list)

    def eval_lmbda_exp(self, u, lmbda):
        self.im_lmbda_exp(u.np_list, lmbda.np_list)
