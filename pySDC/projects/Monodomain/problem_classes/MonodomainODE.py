from pathlib import Path
import logging
import numpy as np
from pySDC.core.problem import Problem
from pySDC.implementations.datatype_classes.mesh import mesh
from pySDC.projects.Monodomain.datatype_classes.my_mesh import imexexp_mesh
from pySDC.projects.Monodomain.problem_classes.space_discretizazions.Parabolic_DCT import Parabolic_DCT
import pySDC.projects.Monodomain.problem_classes.ionicmodels.cpp as ionicmodels


class MonodomainODE(Problem):
    """
    A class for the discretization of the Monodomain equation. The Monodomain equation is a parabolic PDE composed of
    a reaction-diffusion equation coupled with an ODE system. The unknowns are the potential V and the ionic model variables (g_1,...,g_N).
    The reaction-diffusion equation is discretized in another class, where any spatial discretization can be used.
    The ODE system is the ionic model, which doesn't need spatial discretization, being a system of ODEs.



    Attributes:
    -----------
    parabolic: The parabolic problem class used to discretize the reaction-diffusion equation
    ionic_model: The ionic model used to discretize the ODE system. This is a wrapper around the actual ionic model, which is written in C++.
    size: The number of variables in the ionic model
    vector_type: The type of vector used to store a single unknown (e.g. V). This data type depends on spatial discretization, hence on the parabolic class.
    dtype_u: The type of vector used to store all the unknowns (V,g_1,...,g_N). This is a vector of vector_type.
    dtype_f: The type of vector used to store the right-hand side of the ODE system stemming from the monodomain equation. This is a vector of vector_type.
    output_folder: The folder where the solution is written to file
    t0: The initial simulation time. This is 0.0 by default but can be changed by the user in order to skip the initial stimulus.
    Tend: The duration of the simulation.
    output_V_only: If True, only the potential V is written to file. If False, all the ionic model variables are written to file.
    read_init_val: If True, the initial value is read from file. If False, the initial value is at equilibrium.
    """

    dtype_u = mesh
    dtype_f = mesh

    def __init__(self, **problem_params):
        self.logger = logging.getLogger("step")

        self.parabolic = Parabolic_DCT(**problem_params)

        self.define_ionic_model(problem_params["ionic_model_name"])

        self.init = ((self.size, *self.parabolic.init[0]), self.parabolic.init[1], self.parabolic.init[2])

        # invoke super init
        super(MonodomainODE, self).__init__(self.init)
        # store all problem params dictionary values as attributes
        self._makeAttributeAndRegister(*problem_params.keys(), localVars=problem_params, readOnly=True)

        self.define_stimulus()

        # initial and end time
        self.t0 = 0.0
        self.Tend = 50.0 if self.end_time < 0.0 else self.end_time

        # init output stuff
        self.output_folder = (
            Path(self.output_root)
            / Path(self.parabolic.domain_name)
            / Path(self.parabolic.mesh_name)
            / Path(self.ionic_model_name)
        )
        self.parabolic.init_output(self.output_folder)

    def init_exp_extruded(self, new_dim_shape):
        # The info needed to initialize a new vector of size (M,N) where M is the number of variables in the
        # ionic model with exponential terms and N is the number of dofs in the mesh.
        # The vector is further extruded to additional dimensions with shape new_dim_shape.
        return ((*new_dim_shape, len(self.rhs_exp_indeces), self.init[0][1]), self.init[1], self.init[2])

    def write_solution(self, uh, t):
        # write solution to file, only the potential V=uh[0], not the ionic model variables
        self.parabolic.write_solution(uh[0], t)

    def write_reference_solution(self, uh, all=False):
        # write solution to file, only the potential V=uh[0] or all variables if all=True
        self.parabolic.write_reference_solution(uh, list(range(self.size)) if all else [0])

    def read_reference_solution(self, uh, ref_file_name, all=False):
        # read solution from file, only the potential V=uh[0] or all variables if all=True
        # returns true if read was successful, false else
        return self.parabolic.read_reference_solution(uh, list(range(self.size)) if all else [0], ref_file_name)

    def initial_value(self):
        # Create initial value. Every variable is constant in space
        u0 = self.dtype_u(self.init)
        init_vals = self.ionic_model.initial_values()
        for i in range(self.size):
            u0[i][:] = init_vals[i]

        # overwrite the initial value with solution from file if desired
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
        ref_sol_V = self.dtype_u(init=self.init, val=0.0)
        read_ok = self.read_reference_solution(ref_sol_V, self.ref_sol, False)
        if read_ok:
            error_L2, rel_error_L2 = self.parabolic.compute_errors(uh[0], ref_sol_V[0])

            print(f"L2-errors: {error_L2}")
            print(f"Relative L2-errors: {rel_error_L2}")

            return True, error_L2, rel_error_L2
        else:
            return False, 0.0, 0.0

    def define_ionic_model(self, ionic_model_name):
        self.scale_Iion = 0.01  # used to convert currents in uA/cm^2 to uA/mm^2
        # scale_im is applied to the rhs of the ionic model, so that the rhs is in units of mV/ms
        self.scale_im = self.scale_Iion / self.parabolic.Cm

        if ionic_model_name in ["HodgkinHuxley", "HH"]:
            self.ionic_model = ionicmodels.HodgkinHuxley(self.scale_im)
        elif ionic_model_name in ["Courtemanche1998", "CRN"]:
            self.ionic_model = ionicmodels.Courtemanche1998(self.scale_im)
        elif ionic_model_name in ["TenTusscher2006_epi", "TTP"]:
            self.ionic_model = ionicmodels.TenTusscher2006_epi(self.scale_im)
        elif ionic_model_name in ["TTP_S", "TTP_SMOOTH"]:
            self.ionic_model = ionicmodels.TenTusscher2006_epi_smooth(self.scale_im)
        elif ionic_model_name in ["BiStable", "BS"]:
            self.ionic_model = ionicmodels.BiStable(self.scale_im)
        else:
            raise Exception("Unknown ionic model.")

        self.size = self.ionic_model.size

    def define_stimulus(self):

        stim_dur = 2.0
        if "cuboid" in self.parabolic.domain_name:
            self.stim_protocol = [[0.0, stim_dur]]  # list of stimuli times and stimuli durations
            self.stim_intensities = [50.0]  # list of stimuli intensities
            self.stim_centers = [[0.0, 0.0, 0.0]]  # list of stimuli centers
            r = 1.5
            self.stim_radii = [[r, r, r]] * len(
                self.stim_protocol
            )  # list of stimuli radii in the three directions (x,y,z)
        elif "cube" in self.parabolic.domain_name:
            self.stim_protocol = [[0.0, 2.0], [1000.0, 10.0]]
            self.stim_intensities = [50.0, 80.0]
            centers = [[0.0, 50.0, 50.0], [58.5, 0.0, 50.0]]
            self.stim_centers = [centers[i] for i in range(len(self.stim_protocol))]
            self.stim_radii = [[1.0, 50.0, 50.0], [1.5, 60.0, 50.0]]
        else:
            raise Exception("Unknown domain name.")

        self.stim_protocol = np.array(self.stim_protocol)
        self.stim_protocol[:, 0] -= self.init_time  # shift stimulus times by the initial time

        # index of the last stimulus applied. The value -1 means no stimulus has been applied yet.
        self.last_stim_index = -1

    def eval_f(self, u, t, fh=None):
        if fh is None:
            fh = self.dtype_f(init=self.init, val=0.0)

        # eval ionic model rhs on u and put result in fh. All indices of the vector of vector fh must be computed (list(range(self.size))
        self.eval_expr(self.ionic_model.f, u, fh, list(range(self.size)), False)
        # apply stimulus
        fh[0] += self.Istim(t)

        # apply diffusion
        self.parabolic.add_disc_laplacian(u[0], fh[0])

        return fh

    def Istim(self, t):
        tol = 1e-8
        for i, (stim_time, stim_dur) in enumerate(self.stim_protocol):
            # Look for which stimulus to apply at the current time t by checking the stimulus protocol:
            # Check if t is in the interval [stim_time, stim_time+stim_dur] with a tolerance tol
            # and apply the corresponding stimulus
            if (t + stim_dur * tol >= stim_time) and (t + stim_dur * tol < stim_time + stim_dur):
                # if the stimulus is not the same as the last one applied, update the last_stim_index and the space_stim vector
                if i != self.last_stim_index:
                    self.last_stim_index = i
                    # get the vector of zeros and ones defining the stimulus region
                    self.space_stim = self.parabolic.stim_region(self.stim_centers[i], self.stim_radii[i])
                    # scale by the stimulus intensity and apply the change of units
                    self.space_stim *= self.scale_im * self.stim_intensities[i]
                return self.space_stim

        return self.parabolic.zero_stim_vec

    def eval_expr(self, expr, u, fh, indeces, zero_untouched_indeces=True):
        # evaluate the expression expr on u and put the result in fh
        # Here expr is a wrapper on a C++ function that evaluates the rhs of the ionic model (or part of it)
        if expr is not None:
            expr(u, fh)

        # indeces is a list of integers indicating which variables are modified by the expression expr.
        # This information is known a priori. Here we use it to zero the variables that are not modified by expr (if zero_untouched_indeces is True)
        if zero_untouched_indeces:
            non_indeces = [i for i in range(self.size) if i not in indeces]
            for i in non_indeces:
                fh[i][:] = 0.0


class MultiscaleMonodomainODE(MonodomainODE):
    """
    The multiscale version of the MonodomainODE problem. This class is used to solve the monodomain equation with a multirate solver.
    The main difference with respect to the MonodomainODE class is that the right-hand side of the ODE system is split into three parts:
    - impl: The discrete Laplacian. This is a stiff term threated implicitly by time integrators.
    - expl: The non stiff term of the ionic models, threated explicitly by time integrators.
    - exp:  The very stiff but diagonal terms of the ionic models, threated exponentially by time integrators.
    """

    dtype_f = imexexp_mesh

    def __init__(self, **problem_params):
        super(MultiscaleMonodomainODE, self).__init__(**problem_params)

        self.define_splittings()

        self.constant_lambda_and_phi = False

    def define_splittings(self):
        """
        This function defines the splittings used in the problem.
        The im_* variables are meant for internal use, the rhs_* for external use (i.e. in the sweeper).
        The *_args and *_indeces are list of integers.
        The *_args are list of variables that are needed to evaluate a function plus the variables that are modified by the function.
        The *_indeces are the list of variables that are modified by the function (subset of args).
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
        # define nonstiff term (explicit part)
        # the wrapper to c++ expression that evaluates the nonstiff part of the ionic model
        self.im_f_nonstiff = self.ionic_model.f_expl
        # the args of f_expl
        self.im_nonstiff_args = self.ionic_model.f_expl_args
        # the indeces of f_expl
        self.im_nonstiff_indeces = self.ionic_model.f_expl_indeces

        # define stiff term (implicit part)
        self.im_f_stiff = None  # no stiff part coming from ionic model to be threated implicitly. Indeed, all the stiff terms are diagonal and are threated exponentially.
        self.im_stiff_args = []
        self.im_stiff_indeces = []

        # define exp term (eponential part)
        # the exponential term is defined by f_exp(u)= lmbda(u)*(u-yinf(u)), hence we only need to define lmbda and yinf
        # the wrapper to c++ expression that evaluates lmbda(u)
        self.im_lmbda_exp = self.ionic_model.lmbda_exp
        # the wrapper to c++ expression that evaluates lmbda(u) and yinf(u)
        self.im_lmbda_yinf_exp = self.ionic_model.lmbda_yinf_exp
        # the args of lmbda and yinf (they are the same)
        self.im_exp_args = self.ionic_model.f_exp_args
        # the indeces of lmbda and yinf
        self.im_exp_indeces = self.ionic_model.f_exp_indeces

        # the spectral radius of the jacobian of non stiff term. We use a bound
        self.rho_nonstiff_cte = self.ionic_model.rho_f_expl()

        self.rhs_stiff_args = self.im_stiff_args
        self.rhs_stiff_indeces = self.im_stiff_indeces
        # Add the potential V index 0 to the rhs_stiff_args and rhs_stiff_indeces.
        # Indeed V is used to compute the Laplacian and is affected by the Laplacian, which is the implicit part of the problem.
        if 0 not in self.rhs_stiff_args:
            self.rhs_stiff_args = [0] + self.rhs_stiff_args
        if 0 not in self.rhs_stiff_indeces:
            self.rhs_stiff_indeces = [0] + self.rhs_stiff_indeces

        self.rhs_nonstiff_args = self.im_nonstiff_args
        self.rhs_nonstiff_indeces = self.im_nonstiff_indeces
        # Add the potential V index 0 to the rhs_nonstiff_indeces. Indeed V is affected by the stimulus, which is a non stiff term.
        if 0 not in self.rhs_nonstiff_indeces:
            self.rhs_nonstiff_indeces = [0] + self.rhs_nonstiff_indeces

        self.im_non_exp_indeces = [i for i in range(self.size) if i not in self.im_exp_indeces]

        self.rhs_exp_args = self.im_exp_args
        self.rhs_exp_indeces = self.im_exp_indeces

        self.rhs_non_exp_indeces = self.im_non_exp_indeces

        # a vector of ones, useful
        self.one = self.dtype_u(init=self.init, val=1.0)

        # some space to store lmbda and yinf
        self.lmbda = self.dtype_u(init=self.init, val=0.0)
        self.yinf = self.dtype_u(init=self.init, val=0.0)

    def solve_system(self, rhs, factor, u0, t, u_sol=None):
        """
        Solve the system u_sol[0] = (M-factor*A)^{-1} * M * rhs[0]
        and sets u_sol[i] = rhs[i] for i>0 (as if A=0 for i>0)

        Arguments:
            rhs (dtype_u): right-hand side
            factor (float): factor multiplying the Laplacian
            u0 (dtype_u): initial guess
            t (float): current time
            u_sol (dtype_u, optional): some space to store the solution. If None, a new space is allocated. Can be the same as rhs.
        """
        if u_sol is None:
            u_sol = self.dtype_u(init=self.init, val=0.0)

        self.parabolic.solve_system(rhs[0], factor, u0[0], t, u_sol[0])

        if rhs is not u_sol:
            for i in range(1, self.size):
                u_sol[i][:] = rhs[i][:]

        return u_sol

    def eval_f(self, u, t, eval_impl=True, eval_expl=True, eval_exp=True, fh=None, zero_untouched_indeces=True):
        """
        Evaluates the right-hand side terms.

        Arguments:
            u (dtype_u): the current solution
            t (float): the current time
            eval_impl (bool, optional): if True, evaluates the implicit part of the right-hand side. Default is True.
            eval_expl (bool, optional): if True, evaluates the explicit part of the right-hand side. Default is True.
            eval_exp (bool, optional): if True, evaluates the exponential part of the right-hand side. Default is True.
            fh (dtype_f, optional): space to store the right-hand side. If None, a new space is allocated. Default is None.
            zero_untouched_indeces (bool, optional): if True, the variables that are not modified by the right-hand side are zeroed. Default is True.
        """

        if fh is None:
            fh = self.dtype_f(init=self.init, val=0.0)

        if eval_expl:
            fh.expl = self.eval_f_nonstiff(u, t, fh.expl, zero_untouched_indeces)

        if eval_impl:
            fh.impl = self.eval_f_stiff(u, t, fh.impl, zero_untouched_indeces)

        if eval_exp:
            fh.exp = self.eval_f_exp(u, t, fh.exp, zero_untouched_indeces)

        return fh

    def eval_f_nonstiff(self, u, t, fh_nonstiff, zero_untouched_indeces=True):
        # eval ionic model nonstiff terms
        self.eval_expr(self.im_f_nonstiff, u, fh_nonstiff, self.im_nonstiff_indeces, zero_untouched_indeces)

        if not zero_untouched_indeces and 0 not in self.im_nonstiff_indeces:
            fh_nonstiff[0][:] = 0.0

        # apply stimulus
        fh_nonstiff[0] += self.Istim(t)

        return fh_nonstiff

    def eval_f_stiff(self, u, t, fh_stiff, zero_untouched_indeces=True):
        # eval ionic model stiff terms
        self.eval_expr(self.im_f_stiff, u, fh_stiff, self.im_stiff_indeces, zero_untouched_indeces)

        if not zero_untouched_indeces and 0 not in self.im_stiff_indeces:
            fh_stiff[0][:] = 0.0

        # apply diffusion
        self.parabolic.add_disc_laplacian(u[0], fh_stiff[0])

        return fh_stiff

    def eval_f_exp(self, u, t, fh_exp, zero_untouched_indeces=True):
        # eval ionic model exp terms f_exp(u)= lmbda(u)*(u-yinf(u)
        self.eval_lmbda_yinf_exp(u, self.lmbda, self.yinf)
        for i in self.im_exp_indeces:
            fh_exp[i][:] = self.lmbda[i] * (u[i] - self.yinf[i])

        if zero_untouched_indeces:
            fh_exp[self.im_non_exp_indeces] = 0.0

        return fh_exp

    def lmbda_eval(self, u, t, lmbda=None):
        if lmbda is None:
            lmbda = self.dtype_u(init=self.init, val=0.0)

        self.eval_lmbda_exp(u, lmbda)

        lmbda[self.im_non_exp_indeces] = 0.0

        return lmbda

    def eval_lmbda_yinf_exp(self, u, lmbda, yinf):
        self.im_lmbda_yinf_exp(u, lmbda, yinf)

    def eval_lmbda_exp(self, u, lmbda):
        self.im_lmbda_exp(u, lmbda)
