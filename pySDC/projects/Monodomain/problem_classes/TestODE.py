from pathlib import Path
import logging
import numpy as np
from pySDC.core.Problem import ptype
from pySDC.core.Common import RegisterParams
from pySDC.projects.Monodomain.datatype_classes.FD_Vector import FD_Vector
from pySDC.projects.Monodomain.datatype_classes.VectorOfVectors import VectorOfVectors, IMEXEXP_VectorOfVectors
from pySDC.core.Collocation import CollBase
import scipy
import os


class dummy_communicator:
    rank = 0


class Parabolic(RegisterParams):
    def __init__(self, **problem_params):
        self._makeAttributeAndRegister(*problem_params.keys(), localVars=problem_params, readOnly=True)
        self.vector_type = FD_Vector

        self.comm = dummy_communicator()

        x = np.array([0.0])
        self.grids = (x,)

    def __del__(self):
        if self.enable_output:
            self.output_file.close()
            with open(self.output_file_path.parent / Path(self.output_file_name + '_t').with_suffix(".npy"), 'wb') as f:
                np.save(f, np.array(self.t_out))

    def init_output(self, output_folder):
        self.output_folder = output_folder
        self.output_file_path = self.output_folder / Path(self.output_file_name).with_suffix(".npy")
        if self.enable_output:
            if self.output_file_path.is_file():
                os.remove(self.output_file_path)
            if not self.output_folder.is_dir():
                os.makedirs(self.output_folder)
            self.output_file = open(self.output_file_path, 'wb')
            self.t_out = []

    def write_solution(self, V, t):
        if self.enable_output:
            np.save(self.output_file, V.values)
            self.t_out.append(t)

    def get_dofs_stats(self):
        data = (1, 1, 0, 0)
        return (data,), data


class TestODE(ptype):
    def __init__(self, **problem_params):
        self.logger = logging.getLogger("step")
        self.comm = dummy_communicator()

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

        # assert self.lmbda_laplacian <= 0.0 and self.lmbda_gating <= 0.0 and self.lmbda_others <= 0.0, "lmbda_laplacian, lmbda_gating and lmbda_others must be negative"

        # dtype_u and dtype_f are super vectors of vector_type
        self.vector_type = self.parabolic.vector_type

        def dtype_u(init=None, val=0.0):
            return VectorOfVectors(init, val, self.vector_type, self.size)

        def dtype_f(init=None, val=0.0):
            return VectorOfVectors(init, val, self.vector_type, self.size)

        self.dtype_u = dtype_u
        self.dtype_f = dtype_f

        # init output stuff
        self.output_folder = Path(self.output_root)
        self.parabolic.init_output(self.output_folder)

    def write_solution(self, uh, t):
        # write solution to file, only the potential V=uh[0], not the ionic model variables
        self.parabolic.write_solution(uh[0], t)

    def write_reference_solution(self, uh, all=False):
        raise Exception("write_reference_solution not implemented")

    def read_reference_solution(self, uh, ref_file_name, all=False):
        raise Exception("read_reference_solution not implemented")

    def initial_value(self):
        u0 = self.dtype_u(self.init, val=1.0)

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
        exact_sol = self.vector_type(init=self.init, val=0.0)
        exact_sol.values[0] = np.exp((self.lmbda_laplacian + self.lmbda_gating + self.lmbda_others) * self.Tend)
        error_L2 = np.linalg.norm(uh[0].values - exact_sol.values)
        rel_error_L2 = error_L2 / np.linalg.norm(exact_sol.values)

        if self.comm.rank == 0:
            print(f"L2-errors: {error_L2}")
            print(f"Relative L2-errors: {rel_error_L2}")

        return True, error_L2, rel_error_L2

    def getSize(self):
        # return number of dofs in the mesh
        return 1

    def eval_on_points(self, u):
        return [u[0].values[0]]

    def eval_f(self, u, t, fh=None):
        if fh is None:
            fh = self.dtype_f(init=self.init, val=0.0)

        # apply stimulus
        fh.val_list[0].values[0] = (self.lmbda_laplacian + self.lmbda_gating + self.lmbda_others) * u[0].values[0]

        return fh

    def rho(self, y, t, fy):
        return abs(self.lmbda_laplacian + self.lmbda_gating + self.lmbda_others)


class MultiscaleTestODE(TestODE):
    def __init__(self, **problem_params):
        super(MultiscaleTestODE, self).__init__(**problem_params)

        def dtype_f(init=None, val=0.0):
            return IMEXEXP_VectorOfVectors(init, val, self.vector_type, self.size)

        self.dtype_f = dtype_f

        self.rhs_stiff_args = [0]
        self.rhs_exp_indeces = [0]

        self.constant_lambda_and_phi = True

    def rho_nonstiff(self, y, t, fy=None):
        return abs(self.lmbda_others)

    def rho_stiff(self, y, t, fy=None):
        return abs(self.lmbda_laplacian)

    def eval_lmbda_exp(self, u, lmbda):
        self.im_lmbda_exp(u.np_list, lmbda.np_list)

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

        # evaluate explicit (non stiff) part M^-1*f_nonstiff(u,t)
        if eval_expl:
            fh.expl.val_list[0].values[0] = self.lmbda_others * u[0].values[0]

        # evaluate implicit (stiff) part M^1*A*u+M^-1*f_stiff(u,t)
        if eval_impl:
            fh.impl.val_list[0].values[0] = self.lmbda_laplacian * u[0].values[0]

        # evaluate exponential part
        if eval_exp:
            fh.exp.val_list[0].values[0] = self.lmbda_gating * u[0].values[0]

        return fh

    def eval_phi_f_exp(self, u, factor, t, phi_f_exp=None, zero_untouched_indeces=True):
        if phi_f_exp is None:
            phi_f_exp = self.dtype_u(init=self.init, val=0.0)

        phi_f_exp.val_list[0].values[0] = (np.exp(factor * self.lmbda_gating) - 1.0) / (factor) * u.val_list[0].values[0]

        return phi_f_exp

    def phi_eval(self, u, factor, t, k, phi=None):
        if phi is None:
            phi = self.dtype_u(init=self.init, val=0.0)

        dt_lmbda = factor * self.lmbda_gating

        if k == 0:
            phi.val_list[0].values[0] = np.exp(dt_lmbda)
        else:
            num_nodes = 10
            self.coll = CollBase(num_nodes=num_nodes, tleft=0, tright=1, node_type='LEGENDRE', quad_type='GAUSS')
            c = self.coll.nodes
            b = self.coll.weights

            phi.val_list[0].values[0] = (b[0] * c[0] ** (k - 1)) * np.exp((1.0 - c[0]) * dt_lmbda)
            for j in range(1, num_nodes):
                phi.val_list[0].values[0] += (b[j] * c[j] ** (k - 1)) * np.exp((1.0 - c[j]) * dt_lmbda)
            phi.val_list[0].values[0] /= scipy.special.factorial(k - 1)

        return phi

    def lmbda_eval(self, u, t, lmbda=None):
        if lmbda is None:
            lmbda = self.dtype_u(init=self.init, val=0.0)

        lmbda.val_list[0].values[0] = self.lmbda_gating

        return lmbda
