import numpy as np
import scipy as sp
from pySDC.core.Common import RegisterParams
from pySDC.projects.Monodomain.datatype_classes.FD_Vector import FD_Vector
from pathlib import Path
import os


class Parabolic_DCT(RegisterParams):
    def __init__(self, **problem_params):
        self._makeAttributeAndRegister(*problem_params.keys(), localVars=problem_params, readOnly=True)

        self.define_domain()
        self.define_coefficients()
        self.define_laplacian()
        self.define_stimulus()
        self.define_eval_points()

        assert self.bc == 'N', "bc must be 'N'"

    def __del__(self):
        if self.enable_output:
            self.output_file.close()
            with open(self.output_file_path.parent / Path(self.output_file_name + '_txyz').with_suffix(".npy"), 'wb') as f:
                np.save(f, np.array(self.t_out))
                xyz = self.grids
                for i in range(self.dim):
                    np.save(f, xyz[i])

    @property
    def vector_type(self):
        return FD_Vector

    @property
    def mesh_name(self):
        return "ref_" + str(self.pre_refinements)

    def define_solver(self):
        pass

    def define_coefficients(self):
        self.chi = 140.0  # mm^-1
        self.Cm = 0.01  # uF/mm^2
        self.si_l = 0.17  # mS/mm
        self.se_l = 0.62  # mS/mm
        self.si_t = 0.019  # mS/mm
        self.se_t = 0.24  # mS/mm

        if "cube" in self.domain_name:
            # if self.pre_refinements == -1: # only for generating initial value
            # self.si_l *= 0.5
            # self.se_l *= 0.5
            # elif self.pre_refinements == 0:
            # self.si_l *= 0.25
            # self.se_l *= 0.25
            self.si_t = self.si_l
            self.se_t = self.se_l

        self.sigma_l = self.si_l * self.se_l / (self.si_l + self.se_l)
        self.sigma_t = self.si_t * self.se_t / (self.si_t + self.se_t)
        self.diff_l = self.sigma_l / self.chi / self.Cm
        self.diff_t = self.sigma_t / self.chi / self.Cm

        if self.dim == 1:
            self.diff = (self.diff_l,)
        elif self.dim == 2:
            self.diff = (self.diff_l, self.diff_t)
        else:
            self.diff = (self.diff_l, self.diff_t, self.diff_t)

    def define_domain(self):
        if "cube" in self.domain_name:
            self.dom_size = ([0.0, 100.0], [0.0, 100.0], [0.0, 100.0])
            self.dim = int(self.domain_name[5])
        else:  # cuboid
            if "smaller" in self.domain_name:
                self.dom_size = ([0.0, 10.0], [0.0, 4.5], [0.0, 2.0])
            elif "small" in self.domain_name:
                self.dom_size = ([0.0, 5.0], [0.0, 3.0], [0.0, 1.0])
            elif "very_large" in self.domain_name:
                self.dom_size = ([0.0, 280.0], [0.0, 112.0], [0.0, 48.0])
            elif "large" in self.domain_name:
                self.dom_size = ([0.0, 60.0], [0.0, 21.0], [0.0, 9.0])
            else:
                self.dom_size = ([0.0, 20.0], [0.0, 7.0], [0.0, 3.0])
            self.dim = int(self.domain_name[7])

        # self.n_elems = 5.0 * np.max(self.dom_size) * 2**self.pre_refinements + 1
        # self.n_elems = int(np.round(self.n_elems))

        assert self.dim == 1, "only 1D implemented"

        self.dom_size = self.dom_size[: self.dim]
        if self.dim == 2:
            assert self.dom_size[0][1] == self.dom_size[1][1], "only cube implemented"
        if self.dim == 3:
            assert self.dom_size[0][1] == self.dom_size[1][1] and self.dom_size[0][1] == self.dom_size[2][1], "only cube implemented"

        self.L = self.dom_size[0][1]
        # self.n_elems = int(2 ** np.round(np.log2(5.0 * self.L * 2**self.pre_refinements)))
        # x = np.linspace(0, self.L, 2 * self.n_elems + 1)
        # self.grids = (x[1::2],)
        self.n_elems, self.grids = self.get_mesh(self.L, self.pre_refinements)
        self.shape = (self.grids[0].size,)
        self.n_dofs = int(np.prod(self.shape))
        self.init = self.n_dofs

        self.dx = np.max([grid[1] - grid[0] for grid in self.grids])

        # self.fine_mesh_pre_refinements = 2
        # self.fine_mesh_n_elems, self.fine_mesh_grids = self.get_mesh(self.L, self.fine_mesh_pre_refinements)

    def define_laplacian(self):
        self.dct_lap = self.get_dct_laplacian(self.n_elems, self.grids)
        # self.fine_mesh_dct_lap = self.get_dct_laplacian(self.fine_mesh_n_elems, self.fine_mesh_grids)

    def define_stimulus(self):
        self.zero_stim_vec = 0.0
        # all remaining stimulus parameters are set in MonodomainODE

    def get_mesh(self, L, pre_refinements):
        n_elems = int(2 ** np.round(np.log2(5.0 * L * 2**pre_refinements)))
        x = np.linspace(0, L, 2 * n_elems + 1)
        grids = (x[1::2],)
        return n_elems, grids

    def get_dct_laplacian(self, n_elems, grids):
        x = grids[0]
        dx = x[1] - x[0]
        dct_lap = self.diff_l * (2.0 * np.cos(np.pi * np.arange(n_elems) / n_elems) - 2.0) / dx**2
        return dct_lap

    def solve_system(self, rhs, factor, u0, t, u_sol):
        rhs_hat = sp.fft.dct(rhs.values)
        u_sol_hat = rhs_hat / (1.0 - factor * self.dct_lap)
        u_sol.values[:] = sp.fft.idct(u_sol_hat)

        # rhs_hat = sp.fft.dct(rhs.values)
        # u_sol_hat = rhs_hat / (1.0 - factor * self.fine_mesh_dct_lap[: self.n_elems])
        # u_sol.values[:] = sp.fft.idct(u_sol_hat)

        return u_sol

    def add_disc_laplacian(self, uh, res):
        res.values += sp.fft.idct(self.dct_lap * sp.fft.dct(uh.values))

        # res.values += sp.fft.idct(self.fine_mesh_dct_lap[: self.n_elems] * sp.fft.dct(uh.values))

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

    def write_solution(self, u, t, all):
        if self.enable_output:
            if not all:
                np.save(self.output_file, u[0].values.reshape(self.shape))
                self.t_out.append(t)
            else:
                raise NotImplementedError("all=True not implemented for Parabolic_FD.write_solution")

    def write_reference_solution(self, uh, indeces):
        if self.output_file_path.is_file():
            os.remove(self.output_file_path)
        if not self.output_file_path.parent.is_dir():
            os.makedirs(self.output_file_path.parent)
        with open(self.output_file_path, 'wb') as file:
            [np.save(file, uh[i].values.reshape(self.shape)) for i in indeces]

    def read_reference_solution(self, uh, indeces, ref_file_name):
        ref_sol_path = Path(self.output_folder) / Path(ref_file_name).with_suffix(".npy")
        if ref_sol_path.is_file():
            with open(ref_sol_path, 'rb') as f:
                for i in indeces:
                    uh[i].values[:] = np.load(f).ravel()
            return True
        else:
            return False

    def define_eval_points(self):
        pass

    def eval_on_points(self, u):
        return None

    def stim_region(self, stim_center, stim_radius):
        grids = self.grids
        coord_inside_stim_box = []
        for i in range(len(grids)):
            coord_inside_stim_box.append(abs(grids[i] - stim_center[i]) < stim_radius[i])

        inside_stim_box = True
        for i in range(len(grids)):
            inside_stim_box = np.logical_and(inside_stim_box, coord_inside_stim_box[i])

        return self.vector_type(inside_stim_box.ravel().astype(float))

    def compute_errors(self, uh, ref_sol):
        # Compute L2 error
        error_L2 = np.linalg.norm(uh.values - ref_sol.values)
        sol_norm_L2 = np.linalg.norm(ref_sol.values)
        rel_error_L2 = error_L2 / sol_norm_L2

        return error_L2, rel_error_L2

    def get_dofs_stats(self):
        tmp = self.vector_type(self.init)
        data = (
            tmp.n_loc_dofs + tmp.n_ghost_dofs,
            tmp.n_loc_dofs,
            tmp.n_ghost_dofs,
            tmp.n_ghost_dofs / (tmp.n_loc_dofs + tmp.n_loc_dofs),
        )
        return (data,), data
