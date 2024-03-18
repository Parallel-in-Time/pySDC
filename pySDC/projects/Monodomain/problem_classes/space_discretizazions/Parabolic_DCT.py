import numpy as np
import scipy as sp
from pySDC.core.Common import RegisterParams
from pySDC.projects.Monodomain.datatype_classes.DCT_Vector import DCT_Vector
from pathlib import Path
import os


class Parabolic_DCT(RegisterParams):
    def __init__(self, **problem_params):
        self._makeAttributeAndRegister(*problem_params.keys(), localVars=problem_params, readOnly=True)

        self.define_domain()
        self.define_coefficients()
        self.define_diffusion()
        self.define_stimulus()

    def __del__(self):
        if self.enable_output:
            self.output_file.close()
            with open(
                self.output_file_path.parent / Path(self.output_file_name + '_txyz').with_suffix(".npy"), 'wb'
            ) as f:
                np.save(f, np.array(self.t_out))
                xyz = self.grids
                for i in range(self.dim):
                    np.save(f, xyz[i])

    @property
    def vector_type(self):
        return DCT_Vector

    @property
    def mesh_name(self):
        return "ref_" + str(self.refinements)

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
            self.dom_size = (100.0, 100.0, 100.0)
            self.dim = int(self.domain_name[5])
        else:  # cuboid
            if "smaller" in self.domain_name:
                self.dom_size = (10.0, 4.5, 2.0)
            elif "small" in self.domain_name:
                self.dom_size = (5.0, 3.0, 1.0)
            elif "very_large" in self.domain_name:
                self.dom_size = (280.0, 112.0, 48.0)
            elif "large" in self.domain_name:
                self.dom_size = (60.0, 21.0, 9.0)
            else:
                self.dom_size = (20.0, 7.0, 3.0)
            self.dim = int(self.domain_name[7])

        self.dom_size = self.dom_size[: self.dim]
        self.n_elems = [int(2 ** np.round(np.log2(5.0 * L * 2**self.refinements))) for L in self.dom_size]
        self.grids, self.dx = self.get_grids_dx(self.dom_size, self.n_elems)

        self.shape = tuple(np.flip([x.size for x in self.grids]))
        self.n_dofs = int(np.prod(self.shape))
        self.init = self.n_dofs

    def define_diffusion(self):
        N = self.n_elems
        dx = self.dx
        dim = len(N)
        if self.order == 2:
            diff_dct = self.diff[0] * (2.0 * np.cos(np.pi * np.arange(N[0]) / N[0]) - 2.0) / dx[0] ** 2
            if dim >= 2:
                diff_dct = (
                    diff_dct[None, :]
                    + self.diff[1]
                    * np.array((2.0 * np.cos(np.pi * np.arange(N[1]) / N[1]) - 2.0) / dx[1] ** 2)[:, None]
                )
            if dim >= 3:
                diff_dct = (
                    diff_dct[None, :, :]
                    + self.diff[2]
                    * np.array((2.0 * np.cos(np.pi * np.arange(N[2]) / N[2]) - 2.0) / dx[2] ** 2)[:, None, None]
                )
        elif self.order == 4:
            diff_dct = (
                self.diff[0]
                * (
                    (-1.0 / 6.0) * np.cos(2.0 * np.pi * np.arange(N[0]) / N[0])
                    + (8.0 / 3.0) * np.cos(np.pi * np.arange(N[0]) / N[0])
                    - 2.5
                )
                / dx[0] ** 2
            )
            if dim >= 2:
                diff_dct = (
                    diff_dct[None, :]
                    + self.diff[1]
                    * np.array(
                        (
                            (-1.0 / 6.0) * np.cos(2.0 * np.pi * np.arange(N[1]) / N[1])
                            + (8.0 / 3.0) * np.cos(np.pi * np.arange(N[1]) / N[1])
                            - 2.5
                        )
                        / dx[1] ** 2
                    )[:, None]
                )
            if dim >= 3:
                diff_dct = (
                    diff_dct[None, :, :]
                    + self.diff[2]
                    * np.array(
                        (
                            (-1.0 / 6.0) * np.cos(2.0 * np.pi * np.arange(N[2]) / N[2])
                            + (8.0 / 3.0) * np.cos(np.pi * np.arange(N[2]) / N[2])
                            - 2.5
                        )
                        / dx[2] ** 2
                    )[:, None, None]
                )
        else:
            raise NotImplementedError("Only order 2 and 4 are implemented for Parabolic_DCT.")
        self.diff_dct = diff_dct

    def grids_from_x(self, x):
        dim = len(x)
        if dim == 1:
            return (x[0],)
        elif dim == 2:
            return (x[0][None, :], x[1][:, None])
        elif dim == 3:
            return (x[0][None, None, :], x[1][None, :, None], x[2][:, None, None])

    def get_grids_dx(self, dom_size, N):
        x = [np.linspace(0, dom_size[i], 2 * N[i] + 1) for i in range(len(N))]
        x = [xi[1::2] for xi in x]
        dx = [xi[1] - xi[0] for xi in x]
        return self.grids_from_x(x), dx

    def define_stimulus(self):
        self.zero_stim_vec = 0.0
        # all remaining stimulus parameters are set in MonodomainODE

    def solve_system(self, rhs, factor, u0, t, u_sol):
        rhs_hat = sp.fft.dctn(rhs.values.reshape(self.shape))
        u_sol_hat = rhs_hat / (1.0 - factor * self.diff_dct)
        u_sol.values[:] = sp.fft.idctn(u_sol_hat).ravel()

        return u_sol

    def add_disc_laplacian(self, uh, res):
        res.values += sp.fft.idctn(self.diff_dct * sp.fft.dctn(uh.values.reshape(self.shape))).ravel()

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
        if ref_file_name == "":
            return False
        ref_sol_path = Path(self.output_folder) / Path(ref_file_name).with_suffix(".npy")
        if ref_sol_path.is_file():
            with open(ref_sol_path, 'rb') as f:
                for i in indeces:
                    uh[i].values[:] = np.load(f).ravel()
            return True
        else:
            return False

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
