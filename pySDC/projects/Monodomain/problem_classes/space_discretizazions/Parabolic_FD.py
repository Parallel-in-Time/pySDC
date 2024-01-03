import numpy as np
import json

from pySDC.core.Common import RegisterParams
from pySDC.projects.Monodomain.problem_classes.space_discretizazions.anysotropic_ND_FD import AnysotropicNDimFinDiff
from scipy.sparse.linalg import spsolve, cg
from pySDC.projects.Monodomain.datatype_classes.FD_Vector import FD_Vector
from pathlib import Path
import os


class Parabolic_FD(RegisterParams):
    def __init__(self, **problem_params):
        self._makeAttributeAndRegister(*problem_params.keys(), localVars=problem_params, readOnly=True)

        self.define_domain()
        self.define_coefficients()
        self.define_Laplacian()
        self.define_solver()
        self.define_stimulus()
        self.define_eval_points()

    def __del__(self):
        if self.enable_output:
            self.output_file.close()
            with open(self.output_file_path.parent / Path(self.output_file_name + '_txyz').with_suffix(".npy"), 'wb') as f:
                np.save(f, np.array(self.t_out))
                xyz = self.NDim_FD.grids
                for i in range(self.dim):
                    np.save(f, xyz[i])

    @property
    def vector_type(self):
        return FD_Vector

    @property
    def mesh_name(self):
        return "ref_" + str(self.pre_refinements)

    def define_domain(self):
        if "small" in self.domain_name:
            self.dom_size = ([0.0, 5.0], [0.0, 3.0], [0.0, 1.0])
        else:
            self.dom_size = ([0.0, 20.0], [0.0, 7.0], [0.0, 3.0])

        self.n_elems = 5.0 * np.max(self.dom_size) * 2**self.pre_refinements + 1
        self.n_elems = int(np.round(self.n_elems))

        self.dim = int(self.domain_name[7])
        self.dom_size = self.dom_size[: self.dim]

    def define_coefficients(self):
        self.chi = 140.0  # mm^-1
        self.Cm = 0.01  # uF/mm^2
        self.si_l = 0.17  # mS/mm
        self.se_l = 0.62  # mS/mm
        self.si_t = 0.019  # mS/mm
        self.se_t = 0.24  # mS/mm

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

    def define_Laplacian(self):
        self.NDim_FD = AnysotropicNDimFinDiff(
            dom_size=self.dom_size,
            nvars=self.n_elems,
            diff=self.diff,
            derivative=2,
            stencil_type='center',
            order=2,
            bc='neumann-zero',
        )
        self.n_dofs = self.NDim_FD.A.shape[0]
        self.init = self.n_dofs

    @property
    def grids(self):
        return self.NDim_FD.grids

    def define_stimulus(self):
        self.scale_Iion = 0.01  # used to convert currents in uA/cm^2 to uA/mm^2
        # scale_im is applied to the rhs of the ionic model, so that the rhs is in units of mV/ms
        self.scale_im = self.scale_Iion / self.Cm
        self.space_stim = self.init_space_stim()
        if abs(self.scale_im - 1.0) > 1e-10:
            self.space_stim *= self.scale_im

    def Istim(self, t):
        for stim_time, stim_dur in self.stim_protocol:
            if (t >= stim_time) and (t <= stim_time + stim_dur):
                return self.space_stim
        return 0.0

    def define_solver(self):
        # we suppose that the problem is symmetric
        if self.dim <= 1:
            self.solver = lambda mat, vec, guess: spsolve(mat, vec)
        else:
            solver_rtol = self.solver_rtol if self.solver_rtol != 'default' else 1e-5
            self.solver = lambda mat, vec, guess: cg(mat, vec, x0=guess, atol=0, tol=solver_rtol)[0]

    def solve_system(self, rhs, factor, u0, t, u_sol):
        u_sol.values[:] = self.solver(self.NDim_FD.Id - factor * self.NDim_FD.A, rhs.values, u0.values)

        return u_sol

    def add_disc_laplacian(self, uh, res):
        res.values += self.NDim_FD.A @ uh.values

    def define_eval_points(self):
        if "small" in self.domain_name:
            n_pts = 5
            a = np.array([[0.5, 0.5, 0.5]])
        else:
            n_pts = 10
            a = np.array([[1.5, 1.5, 1.5]])
        a = a[:, : self.dim]
        dom_size = np.array(self.dom_size)[:, 1]
        b = dom_size.reshape((1, self.dim))
        x = np.reshape(np.linspace(0.0, 1.0, n_pts), (n_pts, 1))
        self.eval_points = a + (b - a) * x

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
            np.save(self.output_file, V.values.reshape(self.NDim_FD.shape))
            self.t_out.append(t)

    def write_reference_solution(self, uh, indeces):
        if self.output_file_path.is_file():
            os.remove(self.output_file_path)
        if not self.output_file_path.parent.is_dir():
            os.makedirs(self.output_file_path.parent)
        with open(self.output_file_path, 'wb') as file:
            [np.save(file, uh[i].values.reshape(self.NDim_FD.shape)) for i in indeces]

    def read_reference_solution(self, uh, indeces, ref_file_name):
        ref_sol_path = Path(self.output_folder) / Path(ref_file_name).with_suffix(".npy")
        if ref_sol_path.is_file():
            with open(ref_sol_path, 'rb') as f:
                for i in indeces:
                    uh[i].values = np.load(f).ravel()
            return True
        else:
            return False

    def eval_on_points(self, u):
        return None

    def init_space_stim(self):
        if not hasattr(self, 'istim_dur'):
            self.istim_dur = -1.0
        stim_dur = self.istim_dur if self.istim_dur >= 0.0 else 2.0
        self.stim_intensity = 35.7143  # in uA/cm^2, it is converted later in uA/mm^2 using self.scale_Iion. This is equivalent to the value used in Niederer et al.

        stim_centers = [[0.0, 0.0, 0.0]]
        if "small" in self.domain_name:
            stim_radius = 0.5
        else:
            stim_radius = 1.5
        self.stim_protocol = [[0.0, stim_dur]]  # list of stim_time, sitm_dur values

        if self.dim == 1:
            dists_stim_centers = []
            x = self.NDim_FD.grids[0]
            for stim_center in stim_centers:
                dists_stim_centers.append(abs(x - stim_center[0]))
        elif self.dim == 2:
            x, y = self.NDim_FD.grids
            dists_stim_centers = []
            for stim_center in stim_centers:
                dists_stim_centers.append(np.maximum(abs(x - stim_center[0]), abs(y - stim_center[1])))
        else:
            x, y, z = self.NDim_FD.grids
            dists_stim_centers = []
            for stim_center in stim_centers:
                dists_stim_centers.append(np.maximum(np.maximum(abs(x - stim_center[0]), abs(y - stim_center[1])), abs(z - stim_center[2])))

        space_conds = []
        for dist_stim_center in dists_stim_centers:
            space_conds.append(dist_stim_center < stim_radius)
        space_conds_or = space_conds[0]
        for i in range(1, len(space_conds)):
            space_conds_or = space_conds_or or space_conds[i]

        space_stim = FD_Vector(self.init)
        space_stim.values[:] = space_conds_or.ravel() * self.stim_intensity

        return space_stim

    def compute_errors(self, uh, ref_sol):
        # Compute L2 error
        error_L2 = np.linalg.norm(uh.values - ref_sol.values)
        sol_norm_L2 = np.linalg.norm(ref_sol.values)
        rel_error_L2 = error_L2 / sol_norm_L2

        return error_L2, rel_error_L2

    def get_dofs_stats(self):
        data = (
            self.space_stim.n_loc_dofs + self.space_stim.n_ghost_dofs,
            self.space_stim.n_loc_dofs,
            self.space_stim.n_ghost_dofs,
            self.space_stim.n_ghost_dofs / (self.space_stim.n_loc_dofs + self.space_stim.n_loc_dofs),
        )
        return (data,), data
