import numpy as np
from dolfinx import mesh, fem, io, geometry
import ufl
import adios4dolfinx
from mpi4py import MPI
import h5py
import basix
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType
from pathlib import Path
import json

from pySDC.core.Common import RegisterParams
from pySDC.core.Errors import ParameterError
from pySDC.projects.Monodomain_NEW.datatype_classes.FEniCSx_Vector import FEniCSx_Vector


class Parabolic_FEniCSx(RegisterParams):
    def __init__(self, **problem_params):
        self._makeAttributeAndRegister(*problem_params.keys(), localVars=problem_params, readOnly=True)

        if self.mass_lumping and (self.family != "CG" or self.order > 1):
            raise Exception("You have specified mass_lumping=True but for order>1 or family!='CG'.")

        self.define_domain_and_function_space()
        self.define_coefficients()
        self.define_domain_dependent_variables()
        self.define_variational_forms()
        self.define_stimulus()
        self.assemble_vec_mat()
        self.set_solver_options()
        self.define_mass_solver()
        self.define_eval_points()

    def __del__(self):
        if self.enable_output:
            self.xdmf.close()

    @property
    def vector_type(self):
        return FEniCSx_Vector

    @property
    def mesh_name(self):
        return "ref_" + str(self.pre_refinements)

    def define_domain_and_function_space(self):
        self.mesh_fibers_folder = Path(self.meshes_fibers_root_folder) / Path(self.domain_name) / Path(self.mesh_name)
        self.import_fibers = "cuboid" not in self.domain_name
        if self.import_fibers:
            # read mesh from same file as fibers
            with io.XDMFFile(self.comm, self.mesh_fibers_folder / Path("fibers.xdmf"), "r") as xdmf:
                self.domain = xdmf.read_mesh(name="mesh", xpath="Xdmf/Domain/Grid")
            self.dim = 3
        else:
            if "small" in self.domain_name:
                dom_size = [[0.0, 0.0, 0.0], [5.0, 3.0, 1.0]]
                n_elems = 25 * 2**self.pre_refinements
            else:
                dom_size = [[0.0, 0.0, 0.0], [20.0, 7.0, 3.0]]
                n_elems = 100 * 2**self.pre_refinements
            self.dim = int(self.domain_name[7])

            d = np.asarray(dom_size[1]) - np.asarray(dom_size[0])
            max_d = np.max(d)
            n = [n_elems] * self.dim
            for i in range(len(n)):
                n[i] = int(np.ceil(n[i] * d[i] / max_d))

            if self.dim == 1:
                self.domain = mesh.create_interval(comm=self.comm, nx=n_elems, points=[dom_size[0][0], dom_size[1][0]])
            elif self.dim == 2:
                self.domain = mesh.create_rectangle(comm=self.comm, n=n, cell_type=mesh.CellType.triangle, points=[dom_size[0][: self.dim], dom_size[1][: self.dim]])
            elif self.dim == 3:
                self.domain = mesh.create_box(comm=self.comm, n=n, cell_type=mesh.CellType.tetrahedron, points=[dom_size[0][: self.dim], dom_size[1][: self.dim]])
            else:
                raise Exception(f"need dim=1,2,3 to instantiate problem, got dim={self.dim}")

            # with io.XDMFFile(self.comm, self.mesh_fibers_folder / Path("fibers.xdmf"), "r") as xdmf:
            #     self.domain = xdmf.read_mesh()

        for i in range(self.post_refinements):
            self.domain.topology.create_connectivity(self.domain.topology.dim, 1)
            self.domain = mesh.refine(self.domain)

        self.dim = self.domain.geometry.dim
        self.V = fem.FunctionSpace(self.domain, (self.family, self.order))
        self.init = self.V

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

    def define_domain_dependent_variables(self):
        self.x = ufl.SpatialCoordinate(self.domain)
        self.n = ufl.FacetNormal(self.domain)
        self.t = fem.Constant(self.domain, 0.0)
        self.zero_fun = fem.Constant(self.domain, 0.0)

        self.define_fibers()
        if self.fibrosis:
            fibrosis_r_field, percentages, quantiles = self.read_fibrosis()
            quantile = quantiles[1]  # quantiles are for 30%, 50%, 70%
            diff_t_ufl_expr = (self.sigma_t / self.chi / self.Cm) * ufl.conditional(ufl.lt(fibrosis_r_field, quantile), 1.0, 0.0)
            self.diff_t = fem.Function(self.V)
            self.diff_t.interpolate(fem.Expression(diff_t_ufl_expr, self.V.element.interpolation_points()))

    def define_variational_forms(self):
        u, v = ufl.TrialFunction(self.V), ufl.TestFunction(self.V)

        self.mass = u * v * ufl.dx

        def diff_tens(w):
            return self.diff_l * self.f0 * ufl.dot(self.f0, w) + self.diff_t * self.s0 * ufl.dot(self.s0, w) + self.diff_t * self.n0 * ufl.dot(self.n0, w)

        if self.family == "CG":
            self.diff = ufl.dot(diff_tens(ufl.grad(u)), ufl.grad(v)) * ufl.dx

        elif self.family == "DG":
            n = ufl.FacetNormal(self.domain)
            h = ufl.FacetArea(self.domain)
            dim = self.dim
            p = 1  # self.order
            beta0 = 1.0 / (dim - 1)
            if dim == 2:
                eta = fem.Constant(self.domain, ScalarType(20.0)) * p * (p + 1)
            elif dim == 3:
                eta = fem.Constant(self.domain, ScalarType(20.0)) * p * (p + 2) * (h**0.5)

            delta = ufl.dot(n, diff_tens(n))
            om_p = delta("-") / (delta("+") + delta("-"))
            om_m = delta("+") / (delta("+") + delta("-"))
            gamma = 2.0 * delta("+") * delta("-") / (delta("-") + delta("+"))

            def avg_o(w):
                return om_p * w("+") + om_m * w("-")

            self.diff = (
                ufl.inner(diff_tens(ufl.grad(u)), ufl.grad(v)) * ufl.dx
                - ufl.inner(avg_o(diff_tens(ufl.grad(v))), ufl.jump(u, n)) * ufl.dS
                - ufl.inner(ufl.jump(v, n), avg_o(diff_tens(ufl.grad(u)))) * ufl.dS
                + eta / (h**beta0) * gamma * ufl.inner(ufl.jump(v, n), ufl.jump(u, n)) * ufl.dS
            )
        else:
            raise ParameterError("problem_params['family'] must be either 'CG' or 'DG'")

        self.mass_form = fem.form(self.mass)
        self.diff_form = fem.form(self.diff)

    def define_stimulus(self):
        self.scale_Iion = 0.01  # used to convert currents in uA/cm^2 to uA/mm^2
        # scale_im is applied to the rhs of the ionic model, so that the rhs is in units of mV/ms
        self.scale_im = self.scale_Iion / self.Cm
        self.stim_ufl = self.stim_ufl_expr()
        if abs(self.scale_im - 1.0) > 1e-10:
            self.stim_ufl *= self.scale_im

        self.stim_vec = FEniCSx_Vector(init=self.init, val=0.0)
        self.stim_expr = fem.Expression(self.stim_ufl, self.V.element.interpolation_points())

    def Istim(self, t):
        self.t.value = t
        self.stim_vec.values.interpolate(self.stim_expr)
        return self.stim_vec

    def assemble_vec_mat(self):
        from dolfinx.fem.petsc import assemble_matrix

        self.K = fem.petsc.assemble_matrix(self.diff_form)
        self.K.assemble()

        if self.mass_lumping:
            u_tmp = fem.Function(self.V)
            u_tmp.interpolate(lambda x: 1.0 + 0.0 * x[0])
            mass_lumped = ufl.action(self.mass, u_tmp)
            self.M = fem.petsc.assemble_matrix(self.mass_form)
            self.M.zeroEntries()
            self.ml = fem.petsc.create_vector(fem.form(mass_lumped))
            with self.ml.localForm() as m_loc:
                m_loc.set(0)
            fem.petsc.assemble_vector(self.ml, fem.form(mass_lumped))
            self.ml.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            self.ml.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            self.M.setDiagonal(self.ml)
        else:
            self.M = fem.petsc.assemble_matrix(self.mass_form)
            self.M.assemble()

        # define two temporary vectors
        self.tmp1 = FEniCSx_Vector(init=self.init, val=0.0)
        self.tmp2 = FEniCSx_Vector(init=self.init, val=0.0)

    def set_solver_options(self):
        # we suppose that the problem is symmetric
        # first set the default options
        if self.dim <= 2:
            def_solver_ksp = "preonly"
            def_solver_pc = "cholesky"
        else:
            def_solver_ksp = "cg"
            def_solver_pc = "hypre"

        if not hasattr(self, "solver_ksp"):
            self.solver_ksp = def_solver_ksp
        if not hasattr(self, "solver_pc"):
            self.solver_pc = def_solver_pc

    def define_mass_solver(self):
        if not self.mass_lumping:
            self.solver_M = PETSc.KSP().create(self.comm)
            self.solver_M.setOperators(self.M)
            self.solver_M.setType(self.solver_ksp)
            self.solver_M.getPC().setType(self.solver_pc)

    def define_solver(self):
        self.prev_factor = -1.0
        self.solver = PETSc.KSP().create(self.comm)
        self.solver.setType(self.solver_ksp)
        self.solver.getPC().setType(self.solver_pc)
        if self.solver_ksp != "preonly" and hasattr(self, "solver_rtol") and self.solver_rtol != "default":
            assert type(self.solver_rtol) is float, 'problem_params["solver_rtol"] must be a float or "default"'
            self.solver.setTolerances(rtol=self.solver_rtol)

    def solve_system(self, rhs, factor, u0, t, u_sol):
        if abs(factor - self.prev_factor) > 1e-8 * factor:
            self.prev_factor = factor
            self.solver.setOperators(self.M + factor * self.K)

        self.M.mult(rhs.values.vector, self.tmp1.values.vector)
        self.solver.solve(self.tmp1.values.vector, u_sol.values.vector)

        return u_sol

    def add_disc_laplacian(self, uh, res):
        self.K.mult(uh.values.vector, self.tmp1.values.vector)  # WARNING: DO NOT DO -u[0].values.vector HERE AND += self.interp_f BELOW SINCE IT CREATES MEMORY LEAK!!!
        self.invert_mass_matrix(self.tmp1.values.vector, self.tmp2.values.vector)
        res -= self.tmp2

    def define_eval_points(self):
        # used to compute CV, valid only on cuboid domain
        if "cuboid" in self.domain_name:
            if "small" in self.domain_name:
                n_pts = 5
                self.dom_size = [[0.0, 0.0, 0.0], [5.0, 3.0, 1.0]]
                a = np.array([[0.5, 0.5, 0.5]])
            else:
                n_pts = 10
                self.dom_size = [[0.0, 0.0, 0.0], [20.0, 7.0, 3.0]]
                a = np.array([[1.5, 1.5, 1.5]])
            b = np.reshape(np.array(self.dom_size[1]), (1, 3))
            x = np.reshape(np.linspace(0.0, 1.0, n_pts), (n_pts, 1))
            self.eval_points = np.zeros((n_pts, 3))
            for i in range(n_pts):
                self.eval_points[i, :] = np.reshape(a + (b - a) * x[i], (3,))
            if self.dim == 1:
                self.eval_points[:, 1:] = 0
            elif self.dim == 2:
                self.eval_points[:, 2] = 0

            bb_tree = geometry.bb_tree(self.domain, self.domain.topology.dim)
            self.cells = []
            self.points_on_proc = []
            # Find cells whose bounding-box collide with the the points
            cell_candidates = geometry.compute_collisions_points(bb_tree, self.eval_points)
            # Choose one of the cells that contains the point
            colliding_cells = geometry.compute_colliding_cells(self.domain, cell_candidates, self.eval_points)
            for i, point in enumerate(self.eval_points):
                if len(colliding_cells.links(i)) > 0:
                    self.points_on_proc.append(point)
                    self.cells.append(np.min(colliding_cells.links(i)[0]))
            self.points_on_proc = np.array(self.points_on_proc, dtype=np.float64)
            self.loc_glob_map = []
            for i in range(self.points_on_proc.shape[0]):
                p = self.points_on_proc[i, :]
                for j in range(self.eval_points.shape[0]):
                    q = self.eval_points[j, :]
                    if np.linalg.norm(p - q) < np.max([np.linalg.norm(p), np.linalg.norm(q)]) * 1e-5:
                        self.loc_glob_map.append(j)
                        break

    def init_output(self, output_folder):
        self.output_folder = output_folder
        if self.enable_output:
            self.xdmf = io.XDMFFile(self.domain.comm, self.output_folder / Path(self.output_file_name).with_suffix(".xdmf"), "w")
            self.xdmf.write_mesh(self.domain)

    def invert_mass_matrix(self, x, y):
        # solves y = M*y
        if self.mass_lumping:
            y.pointwiseDivide(x, self.ml)
        else:
            self.solver_M.solve(x, y)

    def write_solution(self, V, t):
        if self.enable_output:
            if self.family == "CG":
                V.values.name = "V"
                self.xdmf.write_function(V.values, t)
            else:
                raise Exception("write_solution sol not implemented for DG")
                # if not hasattr(self, "V_CG"):
                #     self.V_CG = fem.FunctionSpace(self.domain, ("CG", self.order))
                #     self.u_CG = fem.Function(self.V_CG)
                # V.ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD, False)
                # self.u_CG.interpolate(
                #     V.values,
                #     nmm_interpolation_data=fem.create_nonmatching_meshes_interpolation_data(
                #         self.u_CG.function_space.mesh._cpp_object, self.u_CG.function_space.element, V.values.function_space.mesh._cpp_object
                #     ),
                # )
                # self.u_CG.name = "V"
                # self.xdmf.write_function(self.u_CG, t)

    def write_reference_solution(self, uh, indeces):
        [uh[i].ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD) for i in indeces]
        if self.family == "CG":
            adios4dolfinx.write_mesh(self.domain, self.output_folder / Path(self.output_file_name).with_suffix(".bp"), engine="BP4")
            [adios4dolfinx.write_function(uh[i].values, self.output_folder / Path(self.output_file_name + "_" + str(i)).with_suffix(".bp"), engine="BP4") for i in indeces]
        else:
            raise Exception("write_reference_solution not implemented for DG")

    def read_reference_solution(self, uh, indeces, ref_file_name):
        if self.family == "CG":
            ref_sol_path = Path(self.output_folder) / Path(ref_file_name)
            if ref_sol_path.with_suffix(".bp").is_dir():
                mesh_ref = adios4dolfinx.read_mesh(self.domain.comm, ref_sol_path.with_suffix(".bp"), engine="BP4", ghost_mode=mesh.GhostMode.shared_facet)
                V_ref = fem.FunctionSpace(mesh_ref, (self.family, self.order))
                ref_sol = fem.Function(V_ref)
                for i in indeces:
                    adios4dolfinx.read_function(ref_sol, Path(self.output_folder) / Path(ref_file_name + "_" + str(i)).with_suffix(".bp"), engine="BP4")
                    ref_sol.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
                    uh[i].values.interpolate(
                        ref_sol,
                        nmm_interpolation_data=fem.create_nonmatching_meshes_interpolation_data(
                            uh[i].values.function_space.mesh._cpp_object, uh[i].values.function_space.element, ref_sol.function_space.mesh._cpp_object
                        ),
                    )
                    uh[i].ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
                return True
            else:
                return False
        else:
            raise Exception("read_reference_solution not implemented for DG")

    def define_fibers(self):
        if self.import_fibers:
            fixed_fibers_path = self.mesh_fibers_folder / Path("fibers_fixed_f0")
            if fixed_fibers_path.is_dir():
                self.f0 = self.import_fixed_fiber(self.mesh_fibers_folder / Path("fibers_fixed_f0"))
                self.s0 = self.import_fixed_fiber(self.mesh_fibers_folder / Path("fibers_fixed_s0"))
                self.n0 = self.import_fixed_fiber(self.mesh_fibers_folder / Path("fibers_fixed_n0"))
            else:
                self.f0 = self.read_mesh_data(self.mesh_fibers_folder / Path("fibers.h5"), self.domain, "f0")
                self.s0 = self.read_mesh_data(self.mesh_fibers_folder / Path("fibers.h5"), self.domain, "s0")
                self.n0 = self.read_mesh_data(self.mesh_fibers_folder / Path("fibers.h5"), self.domain, "n0")
        else:
            e1 = np.array([1.0, 0.0, 0.0])
            e2 = np.array([0.0, 1.0, 0.0])
            e3 = np.array([0.0, 0.0, 1.0])
            self.f0 = fem.Constant(self.domain, e1[: self.dim])
            self.s0 = fem.Constant(self.domain, e2[: self.dim])
            self.n0 = fem.Constant(self.domain, e3[: self.dim])

    def eval_on_points(self, u):
        if "cuboid" in self.domain_name:
            # eval only on u[0]
            u_val_loc = u[0].values.eval(self.points_on_proc, self.cells)
            u_val_loc = np.reshape(u_val_loc, (self.points_on_proc.shape[0], 1))
            data = self.loc_glob_map, u_val_loc
            data = self.domain.comm.gather(data, root=0)
            if self.domain.comm.rank == 0:
                u_val = np.zeros((self.eval_points.shape[0], 1))
                for i in range(self.domain.comm.size):
                    loc_glob_map, u_val_loc = data[i]
                    for j, u in zip(loc_glob_map, u_val_loc):
                        u_val[j] = u
            else:
                assert data is None
                u_val = np.zeros((self.eval_points.shape[0], 1))

            self.domain.comm.Bcast(u_val, root=0)

            return u_val
        else:
            return None

    def stim_ufl_expr(self):
        if not hasattr(self, 'istim_dur'):
            self.istim_dur = -1.0
        stim_dur = self.istim_dur if self.istim_dur >= 0.0 else 2.0
        self.stim_intensity = 35.7143  # in uA/cm^2, it is converted later in uA/mm^2 using self.scale_Iion. This is equivalent to the value used in Niederer et al.

        if 'cuboid' in self.domain_name:
            stim_centers = [[0.0, 0.0, 0.0]]
            if "small" in self.domain_name:
                stim_radius = 0.5
            else:
                stim_radius = 1.5
            self.stim_protocol = [[0.0, stim_dur]]  # list of stim_time, sitm_dur values
        else:
            if self.domain_name == "truncated_ellipsoid":
                stim_centers = [[0.0, 25.689, 46.1041]]
                stim_radius = 5
                self.stim_protocol = [[0.0, stim_dur]]
            elif self.domain_name == "03_fastl_LA":
                stim_centers = [[51.8816, 35.1259, 21.8025]]  # [[46.7522, 35.9794, 19.7214], [42.8294, 14.5558, 46.8198]]
                stim_radius = 2.0
                stim_distances = [280.0, 170.0, 160.0, 155.0, 150.0, 145.0, 140.0, 135.0, 130.0, 126.0, 124.0, 124.0, 124.0, 124.0]
                self.stim_intensity = 35.7143
                self.stim_protocol = [[0.0, stim_dur]]
                for stim_distance in stim_distances:
                    self.stim_protocol.append([self.stim_protocol[-1][0] + stim_distance, stim_dur])
            elif self.domain_name == "01_strocchi_LV":
                stim_centers = [[45.6131, 89.6654, 405.38]]
                stim_radius = 2.0
                self.stim_protocol = [[0.0, stim_dur]]
            else:
                raise Exception("Define stimulus variables for this domain.")

        if self.dim == 1:
            dists_stim_centers = []
            for stim_center in stim_centers:
                dists_stim_centers.append(abs(self.x[0] - stim_center[0]))
        elif self.dim == 2:
            dists_stim_centers = []
            for stim_center in stim_centers:
                dists_stim_centers.append(ufl.max_value(abs(self.x[0] - stim_center[0]), abs(self.x[1] - stim_center[1])))
        else:
            dists_stim_centers = []
            for stim_center in stim_centers:
                dists_stim_centers.append(
                    ufl.max_value(
                        ufl.max_value(abs(self.x[0] - stim_center[0]), abs(self.x[1] - stim_center[1])),
                        abs(self.x[2] - stim_center[2]),
                    )
                )

        space_conds = []
        for dist_stim_center in dists_stim_centers:
            space_conds.append(ufl.lt(dist_stim_center, stim_radius))
        space_conds_or = space_conds[0]
        for i in range(1, len(space_conds)):
            space_conds_or = ufl.Or(space_conds_or, space_conds[i])

        time_conds = []
        for stim_time, stim_dur in self.stim_protocol:
            time_conds.append(ufl.And(ufl.gt(self.t, stim_time), ufl.lt(self.t, stim_time + stim_dur)))
        time_conds_or = time_conds[0]
        for i in range(1, len(time_conds)):
            time_conds_or = ufl.Or(time_conds_or, time_conds[i])

        return ufl.conditional(ufl.And(time_conds_or, space_conds_or), self.stim_intensity, self.zero_fun)

    def read_fibrosis(self):
        mesh_fib = adios4dolfinx.read_mesh(self.domain.comm, self.mesh_fibers_folder / Path("fibrosis.bp"), engine="BP4", ghost_mode=mesh.GhostMode.shared_facet)
        V_fib = fem.FunctionSpace(mesh_fib, ("CG", 1))
        fibrosis = fem.Function(V_fib)
        adios4dolfinx.read_function(fibrosis, self.mesh_fibers_folder / Path("fibrosis.bp"), engine="BP4")
        fibrosis_i = fem.Function(self.V)
        fibrosis_i.interpolate(
            fibrosis,
            nmm_interpolation_data=fem.create_nonmatching_meshes_interpolation_data(
                fibrosis_i.function_space.mesh._cpp_object,
                fibrosis_i.function_space.element,
                fibrosis.function_space.mesh._cpp_object,
            ),
        )

        with open(self.mesh_fibers_folder / Path("quantiles.json"), "r") as f:
            data = json.load(f)
        quantiles = data["quantiles"]
        percentages = data["perc"]

        return fibrosis_i, percentages, quantiles

    def compute_errors(self, uh, ref_sol):
        # Compute L2 error and error at nodes
        uh.ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
        ref_sol.ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
        ref_sol_on_V = fem.Function(self.V)
        ref_sol_on_V.interpolate(
            ref_sol.values,
            nmm_interpolation_data=fem.create_nonmatching_meshes_interpolation_data(
                ref_sol_on_V.function_space.mesh._cpp_object, ref_sol_on_V.function_space.element, ref_sol.values.function_space.mesh._cpp_object
            ),
        )
        ref_sol_on_V.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        error_L2 = np.sqrt(self.domain.comm.allreduce(fem.assemble_scalar(fem.form((uh.values - ref_sol_on_V) ** 2 * ufl.dx)), op=MPI.SUM))
        sol_norm_L2 = np.sqrt(self.domain.comm.allreduce(fem.assemble_scalar(fem.form(ref_sol_on_V**2 * ufl.dx)), op=MPI.SUM))
        rel_error_L2 = error_L2 / sol_norm_L2

        return error_L2, rel_error_L2

    def read_mesh_data(self, file_path: Path, mesh: mesh.Mesh, data_path: str):
        # see https://fenicsproject.discourse.group/t/i-o-from-xdmf-hdf5-files-in-dolfin-x/3122/48
        assert file_path.is_file(), f"File {file_path} does not exist"
        infile = h5py.File(file_path, "r", driver="mpio", comm=mesh.comm)
        num_nodes_global = mesh.geometry.index_map().size_global
        assert data_path in infile.keys(), f"Data {data_path} does not exist"
        dataset = infile[data_path]
        shape = dataset.shape
        assert shape[0] == num_nodes_global, f"Got data of shape {shape}, expected {num_nodes_global, shape[1]}"
        # Read data locally on each process
        local_input_range = adios4dolfinx.utils.compute_local_range(mesh.comm, num_nodes_global)
        local_input_data = dataset[local_input_range[0] : local_input_range[1]]

        # Create appropriate function space (based on coordinate map)
        assert len(mesh.geometry.cmaps) == 1, "Mixed cell-type meshes not supported"
        element = basix.ufl.element(
            basix.ElementFamily.P,
            mesh.topology.cell_name(),
            mesh.geometry.cmaps[0].degree,
            mesh.geometry.cmaps[0].variant,
            shape=(shape[1],),
            gdim=mesh.geometry.dim,
        )

        # Assumption: Same doflayout for geometry and function space, cannot test in python
        V = fem.FunctionSpace(mesh, element)
        uh = fem.Function(V, name=data_path)
        # Assume that mesh is first order for now
        assert mesh.geometry.cmaps[0].degree == 1, "Only linear meshes supported"
        x_dofmap = mesh.geometry.dofmap
        igi = np.array(mesh.geometry.input_global_indices, dtype=np.int64)
        global_geom_input = igi[x_dofmap]
        global_geom_owner = adios4dolfinx.utils.index_owner(mesh.comm, global_geom_input.reshape(-1), num_nodes_global)
        for i in range(shape[1]):
            arr_i = adios4dolfinx.comm_helpers.send_dofs_and_recv_values(
                global_geom_input.reshape(-1),
                global_geom_owner,
                mesh.comm,
                local_input_data[:, i],
                local_input_range[0],
            )
            dof_pos = x_dofmap.reshape(-1) * shape[1] + i
            uh.x.array[dof_pos] = arr_i
        infile.close()
        return uh

    def import_fixed_fiber(self, input_folder):
        self.V_fiber = fem.VectorFunctionSpace(self.domain, ("CG", 1))
        domain_r = adios4dolfinx.read_mesh(self.domain.comm, input_folder, "BP4", mesh.GhostMode.shared_facet)
        el = ufl.VectorElement("CG", self.domain.ufl_cell(), 1)
        V_r = fem.FunctionSpace(domain_r, el)
        fib_r = fem.Function(V_r)
        adios4dolfinx.read_function(fib_r, input_folder, "BP4")
        fib = fem.Function(self.V_fiber)
        fib.interpolate(
            fib_r,
            nmm_interpolation_data=fem.create_nonmatching_meshes_interpolation_data(
                fib.function_space.mesh._cpp_object,
                fib.function_space.element,
                fib_r.function_space.mesh._cpp_object,
            ),
        )
        fib.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        return fib

    def get_dofs_stats(self):
        data = (self.tmp1.n_loc_dofs + self.tmp1.n_ghost_dofs, self.tmp1.n_loc_dofs, self.tmp1.n_ghost_dofs, self.tmp1.n_ghost_dofs / (self.tmp1.n_loc_dofs + self.tmp1.n_loc_dofs))
        data = self.comm.gather(data, root=0)
        data = self.comm.bcast(data, root=0)
        avg = [0.0, 0.0, 0.0, 0]
        n = len(data)
        for d in data:
            for i in range(4):
                avg[i] += d[i] / n
        return data, avg
