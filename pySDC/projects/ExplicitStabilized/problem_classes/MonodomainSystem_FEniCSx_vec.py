import numpy as np
from dolfinx import mesh, fem, io, geometry
import ufl
import adios4dolfinx
from mpi4py import MPI
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType
from pathlib import Path
import logging


from pySDC.projects.ExplicitStabilized.datatype_classes.fenicsx_mesh import fenicsx_mesh
from pySDC.projects.ExplicitStabilized.datatype_classes.fenicsx_mesh_vec import fenicsx_mesh_vec, rhs_fenicsx_mesh_vec, exp_rhs_fenicsx_mesh_vec
from pySDC.core.Problem import ptype
from pySDC.core.Errors import ParameterError
from pySDC.projects.ExplicitStabilized.problem_classes.monodomain_system_helpers.monodomain import Monodomain


class monodomain_system(ptype):
    dtype_u_vec = fenicsx_mesh_vec
    dtype_f_vec = fenicsx_mesh_vec

    def __init__(self, **problem_params):
        # # these parameters will be used later, so assert their existence
        # essential_keys = ['family', 'order','enable_output']
        # for key in essential_keys:
        #     if key not in problem_params:
        #         msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
        #         raise ParameterError(msg)
        # if problem_params['enable_output']:
        #     essential_keys = ['output_root','output_file_name']
        #     for key in essential_keys:
        #         if key not in problem_params:
        #             msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
        #             raise ParameterError(msg)

        # if problem_params["dim"]==2 and problem_params["domain_name"]!="cuboid":
        #     raise Exception('In dim==2 the only possible domain_name is "cuboid"')

        self.logger = logging.getLogger("step")

        self.exact = Monodomain(**problem_params)
        self.t0 = self.exact.t0  # Start time
        self.Tend = self.exact.Tend  # End time
        self.dim = self.exact.dim

        self.domain = self.exact.domain

        def dtype_u_vec_fixed(init, val=0.0):
            return self.dtype_u_vec(init, val, self.exact.size)

        def dtype_f_vec_fixed(init, val=0.0):
            return self.dtype_f_vec(init, val, self.exact.size)

        self.dtype_u = dtype_u_vec_fixed
        self.dtype_f = dtype_f_vec_fixed

        self.V = fem.FunctionSpace(self.domain, (problem_params["family"], problem_params["order"]))
        self.exact.define_domain_dependent_variables(self.domain, self.V, self.dtype_u)

        self.set_solver_options(problem_params)

        # invoke super init
        super(monodomain_system, self).__init__(self.V)
        self._makeAttributeAndRegister(*problem_params.keys(), localVars=problem_params, readOnly=True)

        if self.mass_lumping and (self.family != "CG" or self.order > 1):
            raise Exception("You have specified mass_lumping=True but for order>1 or family!='CG'.")

        self.define_variational_forms()
        self.assemble_vec_mat()
        self.define_mass_solver()

        # save in xdmf format
        self.output_folder = self.output_root
        if not self.output_root.endswith("/"):
            self.output_folder = self.output_folder + "/"

        self.mesh_refinement_str = "ref_" + str(self.refinements)
        self.output_folder = self.output_folder + self.domain_name + "/" + self.mesh_refinement_str + "/" + self.ionic_model + "/"

        if self.enable_output:
            self.xdmf = io.XDMFFile(self.domain.comm, self.output_folder + self.output_file_name + ".xdmf", "w")
            self.xdmf.write_mesh(self.domain)

    def __del__(self):
        if self.enable_output:
            self.xdmf.close()

    def set_solver_options(self, params):
        # we suppose that the problem is symmetric
        # first set the default options
        if self.dim <= 2:
            def_solver_ksp = "preonly"
            def_solver_pc = "cholesky"
        else:
            def_solver_ksp = "cg"
            def_solver_pc = "hypre"

        if "solver_ksp" not in params:
            params["solver_ksp"] = def_solver_ksp
        if "solver_pc" not in params:
            params["solver_pc"] = def_solver_pc

    def define_variational_forms(self):
        u, v = ufl.TrialFunction(self.V), ufl.TestFunction(self.V)

        self.mass = u * v * ufl.dx

        def diff_tens(w):
            return (
                self.exact.diff_l * self.exact.f0 * ufl.dot(self.exact.f0, w)
                + self.exact.diff_t * self.exact.s0 * ufl.dot(self.exact.s0, w)
                + self.exact.diff_t * self.exact.n0 * ufl.dot(self.exact.n0, w)
            )

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

        self.interp_f = fenicsx_mesh(self.init, 0.0)
        from dolfinx.fem.petsc import create_vector

        self.b = fem.petsc.create_vector(fem.form(v * ufl.dx))

        self.stim_expr = fem.Expression(self.exact.stim_expr, self.V.element.interpolation_points())
        self.im_f_expr = self.exact.im_f

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

    def define_mass_solver(self):
        self.solver_M = PETSc.KSP().create(self.domain.comm)
        self.solver_M.setOperators(self.M)
        self.solver_M.setType(self.solver_ksp)
        self.solver_M.getPC().setType(self.solver_pc)

    def invert_mass_matrix(self, x, y):
        # solves y = M*y
        if self.mass_lumping:
            y.pointwiseDivide(x, self.ml)
        else:
            self.solver_M.solve(x, y)

    def write_solution(self, uh, t):
        if self.family == "DG" and not hasattr(self, "V_CG"):
            self.V_CG = fem.FunctionSpace(self.domain, ("CG", self.order))
            self.u_CG = fem.Function(self.V_CG)

        if self.enable_output and self.output_V_only:
            uh.sub(0).name = "V"
            if self.family == "CG":
                self.xdmf.write_function(uh.sub(0), t)
            else:
                uh.ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD, False)
                self.u_CG.interpolate(
                    uh.sub(0),
                    nmm_interpolation_data=fem.create_nonmatching_meshes_interpolation_data(
                        self.u_CG.function_space.mesh._cpp_object, self.u_CG.function_space.element, uh.sub(0).function_space.mesh._cpp_object
                    ),
                )
                self.xdmf.write_function(self.u_CG, t)
        elif self.enable_output:
            for i in range(self.exact.size):
                uh.sub(i).name = f"u_{i+1}"
                self.xdmf.write_function(uh.sub(i), t)

    def write_reference_solution(self, uh):
        uh.ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD, False)
        adios4dolfinx.write_mesh(self.domain, self.output_folder + self.output_file_name + ".bp", engine="BP4")
        if self.family == "CG":
            adios4dolfinx.write_function(uh.sub(0), self.output_folder + self.output_file_name + ".bp", engine="BP4")
        else:
            if not hasattr(self, "V_CG"):
                self.V_CG = fem.FunctionSpace(self.domain, ("CG", self.order))
                self.u_CG = fem.Function(self.V_CG)
            self.u_CG.interpolate(
                uh.sub(0),
                nmm_interpolation_data=fem.create_nonmatching_meshes_interpolation_data(
                    self.u_CG.function_space.mesh._cpp_object, self.u_CG.function_space.element, uh.sub(0).function_space.mesh._cpp_object
                ),
            )
            self.u_CG.vector.ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
            adios4dolfinx.write_function(self.u_CG, self.output_folder + self.output_file_name + ".bp", engine="BP4")

    def write_full_solution(self, uh):
        uh.ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD, True)
        adios4dolfinx.write_mesh(self.domain, self.output_folder + self.output_file_name + ".bp", engine="BP4")
        if self.family == "CG":
            for i in range(self.exact.size):
                adios4dolfinx.write_function(uh.sub(i), self.output_folder + self.output_file_name + "_" + str(i) + ".bp", engine="BP4")
        else:
            raise Exception("write full sol not implemented for DG")

    def read_full_solution(self, uh):
        if self.family == "CG":
            init_val_path = Path(self.output_folder) / Path(self.init_val_name)
            mesh_ref = adios4dolfinx.read_mesh(self.domain.comm, init_val_path.with_suffix(".bp"), engine="BP4", ghost_mode=mesh.GhostMode.shared_facet)
            V_ref = fem.FunctionSpace(mesh_ref, ("CG", self.order))
            sol_ref = fem.Function(V_ref)
            for i in range(uh.size):
                adios4dolfinx.read_function(sol_ref, str(init_val_path) + ("_" + str(i) + ".bp"), engine="BP4")
                sol_ref.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
                uh.sub(i).interpolate(
                    sol_ref,
                    nmm_interpolation_data=fem.create_nonmatching_meshes_interpolation_data(
                        uh.sub(i).function_space.mesh._cpp_object, uh.sub(i).function_space.element, sol_ref.function_space.mesh._cpp_object
                    ),
                )
            uh.ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD, True)
        else:
            raise Exception("read full sol not implemented for DG")

        return uh

    def initial_value(self):
        u0 = self.dtype_u(self.init, val=0.0)
        for i in range(self.exact.size):
            u0.sub(i).name = f"u_{i+1}"

        if not self.read_initial_value:
            for i in range(self.exact.size):
                u0.sub(i).interpolate(fem.Expression(self.exact.u0_expr[i], self.V.element.interpolation_points()))
        else:
            self.read_full_solution(u0)

        # with io.XDMFFile(self.domain.comm, self.output_folder + self.output_file_name + "init_val.xdmf", "w") as xdmf:
        #     xdmf.write_mesh(self.domain)
        #     xdmf.write_function(u0.sub(0))

        return u0

    def compute_errors(self, uh):
        # Compute L2 error and error at nodes
        ref_sol_path = Path(self.output_folder) / Path(self.ref_sol).with_suffix(".bp")
        if ref_sol_path.is_dir():
            uh.ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD, False)
            mesh_ref = adios4dolfinx.read_mesh(self.domain.comm, ref_sol_path, engine="BP4", ghost_mode=mesh.GhostMode.shared_facet)
            V_ref = fem.FunctionSpace(mesh_ref, ("CG", self.order))
            sol_ref = fem.Function(V_ref)
            adios4dolfinx.read_function(sol_ref, ref_sol_path, engine="BP4")
            sol_ref.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            uh_on_ref_space = fem.Function(V_ref)
            uh_on_ref_space.interpolate(
                uh.sub(0),
                nmm_interpolation_data=fem.create_nonmatching_meshes_interpolation_data(
                    uh_on_ref_space.function_space.mesh._cpp_object, uh_on_ref_space.function_space.element, uh.sub(0).function_space.mesh._cpp_object
                ),
            )
            uh_on_ref_space.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            error_L2 = np.sqrt(self.domain.comm.allreduce(fem.assemble_scalar(fem.form((uh_on_ref_space - sol_ref) ** 2 * ufl.dx)), op=MPI.SUM))
            sol_norm_L2 = np.sqrt(self.domain.comm.allreduce(fem.assemble_scalar(fem.form(sol_ref**2 * ufl.dx)), op=MPI.SUM))
            rel_error_L2 = error_L2 / sol_norm_L2

            if self.domain.comm.rank == 0:
                print(f"L2-errors: {error_L2}")
                print(f"Relative L2-errors: {rel_error_L2}")

            return True, error_L2, rel_error_L2
        else:
            return False, 0.0, 0.0

    def get_size(self):
        return self.uD[0].vector.getSize()

    def eval_on_points(self, u):
        if self.exact.domain_name == "cuboid_2D" or self.exact.domain_name == "cuboid_3D":
            if not hasattr(self, "points_on_proc"):
                bb_tree = geometry.bb_tree(self.domain, self.domain.topology.dim)
                self.cells = []
                self.points_on_proc = []
                # Find cells whose bounding-box collide with the the points
                cell_candidates = geometry.compute_collisions_points(bb_tree, self.exact.eval_points)
                # Choose one of the cells that contains the point
                colliding_cells = geometry.compute_colliding_cells(self.domain, cell_candidates, self.exact.eval_points)
                for i, point in enumerate(self.exact.eval_points):
                    if len(colliding_cells.links(i)) > 0:
                        self.points_on_proc.append(point)
                        self.cells.append(np.min(colliding_cells.links(i)[0]))
                self.points_on_proc = np.array(self.points_on_proc, dtype=np.float64)
                self.loc_glob_map = []
                for i in range(self.points_on_proc.shape[0]):
                    p = self.points_on_proc[i, :]
                    for j in range(self.exact.eval_points.shape[0]):
                        q = self.exact.eval_points[j, :]
                        if np.linalg.norm(p - q) < np.max([np.linalg.norm(p), np.linalg.norm(q)]) * 1e-5:
                            self.loc_glob_map.append(j)
                            break

            # eval only on u[0]
            u_val_loc = u.sub(0).eval(self.points_on_proc, self.cells)
            u_val_loc = np.reshape(u_val_loc, (self.points_on_proc.shape[0], 1))
            data = self.loc_glob_map, u_val_loc
            data = self.domain.comm.gather(data, root=0)
            if self.domain.comm.rank == 0:
                u_val = np.zeros((self.exact.eval_points.shape[0], 1))
                for i in range(self.domain.comm.size):
                    loc_glob_map, u_val_loc = data[i]
                    for j, u in zip(loc_glob_map, u_val_loc):
                        u_val[j] = u
            else:
                assert data == None
                u_val = np.zeros((self.exact.eval_points.shape[0], 1))

            self.domain.comm.Bcast(u_val, root=0)

            return u_val
        else:
            return None

    def eval_f(self, u, t, fh=None):
        """
        Evaluates F(u,t) = M^-1*( A*u + f(u,t) )

        Returns:
            dtype_u: solution as mesh
        """

        self.exact.update_time(t)

        if fh is None:
            fh = self.dtype_f(init=self.V, val=0.0)

        self.update_u_and_uh(u, False)

        # eval ionic model
        self.eval_expr(self.im_f_expr, u, fh, list(range(self.exact.size)))
        # apply stimulus
        self.interp_f.values.interpolate(self.stim_expr)
        fh.val_list[0] += self.interp_f

        # eval diffusion
        self.K.mult(u[0].values.vector, self.b)  # WARNING: DO NOT DO -u[0].values.vector HERE AND += self.interp_f BELOW SINCE IT CREATES MEMORY LEAK!!!
        self.invert_mass_matrix(self.b, self.interp_f.values.vector)
        fh.val_list[0] -= self.interp_f

        return fh

    def update_u_and_uh(self, u, ghost_update_all=True):
        u.ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD, ghost_update_all)
        if self.ionic_model_eval in ["c++", "numpy"]:
            pass
        elif self.ionic_model_eval == "ufl":
            self.exact.uh.copy(u)

    def eval_expr(self, expr, u, fh, indeces):
        if self.ionic_model_eval in ["c++", "numpy"] and expr is not None:
            expr(u.np_list, fh.np_list)
        elif self.ionic_model_eval == "ufl":
            for i in indeces:
                fh[i].values.interpolate(expr[i])

        non_indeces = [i for i in range(self.exact.size) if i not in indeces]
        for i in non_indeces:
            fh[i].zero()


class monodomain_system_imex(monodomain_system):
    dtype_u_vec = fenicsx_mesh_vec
    dtype_f_vec = rhs_fenicsx_mesh_vec

    def __init__(self, **problem_params):
        super(monodomain_system_imex, self).__init__(**problem_params)

        self.prev_factor = -1.0
        self.solver = PETSc.KSP().create(self.domain.comm)
        self.solver.setType(self.solver_ksp)
        self.solver.getPC().setType(self.solver_pc)

        if self.exact.dim > 2 and "solver_rtol" in problem_params and problem_params["solver_rtol"] != "default":
            assert type(problem_params["solver_rtol"]) is float, 'problem_params["solver_rtol"] must be a float or "default"'
            self.solver.setTolerances(rtol=problem_params["solver_rtol"])

    def solve_system(self, rhs, factor, u0, t, u_sol=None):
        self.exact.update_time(t)

        if u_sol is None:
            u_sol = self.dtype_u(self.V)

        if abs(factor - self.prev_factor) > 1e-8 * factor:
            self.prev_factor = factor
            self.solver.setOperators(self.M + factor * self.K)

        self.M.mult(rhs[0].values.vector, self.b)
        self.solver.solve(self.b, u_sol[0].values.vector)

        if rhs is not u_sol:
            for i in range(1, self.exact.size):
                u_sol.val_list[i].copy(rhs.val_list[i])

        return u_sol


class monodomain_system_exp_expl_impl(monodomain_system_imex):
    dtype_u_vec = fenicsx_mesh_vec
    dtype_f_vec = exp_rhs_fenicsx_mesh_vec

    def __init__(self, **problem_params):
        super(monodomain_system_exp_expl_impl, self).__init__(**problem_params)
        self.define_functions()

    def define_functions(self):
        splitting = self.splitting
        self.im_f_nonstiff = self.exact.im_f_nonstiff[splitting]
        self.im_nonstiff_args = self.exact.im_nonstiff_args[splitting]
        self.im_nonstiff_indeces = self.exact.im_nonstiff_indeces[splitting]
        self.im_f_stiff = self.exact.im_f_stiff[splitting]
        self.im_stiff_args = self.exact.im_stiff_args[splitting]
        self.im_stiff_indeces = self.exact.im_stiff_indeces[splitting]
        self.im_lmbda_exp = self.exact.im_lmbda_exp[splitting]
        self.im_lmbda_yinf_exp = self.exact.im_lmbda_yinf_exp[splitting]
        self.im_exp_args = self.exact.im_exp_args[splitting]
        self.im_exp_indeces = self.exact.im_exp_indeces[splitting]

        if self.ionic_model_eval == "ufl":
            self.ufl_im_lmbda_exp, self.ufl_im_yinf_exp = self.im_lmbda_yinf_exp

        self.one = self.dtype_u(init=self.V, val=0.0)
        for i in range(self.exact.size):
            self.one.sub(i).interpolate(fem.Expression(fem.Constant(self.domain, 1.0), self.V.element.interpolation_points()))

        self.im_non_exp_indeces = [i for i in range(self.exact.size) if i not in self.im_exp_indeces]
        self.rhs_stiff_args = self.im_stiff_args
        self.rhs_stiff_indeces = self.im_stiff_indeces
        if 0 not in self.rhs_stiff_args:
            self.rhs_stiff_args = [0] + self.rhs_stiff_args
        self.rhs_nonstiff_args = self.im_nonstiff_args
        if 0 not in self.rhs_nonstiff_args:
            self.rhs_nonstiff_args = [0] + self.rhs_nonstiff_args
        self.rhs_exp_args = self.im_exp_args
        self.rhs_exp_indeces = self.im_exp_indeces

        self.lmbda = self.dtype_u(init=self.V, val=0.0)
        self.yinf = self.dtype_u(init=self.V, val=0.0)

    def eval_lmbda_yinf_exp(self, u, lmbda, yinf):
        if self.ionic_model_eval == "c++":
            self.im_lmbda_yinf_exp(u.np_list, lmbda.np_list, yinf.np_list)
        else:
            for i in self.im_exp_indeces:
                lmbda[i].values.interpolate(self.ufl_im_lmbda_exp[i])
                yinf[i].values.interpolate(self.ufl_im_yinf_exp[i])

    def eval_lmbda_exp(self, u, lmbda):
        if self.ionic_model_eval == "c++":
            self.im_lmbda_exp(u.np_list, lmbda.np_list)
        else:
            for i in self.im_exp_indeces:
                lmbda[i].values.interpolate(self.ufl_im_lmbda_exp[i])

    def eval_f(self, u, t, eval_impl=True, eval_expl=True, eval_exp=True, fh=None):
        """
        Evaluates F(u,t) = M^-1*( A*u + f(u,t) )

        Returns:
            dtype_u: solution as mesh
        """

        self.exact.update_time(t)
        self.update_u_and_uh(u, False)

        if fh is None:
            fh = self.dtype_f(init=self.V, val=0.0)

        # evaluate explicit (non stiff) part M^-1*f_nonstiff(u,t)
        if eval_expl:
            fh.expl = self.eval_f_nonstiff(u, t, fh.expl)

        # evaluate implicit (stiff) part M^1*A*u+M^-1*f_stiff(u,t)
        if eval_impl:
            fh.impl = self.eval_f_stiff(u, t, fh.impl)

        # evaluate exponential part
        if eval_exp:
            fh.exp = self.eval_f_exp(u, t, fh.exp)

        return fh

    def eval_f_nonstiff(self, u, t, fh_nonstiff):
        # eval ionic model nonstiff terms
        self.eval_expr(self.im_f_nonstiff, u, fh_nonstiff, self.im_nonstiff_indeces)

        # apply stimulus
        self.interp_f.values.interpolate(self.stim_expr)
        fh_nonstiff.val_list[0] += self.interp_f

        return fh_nonstiff

    def eval_f_stiff(self, u, t, fh_stiff):
        # eval ionic model stiff terms
        self.eval_expr(self.im_f_stiff, u, fh_stiff, self.im_stiff_indeces)

        # apply diffusion
        self.K.mult(u[0].values.vector, self.b)  # WARNING: DO NOT DO -u[0].values.vector HERE AND += self.interp_f BELOW SINCE IT CREATES MEMORY LEAK!!!
        self.invert_mass_matrix(self.b, self.interp_f.values.vector)
        fh_stiff.val_list[0] -= self.interp_f

        return fh_stiff

    def eval_f_exp(self, u, t, fh_exp):
        self.update_u_and_uh(u, False)
        self.eval_lmbda_yinf_exp(u, self.lmbda, self.yinf)
        for i in self.im_exp_indeces:
            fh_exp.np_list[i][:] = self.lmbda.np_list[i] * (u.np_list[i] - self.yinf.np_list[i])

        fh_exp.zero_sub(self.im_non_exp_indeces)

        return fh_exp

    def eval_phi_f_exp(self, u, factor, t, u_sol=None):
        self.update_u_and_uh(u, False)

        if u_sol is None:
            u_sol = self.dtype_u(init=self.V, val=0.0)

        self.eval_lmbda_yinf_exp(u, self.lmbda, self.yinf)
        for i in self.im_exp_indeces:
            u_sol.np_list[i][:] = (np.exp(factor * self.lmbda.np_list[i]) - 1.0) / (factor) * (u.np_list[i] - self.yinf.np_list[i])

        u_sol.zero_sub(self.im_non_exp_indeces)

        return u_sol

    def phi_eval(self, u, factor, t, k, u_sol=None):
        self.update_u_and_uh(u, False)

        if u_sol is None:
            u_sol = self.dtype_u(init=self.V, val=0.0)

        self.eval_lmbda_exp(u, self.lmbda)
        self.lmbda *= factor
        for i in self.im_exp_indeces:  # phi_0
            u_sol.np_list[i][:] = np.exp(self.lmbda.np_list[i])
        k_fac = 1  # 0!
        for j in range(1, k + 1):  # phi_j, j=1,...,k
            for i in self.im_exp_indeces:
                u_sol.np_list[i][:] = (u_sol.np_list[i] - 1.0 / k_fac) / self.lmbda.np_list[i]
            k_fac = k_fac * j

        u_sol.copy_sub(self.one, self.im_non_exp_indeces)
        if k > 1.0:
            u_sol.imul_sub(1 / k_fac, self.im_non_exp_indeces)

        return u_sol

    def lmbda_eval(self, u, t, lmbda=None):
        self.exact.update_time(t)

        self.update_u_and_uh(u, False)

        if lmbda is None:
            lmbda = self.dtype_u(init=self.V, val=0.0)

        self.eval_lmbda_exp(u, lmbda)

        lmbda.zero_sub(self.im_non_exp_indeces)

        return lmbda
