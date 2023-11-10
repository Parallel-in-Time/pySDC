import numpy as np
import dolfinx
from dolfinx import fem, mesh, io
from mpi4py import MPI
from petsc4py import PETSc
import ufl
import adios4dolfinx
import basix
import h5py
from pathlib import Path
import json
import logging


class Monodomain:
    def __init__(self, **problem_params):
        self.logger = logging.getLogger("step")

        self.domain_name = problem_params["domain_name"]
        self.mesh_refinement_str = "ref_" + str(problem_params["refinements"])
        self.mesh_fibers_folder = Path(problem_params["meshes_fibers_root_folder"]) / Path(self.domain_name) / Path(self.mesh_refinement_str)
        self.fibrosis = problem_params["fibrosis"]

        if "cuboid" in problem_params["domain_name"]:
            if "small" in problem_params["domain_name"]:
                self.dom_size = [[0.0, 0.0, 0.0], [5.0, 3.0, 1.0]]
            else:
                self.dom_size = [[0.0, 0.0, 0.0], [20.0, 7.0, 3.0]]
            self.import_fibers = False
        else:
            self.import_fibers = True

        self.comm = problem_params["communicator"]
        self.define_domain()
        self.define_eval_points()

        self.t0 = 0.0
        self.Tend = self.get_tend()
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

        self.scale_Iion = 0.01  # used to convert currents in uA/cm^2 to uA/mm^2

        self.scale_im = self.scale_Iion / self.Cm
        self.ionic_model_eval = problem_params["ionic_model_eval"]
        self.ionic_model = self.define_ionic_model(problem_params["ionic_model"], self.ionic_model_eval, self.scale_im)
        self.size = self.ionic_model.size
        self.ionic_model_name = problem_params["ionic_model"]
        self.istim_dur = problem_params["istim_dur"] if "istim_dur" in problem_params else -1.0

        self.problem_params = problem_params

    def get_tend(self):
        if "cuboid" in self.domain_name:
            if "small" in self.domain_name:
                return 0.05
            else:
                return 25.0
        else:
            return 50.0  # for the initial value simulate till 800 ms, then for 50 ms

    def define_eval_points(self):
        # used to compute CV, valid only on cuboid domain
        if "cuboid" in self.domain_name:
            if "small" in self.domain_name:
                n_pts = 5
                a = np.array([[0.5, 0.5, 0.5]])
            else:
                n_pts = 10
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

    def define_ionic_model(self, model_name, ionic_model_eval, scale):
        if ionic_model_eval == "ufl":
            import pySDC.projects.Monodomain.problem_classes.monodomain_system_helpers.ionicmodels.ufl as ionicmodels
        elif ionic_model_eval == "c++":
            import pySDC.projects.Monodomain.problem_classes.monodomain_system_helpers.ionicmodels.cpp as ionicmodels
        else:
            raise Exception("Unknown ionic model evaluation language. User either 'c++' (preferred) or 'ufl'")

        if model_name == "HodgkinHuxley" or model_name == "HH":
            return ionicmodels.HodgkinHuxley(scale)
        elif model_name == "Courtemanche1998" or model_name == "CRN":
            return ionicmodels.Courtemanche1998(scale)
        elif model_name == "TenTusscher2006_epi" or model_name == "TTP":
            return ionicmodels.TenTusscher2006_epi(scale)
        else:
            raise Exception("Unknown ionic model.")

    def define_domain(self):
        if self.import_fibers:
            with io.XDMFFile(self.comm, self.mesh_fibers_folder / Path("fibers.xdmf"), "r") as xdmf:
                self.domain = xdmf.read_mesh(name="mesh", xpath="Xdmf/Domain/Grid")
        else:
            with io.XDMFFile(self.comm, self.mesh_fibers_folder / Path("fibers.xdmf"), "r") as xdmf:
                self.domain = xdmf.read_mesh()
        self.dim = self.domain.geometry.dim

    def define_domain_dependent_variables(self, domain, V, dtype_u):
        self.domain = domain
        self.V = V
        self.x = ufl.SpatialCoordinate(self.domain)
        self.n = ufl.FacetNormal(self.domain)
        self.t = fem.Constant(self.domain, 0.0)
        self.dt = fem.Constant(self.domain, 0.0)
        self.dtype_u = dtype_u
        self.uh = self.dtype_u(init=self.V, val=0.0)
        self.zero_fun = fem.Constant(self.domain, 0.0)

        self.define_fibers()
        if self.fibrosis:
            fibrosis_r_field, percentages, quantiles = self.read_fibrosis()
            quantile = quantiles[1]  # 50 %
            sigma_t_fibrotic_expr = self.sigma_t * ufl.conditional(ufl.lt(fibrosis_r_field, quantile), 1.0, 0.0)
            self.diff_t = fem.Function(self.V)
            self.diff_t.interpolate(
                fem.Expression(
                    sigma_t_fibrotic_expr / self.chi / self.Cm,
                    self.V.element.interpolation_points(),
                )
            )

        self.stim = self.Istim()
        if abs(self.scale_im - 1.0) > 1e-10:
            self.stim *= self.scale_im

        # initial value
        self.u0 = [fem.Constant(self.domain, y0) for y0 in self.ionic_model.initial_values()]

        self.define_splittings()

    def define_splittings(self):
        # Here we define different splittings of the rhs into stiff, nonstiff and exponential terms

        self.im_f_nonstiff = dict()
        self.im_nonstiff_args = dict()
        self.im_nonstiff_indeces = dict()
        self.im_f_stiff = dict()
        self.im_stiff_args = dict()
        self.im_stiff_indeces = dict()
        self.im_lmbda_exp = dict()
        self.im_lmbda_yinf_exp = dict()
        self.im_exp_args = dict()
        self.im_exp_indeces = dict()

        if self.ionic_model_eval == "ufl":
            self.ionic_model.set_domain(self.domain)
            self.ionic_model.set_y(self.uh)
            self.ionic_model.set_dt(self.dt)
            self.ionic_model.set_V(self.V)

        # WITHOUT SPLITTING
        self.im_f = self.ionic_model.f

        # SPLITTING stiff_nonstiff
        # this is a splitting to be used in multirate explicit stabilized methods. W euse it for the mES schemes.
        # define nonstiff
        self.im_f_nonstiff["stiff_nonstiff"] = self.ionic_model.f_nonstiff
        self.im_nonstiff_args["stiff_nonstiff"] = self.ionic_model.f_nonstiff_args
        self.im_nonstiff_indeces["stiff_nonstiff"] = self.ionic_model.f_nonstiff_indeces
        # define stiff
        self.im_f_stiff["stiff_nonstiff"] = self.ionic_model.f_stiff
        self.im_stiff_args["stiff_nonstiff"] = self.ionic_model.f_stiff_args
        self.im_stiff_indeces["stiff_nonstiff"] = self.ionic_model.f_stiff_indeces
        # define exp
        self.im_lmbda_exp["stiff_nonstiff"] = None
        self.im_lmbda_yinf_exp["stiff_nonstiff"] = None
        self.im_exp_args["stiff_nonstiff"] = []
        self.im_exp_indeces["stiff_nonstiff"] = []

        # SPLITTING exp_nonstiff
        # this is the standard splitting used in Rush-Larsen methods. We use it for the IMEXEXP (IMEX+RL) and exp_mES schemes.
        # define nonstiff.
        self.im_f_nonstiff["exp_nonstiff"] = self.ionic_model.f_expl
        self.im_nonstiff_args["exp_nonstiff"] = self.ionic_model.f_expl_args
        self.im_nonstiff_indeces["exp_nonstiff"] = self.ionic_model.f_expl_indeces
        # define stiff
        self.im_f_stiff["exp_nonstiff"] = None  # no stiff part coming from ionic model
        self.im_stiff_args["exp_nonstiff"] = []
        self.im_stiff_indeces["exp_nonstiff"] = []
        # define exp
        self.im_lmbda_exp["exp_nonstiff"] = self.ionic_model.lmbda_exp
        self.im_lmbda_yinf_exp["exp_nonstiff"] = self.ionic_model.lmbda_yinf_exp
        self.im_exp_args["exp_nonstiff"] = self.ionic_model.f_exp_args
        self.im_exp_indeces["exp_nonstiff"] = self.ionic_model.f_exp_indeces

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

    def update_time(self, t):
        self.t.value = t

    def update_dt(self, dt):
        self.dt.value = dt

    @property
    def stim_expr(self):
        return self.stim

    @property
    def u0_expr(self):
        return self.u0

    def Istim(self):
        self.stim_intensity = 35.7143  # in uA/cm^2, it is converted later in uA/mm^2 using self.scale_Iion. This is equivalent to the value used in Niederer et al.
        stim_dur = self.istim_dur if self.istim_dur >= 0.0 else 2.0

        if not self.import_fibers:
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

    def rho_nonstiff(self, y, t, fy=None):
        if self.ionic_model_name == "HodgkinHuxley" or self.ionic_model_name == "HH":
            return 40.0
        elif self.ionic_model_name == "Courtemanche1998" or self.ionic_model_name == "CRN":
            return 7.5
        elif self.ionic_model_name == "TenTusscher2006_epi" or self.ionic_model_name == "TTP":
            return 6.5
        else:
            raise Exception("unknown rho_nonstiff for this ionic model. Compute it!")

    def read_fibrosis(self):
        mesh_fib = adios4dolfinx.read_mesh(
            self.domain.comm,
            self.mesh_fibers_folder / Path("fibrosis.bp"),
            engine="BP4",
            ghost_mode=mesh.GhostMode.shared_facet,
        )
        V_fib = fem.FunctionSpace(mesh_fib, ("CG", 1))
        fibrosis = dolfinx.fem.Function(V_fib)
        adios4dolfinx.read_function(fibrosis, self.mesh_fibers_folder / Path("fibrosis.bp"), engine="BP4")
        fibrosis_i = dolfinx.fem.Function(self.V)
        fibrosis_i.interpolate(
            fibrosis,
            nmm_interpolation_data=dolfinx.fem.create_nonmatching_meshes_interpolation_data(
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

    def read_mesh_data(self, file_path: Path, mesh: dolfinx.mesh.Mesh, data_path: str):
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
        V = dolfinx.fem.FunctionSpace(mesh, element)
        uh = dolfinx.fem.Function(V, name=data_path)
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
        domain_r = adios4dolfinx.read_mesh(self.domain.comm, input_folder, "BP4", dolfinx.mesh.GhostMode.shared_facet)
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
