import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
import basix
import dolfinx as dfx
import dolfinx.fem.petsc
import dolfinx_mpc

from pySDC.core.problem import Problem
from pySDC.implementations.datatype_classes.mesh import mesh


class fenicsx_NSE_mass(Problem):
    r"""
    Monolithic incompressible Navier–Stokes problem in 2D using FEniCSx.

    This class implements a finite element discretization of the incompressible
    Navier–Stokes equations in a square domain using a monolithic formulation
    for velocity and pressure.

    Governing equations:
    .. math::
        \frac{\partial \mathbf{u}}{\partial t}
        + (\mathbf{u} \cdot \nabla)\mathbf{u}
        - \nu \Delta \mathbf{u}
        + \nabla p = f,
        \quad \nabla \cdot \mathbf{u} = 0

    Domain:
        A square domain :math:`\Omega = [-0.5, 0.5]^2` discretized with triangular elements.

    Spatial discretization:
        - Continuous Lagrange elements for velocity and pressure
        - Mixed finite element space :math:`W = V \times Q`
        - Monolithic formulation for the coupled velocity–pressure system

    Manufactured solution:
        A smooth, analytical, time-dependent solution is given by:
        ..math::
            u(x,y,t) = 1 - e^{-8\pi^{2}\nu t}\sin(2\pi(x-t))\sin(\pi y)\cos(\pi y)
            v(x,y,t) =   - e^{-8\pi^{2}\nu t} \cos\(2\pi(x-t)) \cos^{2}(\pi y)
            p(x,y,t) = 1 + \frac{4}{17} e^{-16\pi^{2}\nu t} \cos\(4\pi(x-t)) \cos(\pi y)

        The solution is specifically constructed to investigate the impact of boundary
        conditions on numerical accuracy. The source term g =(g_1, g_2) is derived from
        the exact solution to ensure that it satisfies the Navier–Stokes equations.

    Boundary conditions:
        - Time-dependent Dirichlet conditions on the left and right boundaries
        - Time-independent Dirichlet conditions on the top and bottom boundaries

        This combination is intentionally chosen to study order reduction effects
        induced by time-dependent boundary conditions.

        In the related class fenicsx_NSE_periodic_mass, the same problem is formulated with
        periodic boundary conditions on the left and right boundaries, while retaining the
        time-independent Dirichlet conditions on the top and bottom boundaries, in order to
        isolate and compare the effects of time-dependent boundary data on the numerical accuracy.


    Parameters
    ----------
    nelems : int
        Number of elements per spatial direction.
    t0 : float
        Initial time.
    family : str
        Finite element family (e.g., 'CG' for Lagrange).
    order : int
        Polynomial degree of the function spaces.
    nu : float
        Kinematic viscosity.
    comm : MPI.Comm
        Spatial MPI communicator.


    Attributes
    ----------
    domain : dolfinx.mesh.Mesh
        Computational mesh.
    W : FunctionSpace
        Mixed function space for velocity and pressure.
    V, Q : FunctionSpace
        Velocity and pressure subspaces.
    w : Function
        Solution vector (velocity–pressure).
    rhs : Function
        Right-hand side of the nonlinear system.
    F : UFL form
        Nonlinear residual form.
    M : PETSc.Mat
        Mass matrix (velocity block).
     Mf : PETSc.Mat
        Mass matrix for the monolithic system (velocity and pressure).

    """

    dtype_u = mesh
    dtype_f = mesh

    def __init__(self, nelems=128, t0=0.0, family='CG', order=4, nu=0.1, comm=MPI.COMM_WORLD):

        # Create mesh
        domain = dfx.mesh.create_rectangle(
            comm,
            [np.array([-0.5, -0.5]), np.array([0.5, 0.5])],
            [nelems, nelems],
            cell_type=dfx.mesh.CellType.triangle,
        )

        # Define mixed function space for velocity and pressure
        Ve = basix.ufl.element(family, domain.topology.cell_name(), order, shape=(domain.geometry.dim,))
        Pe = basix.ufl.element(family, domain.topology.cell_name(), order - 1, shape=())
        We = basix.ufl.mixed_element([Ve, Pe])
        self.W = dfx.fem.functionspace(domain, We)
        self.V, WV_map = self.W.sub(0).collapse()
        self.Q, WQ_map = self.W.sub(1).collapse()

        # get number of dofs and set up the problem
        self.tmp_u = dfx.fem.Function(self.W)
        nx = len(self.tmp_u.x.array)
        print('DoFs on this level:', nx)

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super().__init__(init=(nx, comm, np.dtype('float64')))
        self._makeAttributeAndRegister(
            'nelems', 't0', 'family', 'order', 'nu', 'comm', localVars=locals(), readOnly=True
        )

        # initialize functions for the solution, source term, and right-hand side
        w_zero = dfx.fem.Function(self.W)
        self.g = dfx.fem.Function(self.V)
        self.rhs = dfx.fem.Function(self.W)
        self.tmp_f = dfx.fem.Function(self.W)
        self.u_bndry = dfx.fem.Function(self.W)

        # interpolate initial and boundary conditions
        self.convert_to_fenicsx_vector(input=self.u_exact(t0), output=self.u_bndry)

        w_zero.sub(0).interpolate(
            lambda x: np.vstack(
                (np.full(x.shape[1], 0.0, dtype=np.float64), np.full(x.shape[1], 0.0, dtype=np.float64))
            )
        )
        w_zero.sub(1).interpolate(lambda x: np.full(x.shape[1], 0.0, dtype=np.float64))

        # locate boundary facets
        fdim = domain.topology.dim - 1

        w_dbc = dfx.fem.Function(self.W)
        w_dbc.sub(0).interpolate(
            lambda x: np.vstack(
                (np.full(x.shape[1], 1.0, dtype=np.float64), np.full(x.shape[1], 0.0, dtype=np.float64))
            )
        )
        w_dbc.sub(1).interpolate(lambda x: np.full(x.shape[1], 1.0, dtype=np.float64))
        #
        TopBottom = dfx.mesh.locate_entities_boundary(
            domain, fdim, lambda x: np.logical_or(np.isclose(x[1], -0.5), np.isclose(x[1], 0.5))
        )
        LeftRight = dfx.mesh.locate_entities_boundary(
            domain, fdim, lambda x: np.logical_or(np.isclose(x[0], -0.5), np.isclose(x[0], 0.5))
        )

        dofs_TopBottom_u = dfx.fem.locate_dofs_topological((self.W.sub(0), self.V), fdim, TopBottom)
        dofs_TopBottom_p = dfx.fem.locate_dofs_topological((self.W.sub(1), self.Q), fdim, TopBottom)
        self.dofs_LeftRight_u = dfx.fem.locate_dofs_topological((self.W.sub(0), self.V), fdim, LeftRight)
        self.dofs_LeftRight_p = dfx.fem.locate_dofs_topological((self.W.sub(1), self.Q), fdim, LeftRight)
        #
        bcu_TB = dfx.fem.dirichletbc(w_dbc.sub(0).collapse(), dofs_TopBottom_u, self.W.sub(0))
        bcp_TB = dfx.fem.dirichletbc(w_dbc.sub(1).collapse(), dofs_TopBottom_p, self.W.sub(1))
        self.bcs = [bcu_TB, bcp_TB]

        # homogeneous boundary conditions for the velocity and pressure for the residual
        bndry_facets = dfx.mesh.locate_entities_boundary(domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
        dofs_bndry_u = dfx.fem.locate_dofs_topological((self.W.sub(0), self.V), fdim, bndry_facets)
        dofs_bndry_p = dfx.fem.locate_dofs_topological((self.W.sub(1), self.Q), fdim, bndry_facets)
        bcu_hom = dfx.fem.dirichletbc(w_zero.sub(0).collapse(), dofs_bndry_u, self.W.sub(0))
        bcp_hom = dfx.fem.dirichletbc(w_zero.sub(1).collapse(), dofs_bndry_p, self.W.sub(1))
        self.bc_hom = [bcu_hom, bcp_hom]
        self.fix_bc_for_residual = True

        # Define trial and test functions
        self.u, self.p = ufl.TrialFunctions(self.W)
        self.v, self.q = ufl.TestFunctions(self.W)

        # Assemble mass matrices
        '''
                | M_v   0 |                  | M_v    0  |
            M = |         |      and    Mf = |           |
                |  0    0 |                  |  0    M_p |

        where M_v and M_p are the mass matrix in the velocity and pressure spaces, respectively.
        '''
        a_M = ufl.dot(self.u, self.v) * ufl.dx
        self.M = dolfinx.fem.petsc.assemble_matrix(dfx.fem.form(a_M), bcs=[])
        self.M.assemble()

        a_Mf = ufl.dot(self.u, self.v) * ufl.dx + ufl.dot(self.p, self.q) * ufl.dx
        self.Mf = dolfinx.fem.petsc.assemble_matrix(dfx.fem.form(a_Mf), bcs=[])
        self.Mf.assemble()

        self.Lin_solver = PETSc.KSP().create(domain.comm)
        self.Lin_solver.setType(PETSc.KSP.Type.BCGS)
        self.Lin_solver.setOperators(self.Mf)
        self.Lin_solver.setTolerances(rtol=1e-10, atol=1e-10)
        pc = self.Lin_solver.getPC()
        pc.setType(PETSc.PC.Type.JACOBI)

        # Initialize the nonlinear form for the Navier–Stokes equations
        self.factor = dfx.fem.Constant(domain, PETSc.ScalarType(0.0))

        self.w = dfx.fem.Function(self.W)

        u, p = ufl.split(self.w)
        rhs_u, rhs_p = ufl.split(self.rhs)

        # define the nonlinear form to be compatible with the multi-point constraints
        F = (1 / self.factor) * ufl.dot(u, self.v) * ufl.dx
        F += ufl.dot(ufl.dot(u, ufl.nabla_grad(u)), self.v) * ufl.dx
        F += ufl.inner(self.nu * ufl.nabla_grad(u), ufl.nabla_grad(self.v)) * ufl.dx
        F -= ufl.dot(p, ufl.div(self.v)) * ufl.dx
        F -= ufl.dot(self.g, self.v) * ufl.dx
        F -= ufl.dot(ufl.div(u), self.q) * ufl.dx
        F -= (1 / self.factor) * ufl.dot(rhs_u, self.v) * ufl.dx
        F -= (1 / self.factor) * ufl.dot(rhs_p, self.q) * ufl.dx
        self.F = F

        # Assemble the nonlinear problem and configure the Newton solver
        self.petsc_options = {
            "snes_type": "newtonls",
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "snes_atol": 1e-12,
            "snes_rtol": 1e-12,
            "snes_linesearch_type": "none",
            "ksp_error_if_not_converged": True,
            "snes_error_if_not_converged": True,
        }

        # plotter for solution visualization
        self.plotter = None

    @staticmethod
    def convert_to_fenicsx_vector(input, output):
        output.x.array[:] = input[:]

    @staticmethod
    def convert_from_fenicsx_vector(input, output):
        output[:] = input.x.array[:]

    def solve_system(self, rhs, factor, u0, t):
        r"""
        Dolfin's nonlinear solver for :
        .. math::
           (u,v) + factor * (u \cdot \nabla u, v) + factor * \nu (\nabla u, \nabla v) - factor * (p, \nabla \cdot v) - factor * (g, v) - factor * (div(u), q) = (rhs_u, v) + (rhs_p, q)

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side for the nonlinear system.
        factor : float
            Abbrev. for the node-to-node stepsize (or any other factor required).
        u0 : dtype_u
            Initial guess for the iterative solver (not used here so far).
        t : float
            Current time.

        Returns
        -------
        w : dtype_u
            Solution.
        """

        w = self.dtype_u(self.init)
        self.factor.value = factor

        # update the source term at time t
        self.g.interpolate(self.source_term(t))

        # get boundary values from the exact solution at time t and update the boundary conditions
        self.convert_to_fenicsx_vector(input=self.u_exact(t), output=self.u_bndry)
        #
        bcu_LR = dfx.fem.dirichletbc(self.u_bndry.sub(0).collapse(), self.dofs_LeftRight_u, self.W.sub(0))
        bcp_LR = dfx.fem.dirichletbc(self.u_bndry.sub(1).collapse(), self.dofs_LeftRight_p, self.W.sub(1))
        bcup = self.bcs + [bcu_LR, bcp_LR]
        #
        self.convert_to_fenicsx_vector(input=rhs, output=self.tmp_f)
        self.tmp_f = self.__invert_mass_matrix(self.tmp_f)

        self.rhs.x.array[:] = self.tmp_f.x.array[:]

        problem = dolfinx.fem.petsc.NonlinearProblem(
            self.F, self.w, bcs=bcup, petsc_options=self.petsc_options, petsc_options_prefix="nonlinearsolver"
        )

        # solve the nonlinear system using Dolfinx's nonlinear solver
        problem.solve()

        self.w.x.scatter_forward()

        #
        self.convert_from_fenicsx_vector(input=self.w, output=w)

        return w

    def eval_f(self, w, t):
        """
        Routine to evaluate both parts of the right-hand side of the problem.

        Parameters
        ----------
        w : dtype_u
            Current values of the numerical solution.
        t : float
            Current time at which the numerical solution is computed.

        Returns
        -------
        f : dtype_f
            The right-hand side.
        """

        f = self.dtype_f(self.init)

        self.convert_to_fenicsx_vector(input=w, output=self.tmp_u)

        u, p = ufl.split(self.tmp_u)

        # get the forcing term
        g = self.source_term(t)

        F = -ufl.dot(ufl.dot(u, ufl.nabla_grad(u)), self.v) * ufl.dx
        F -= ufl.inner(self.nu * ufl.nabla_grad(u), ufl.nabla_grad(self.v)) * ufl.dx
        F += ufl.dot(p, ufl.div(self.v)) * ufl.dx
        F += ufl.dot(g, self.v) * ufl.dx
        F += ufl.dot(ufl.div(u), self.q) * ufl.dx

        b = dolfinx.fem.petsc.assemble_vector(dfx.fem.form(F))
        b.assemble()
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

        #
        with b.localForm() as loc_b, self.tmp_f.x.petsc_vec.localForm() as loc_f:
            loc_b.copy(loc_f)

        self.convert_from_fenicsx_vector(input=self.tmp_f, output=f)

        return f

    def apply_mass_matrix(self, u):
        r"""
        Routine to apply mass matrix.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution.

        Returns
        -------
        me : dtype_u
            The product :math:`M \vec{u}`.
        """

        self.convert_to_fenicsx_vector(input=u, output=self.tmp_u)
        self.M.mult(self.tmp_u.x.petsc_vec, self.tmp_f.x.petsc_vec)
        me = self.dtype_u(self.init)
        self.convert_from_fenicsx_vector(input=self.tmp_f, output=me)
        return me

    def __invert_mass_matrix(self, u):
        r"""
        Helper routine to invert mass matrix.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution.

        Returns
        -------
        me : dtype_u
            The product :math:`Mf^{-1} \vec{u}`.
        """
        self.Lin_solver.solve(u.x.petsc_vec, self.tmp_f.x.petsc_vec)
        self.tmp_f.x.scatter_forward()

        return self.tmp_f

    def source_term(self, t):
        r"""
        Routine to compute the source term at time :math:`t`.

        Parameters
        ----------
        t : float
            Time of the exact solution.

        Returns
        -------
        g : dtype_u
            Source term.
        """

        self.g.interpolate(
            lambda x: np.vstack(
                (
                    np.pi
                    / 34
                    * np.exp(-16 * np.pi**2 * self.nu * t)
                    * np.sin(4 * np.pi * (t - x[0]))
                    * np.cos(np.pi * x[1])
                    * (32 - 17 * np.cos(np.pi * x[1])),
                    #
                    2 * np.pi**2 * self.nu * np.exp(-8 * np.pi**2 * self.nu * t) * np.cos(2 * np.pi * (t - x[0]))
                    - np.pi
                    * np.exp(-16 * np.pi**2 * self.nu * t)
                    * np.sin(np.pi * x[1])
                    * (2 * np.cos(np.pi * x[1]) ** 3 + 4 / 17 * np.cos(4 * np.pi * (t - x[0]))),
                )
            )
        )
        return self.g

    def u_exact(self, t):
        r"""
        Routine to compute the exact solution at time :math:`t`.

        Parameters
        ----------
        t : float
            Time of the exact solution.

        Returns
        -------
        me : dtype_u
            Exact solution.
        """

        w = dfx.fem.Function(self.W)

        w.sub(0).interpolate(
            lambda x: np.vstack(
                (
                    1.0
                    - np.exp(-8 * np.pi**2 * self.nu * t)
                    * np.sin(2.0 * np.pi * (x[0] - t))
                    * np.sin(np.pi * x[1])
                    * np.cos(np.pi * x[1]),
                    #
                    -np.exp(-8 * np.pi**2 * self.nu * t)
                    * np.cos(2.0 * np.pi * (x[0] - t))
                    * (np.cos(np.pi * x[1])) ** 2,
                )
            )
        )

        w.sub(1).interpolate(
            lambda x: (4 / 17)
            * np.exp(-16 * np.pi**2 * self.nu * t)
            * np.cos(4 * np.pi * (x[0] - t))
            * np.cos(np.pi * x[1])
            + 1
        )

        me = self.dtype_u(self.init)
        self.convert_from_fenicsx_vector(input=w, output=me)

        return me

    def fix_residual(self, res):
        """
        Applies homogeneous Dirichlet boundary conditions to the residual

        Parameters
        ----------
        res : dtype_u
              Residual
        """
        self.convert_to_fenicsx_vector(input=res, output=self.tmp_u)
        dolfinx.fem.petsc.set_bc(self.tmp_u.x.petsc_vec, self.bc_hom)
        self.tmp_u.x.scatter_forward()
        self.convert_from_fenicsx_vector(input=self.tmp_u, output=res)
        return None


class fenicsx_NSE_periodic_mass(fenicsx_NSE_mass):
    r"""
    Monolithic incompressible Navier–Stokes problem in 2D using FEniCSx.

    This class implements a finite element discretization of the incompressible
    Navier–Stokes equations in a square domain using a monolithic formulation
    for velocity and pressure.

    Governing equations:
    .. math::
        \frac{\partial \mathbf{u}}{\partial t}
        + (\mathbf{u} \cdot \nabla)\mathbf{u}
        - \nu \Delta \mathbf{u}
        + \nabla p = f,
        \quad \nabla \cdot \mathbf{u} = 0

    Domain:
        A square domain :math:`\Omega = [-0.5, 0.5]^2` discretized with triangular
        elements.

    Spatial discretization:
        - Continuous Lagrange elements for velocity and pressure
        - Mixed finite element space :math:`W = V \times Q`
        - Monolithic formulation for the coupled velocity–pressure system

    Manufactured solution:
        A smooth, analytical, time-dependent solution is given by:
        ..math::
            u(x,y,t) = 1 - e^{-8\pi^{2}\nu t}\sin(2\pi(x-t))\sin(\pi y)\cos(\pi y)
            v(x,y,t) =   - e^{-8\pi^{2}\nu t} \cos\(2\pi(x-t)) \cos^{2}(\pi y)
            p(x,y,t) = 1 + \frac{4}{17} e^{-16\pi^{2}\nu t} \cos\(4\pi(x-t)) \cos(\pi y)

        The solution is specifically constructed to investigate the impact of boundary
        conditions on numerical accuracy. The source term g =(g_1, g_2) is derived from
        the exact solution to ensure that it satisfies the Navier–Stokes equations.

    Boundary conditions:
        - periodic boundary conditions on the left and right boundaries
        - Time-independent Dirichlet conditions on the top and bottom boundaries

        In this class, the same problem as "fenicsx_NSE_mass" is formulated with periodic
        boundary conditions on the left and right boundaries, while retaining the time-
        independent Dirichlet conditions on the top and bottom boundaries, in order to
        isolate and compare the effects of time-dependent boundary data on the numerical
        accuracy.


    Parameters
    ----------
    nelems : int
        Number of elements per spatial direction.
    t0 : float
        Initial time.
    family : str
        Finite element family (e.g., 'CG' for Lagrange).
    order : int
        Polynomial degree of the function spaces.
    nu : float
        Kinematic viscosity.
    comm : MPI.Comm
        Spatial MPI communicator.


    Attributes
    ----------
    domain : dolfinx.mesh.Mesh
        Computational mesh.
    W : FunctionSpace
        Mixed function space for velocity and pressure.
    V, Q : FunctionSpace
        Velocity and pressure subspaces.
    w : Function
        Solution vector (velocity–pressure).
    rhs : Function
        Right-hand side of the nonlinear system.
    F : UFL form
        Nonlinear residual form.
    M : PETSc.Mat
        Mass matrix (velocity block).
    Mf : PETSc.Mat
        Mass matrix for the monolithic system (velocity and pressure).
    """

    def __init__(self, nelems=128, t0=0.0, family='CG', order=4, nu=0.1, comm=MPI.COMM_WORLD):
        super().__init__(nelems=nelems, t0=t0, family=family, order=order, nu=nu, comm=comm)

        # get the mesh and its geometric properties for periodicity
        domain = self.W.mesh

        self.xmin = domain.geometry.x[:, 0].min()
        self.xmax = domain.geometry.x[:, 0].max()
        self.ymin = domain.geometry.x[:, 1].min()
        self.ymax = domain.geometry.x[:, 1].max()
        self.Lx = float(self.xmax - self.xmin)
        self.Ly = float(self.ymax - self.ymin)

        fdim = domain.topology.dim - 1

        # locate facets on the right boundary for periodicity and create meshtags for them
        facets_x = dfx.mesh.locate_entities_boundary(domain, fdim, self.periodic_boundary_x)
        arg_sort_x = np.argsort(facets_x)
        mt_x = dfx.mesh.meshtags(domain, fdim, facets_x[arg_sort_x], np.full(len(facets_x), 2, dtype=np.int32))

        # create multi-point constraints for periodicity and add the periodic constraints to tthe boundary conditions
        self.mpc = dolfinx_mpc.MultiPointConstraint(self.W)
        self.mpc.create_periodic_constraint_topological(
            self.W.sub(0).sub(0), mt_x, 2, self.periodic_relation_x, self.bcs
        )
        self.mpc.create_periodic_constraint_topological(
            self.W.sub(0).sub(1), mt_x, 2, self.periodic_relation_x, self.bcs
        )
        self.mpc.create_periodic_constraint_topological(self.W.sub(1), mt_x, 2, self.periodic_relation_x, self.bcs)
        self.mpc.finalize()

        # reassemble the nonlinear problem
        self.factor = dfx.fem.Constant(domain, PETSc.ScalarType(0.0))

        # redefine the solution function to be compatible with the multi-point constraints
        W_mpc = self.mpc.function_space
        self.w = dfx.fem.Function(W_mpc)
        u, p = ufl.split(self.w)
        rhs_u, rhs_p = ufl.split(self.rhs)

        # redefine the nonlinear form to be compatible with the multi-point constraints
        F = (1 / self.factor) * ufl.dot(u, self.v) * ufl.dx
        F += ufl.dot(ufl.dot(u, ufl.nabla_grad(u)), self.v) * ufl.dx
        F += ufl.inner(self.nu * ufl.nabla_grad(u), ufl.nabla_grad(self.v)) * ufl.dx
        F -= ufl.dot(p, ufl.div(self.v)) * ufl.dx
        F -= ufl.dot(self.g, self.v) * ufl.dx
        F -= ufl.dot(ufl.div(u), self.q) * ufl.dx
        F -= (1 / self.factor) * ufl.dot(rhs_u, self.v) * ufl.dx
        F -= (1 / self.factor) * ufl.dot(rhs_p, self.q) * ufl.dx

        # compute the Jacobian of the nonlinear form for use in the Newton solver
        Jac = ufl.derivative(F, self.w, ufl.TrialFunction(self.W))

        # assemble the nonlinear problem with the multi-point constraints and configure the Newton solver
        self.problem = dolfinx_mpc.NonlinearProblem(
            F, self.w, mpc=self.mpc, bcs=self.bcs, J=Jac, petsc_options=self.petsc_options
        )

    def solve_system(self, rhs, factor, u0, t):
        r"""
        Dolfin's nonlinear solver for :
        .. math::
           (u,v) + factor * (u \cdot \nabla u, v) + factor * \nu (\nabla u, \nabla v) - factor * (p, \nabla \cdot v) - factor * (g, v) - factor * (div(u), q) = (rhs_u, v) + (rhs_p, q)

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side for the nonlinear system.
        factor : float
            Abbrev. for the node-to-node stepsize (or any other factor required).
        u0 : dtype_u
            Initial guess for the iterative solver (not used here so far).
        t : float
            Current time.

        Returns
        -------
        w : dtype_u
            Solution.
        """
        self.g.interpolate(self.source_term(t))

        u = self.dtype_u(self.init)
        self.factor.value = factor
        #
        self.convert_to_fenicsx_vector(input=rhs, output=self.tmp_f)
        self.tmp_f = super()._fenicsx_NSE_mass__invert_mass_matrix(self.tmp_f)
        #
        self.rhs.x.array[:] = self.tmp_f.x.array[:]
        #
        self.problem.solve()
        self.w.x.scatter_forward()
        #
        self.convert_from_fenicsx_vector(input=self.w, output=u)

        return u

    def periodic_boundary_x(self, x):
        r"""
        Identifies the left and right boundaries for periodicity.
        Parameters
        ----------
        x : array_like
            Coordinates of the points to check for periodicity.
        Returns
        -------
        boolTrue if the point is on the right boundary (x = xmax), False otherwise.

        """
        eps = 10 * np.finfo(x.dtype).resolution
        return np.isclose(x[0], self.xmax, atol=eps)

    def periodic_relation_x(self, x):
        r"""
        Maps points from the right boundary to the left boundary for periodicity.
        Parameters
        ----------
        x : array_like
            Coordinates of the points on the right boundary to be mapped.
        Returns
        -------
        out_x : array_like
            Coordinates of the corresponding points on the left boundary after applying periodicity.
        """
        out_x = np.zeros_like(x)
        out_x[0] = x[0] - self.Lx
        out_x[1] = x[1]
        out_x[2] = x[2]
        return out_x
