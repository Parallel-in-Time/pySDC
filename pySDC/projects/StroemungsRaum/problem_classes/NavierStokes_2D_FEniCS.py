import logging
import dolfin as df
import numpy as np
import os

from pySDC.core.problem import Problem
from pySDC.implementations.datatype_classes.fenics_mesh import fenics_mesh, rhs_fenics_mesh


class fenics_NSE_2D_mass(Problem):
    r"""
    Example implementing a forced two-dimensional incompressible Navier–Stokes problem for the DFG
    benchmark flow around cylinder using FEniCS (dolfin).

    The unknowns are the velocity field :math:`\mathbf{u}(x,y,t)` and the pressure field
    :math:`p(x,y,t)` on a 2D domain :math:`\Omega` (here loaded from ``cylinder.xml``).
    The model is

    .. math::
        \frac{\partial \mathbf{u}}{\partial t}
        + (\mathbf{u}\cdot\nabla)\mathbf{u}
        - \nu \Delta \mathbf{u}
        + \nabla p
        = \mathbf{f},
        \qquad
        \nabla\cdot \mathbf{u} = 0,

    where :math:`\nu` is the kinematic viscosity and :math:`\mathbf{f}` is a forcing term.

    This implementation uses a fractional-step /projection method:
    a viscous velocity solve, a pressure Poisson solve enforcing incompressibility, and a velocity
    correction using the pressure gradient.

    Boundary conditions are applied on subsets of the boundary:
    - no-slip on channel walls and cylinder surface,
    - a time-dependent inflow profile at the inlet,
    - pressure condition at the outflow.

    Parameters
    ----------
    t0 : float, optional
        Starting time.
    family : str, optional
        Finite element family for velocity and pressure spaces. Default is ``'CG'``.
    order : int, optional
        Polynomial degree for the velocity space. The pressure degree is chosen as ``order - 1``.
    nu : float, optional
        Kinematic viscosity :math:`\nu`.

    Attributes
    ----------
    mesh : Mesh
        Computational mesh loaded from ``cylinder.xml``.
    V : VectorFunctionSpace
        Function space for the velocity field.
    Q : FunctionSpace
        Function space for the pressure field.
    M : Matrix
        Assembled velocity mass matrix, corresponding to :math:`\int_\Omega \mathbf{u}\cdot\mathbf{v}\,dx`.
    K : Matrix
        Assembled viscous operator matrix, corresponding to
        :math:`- \int_\Omega \nabla \mathbf{u} : (\nu \nabla \mathbf{v})\,dx`.
    Mp : Matrix
        Assembled pressure mass matrix, corresponding to :math:`\int_\Omega p\,q\,dx`.
    g : Expression
        Forcing term :math:`\mathbf{f}` (here set to zero).
    bcu : list[DirichletBC]
        Velocity Dirichlet boundary conditions (walls, inflow, cylinder).
    bcp : list[DirichletBC]
        Pressure boundary conditions (outflow).
    bc_hom : DirichletBC
        Homogeneous velocity boundary condition used to fix the residual when requested.

    Notes
    -----
    The right-hand side evaluation splits into an implicit viscous part and an explicit part
    containing forcing and nonlinear convection:
    - implicit: :math:`K\,\mathbf{u}`
    - explicit: :math:`M\,\mathbf{f} - M\,(\mathbf{u}\cdot\nabla)\mathbf{u}`
    """

    dtype_u = fenics_mesh
    dtype_f = rhs_fenics_mesh

    def __init__(self, t0=0.0, family='CG', order=2, nu=0.001):

        # set logger level for FFC and dolfin
        logging.getLogger('FFC').setLevel(logging.WARNING)
        logging.getLogger('UFL').setLevel(logging.WARNING)

        # set solver and form parameters
        df.parameters["form_compiler"]["optimize"] = True
        df.parameters["form_compiler"]["cpp_optimize"] = True
        df.parameters['allow_extrapolation'] = True

        # load mesh
        path = f"{os.path.dirname(__file__)}/../meshs/"
        mesh = df.Mesh(path + '/cylinder.xml')

        # define function space for future reference
        self.V = df.VectorFunctionSpace(mesh, family, order)
        self.Q = df.FunctionSpace(mesh, family, order - 1)

        tmp_u = df.Function(self.V)
        tmp_p = df.Function(self.Q)
        self.logger.debug('Velocity DoFs on this level:', len(tmp_u.vector()[:]))
        self.logger.debug('Pressure DoFs on this level:', len(tmp_p.vector()[:]))

        super().__init__(self.V)
        self._makeAttributeAndRegister('t0', 'family', 'order', 'nu', localVars=locals(), readOnly=True)

        # define trial and test functions
        self.u = df.TrialFunction(self.V)
        self.v = df.TestFunction(self.V)
        self.p = df.TrialFunction(self.Q)
        self.q = df.TestFunction(self.Q)

        # initialize solution functions for pressure
        self.pn = df.Function(self.Q)

        # mass and stiffness terms
        a_K = -1.0 * df.inner(df.nabla_grad(self.u), self.nu * df.nabla_grad(self.v)) * df.dx
        a_M = df.inner(self.u, self.v) * df.dx
        a_S = df.inner(df.nabla_grad(self.p), df.nabla_grad(self.q)) * df.dx
        a_Mp = df.inner(self.p, self.q) * df.dx

        # assemble the forms
        self.M = df.assemble(a_M)
        self.K = df.assemble(a_K)
        self.S = df.assemble(a_S)
        self.Mp = df.assemble(a_Mp)

        # set inflow profile at the domain inlet
        self.inflow_profile = df.Expression(
            ('4.0*1.5*sin(pi*t/8)*x[1]*(0.41 - x[1]) / pow(0.41, 2)', '0'), pi=np.pi, t=t0, degree=self.order
        )

        # define boundaries
        inflow = 'near(x[0], 0)'
        outflow = 'near(x[0], 2.2)'
        walls = 'near(x[1], 0) || near(x[1], 0.41)'
        cylinder = 'on_boundary && x[0]>0.1 && x[0]<0.3 && x[1]>0.1 && x[1]<0.3'

        # define boundary conditions
        bcu_noslip = df.DirichletBC(self.V, df.Constant((0, 0)), walls)
        bcu_inflow = df.DirichletBC(self.V, self.inflow_profile, inflow)
        bcu_cylinder = df.DirichletBC(self.V, df.Constant((0, 0)), cylinder)
        bcp_outflow = df.DirichletBC(self.Q, df.Constant(0), outflow)

        # set boundary values
        self.bcu = [bcu_noslip, bcu_inflow, bcu_cylinder]
        self.bcp = [bcp_outflow]
        self.bc_hom = df.DirichletBC(self.V, df.Constant((0.0, 0.0)), 'on_boundary')
        self.fix_bc_for_residual = True

        # Define measure for drag and lift computation
        Cylinder = df.CompiledSubDomain(cylinder)
        CylinderBoundary = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
        Cylinder.mark(CylinderBoundary, 1)
        self.dsc = df.Measure("ds", domain=mesh, subdomain_data=CylinderBoundary, subdomain_id=1)

        # set forcing term as expression
        self.g = df.Expression(('0.0', '0.0'), t=self.t0, degree=self.order)

        # initialize XDMF files for velocity and pressure if needed
        self.xdmffile_p = None
        self.xdmffile_u = None

    def solve_system(self, rhs, factor, u0, t, dtau):
        r"""
        Dolfin's linear solver for :math:`(M - factor A) \vec{u} = \vec{rhs}`.

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side for the nonlinear system.
        factor : float
           Factor for the linear system.
        u0 : dtype_u
            Initial guess for the iterative solver (not used here so far).
        t : float
            Current time.
        dtau : float
              Zero-to-node step size in the SDC sweep, used in the pressure correction step.

        Returns
        -------
        u : dtype_u
            Solution: velocity.
        p : space function
            Solution: pressure.

        """

        u = self.dtype_u(u0)
        p = df.Function(self.Q)
        T = self.M - factor * self.K

        # solve for the intermediate velocity
        self.inflow_profile.t = t
        [bc.apply(T, rhs.values.vector()) for bc in self.bcu]
        df.solve(T, u.values.vector(), rhs.values.vector())

        # solve for the pressure correction
        L2 = -(1 / dtau) * df.div(u.values) * self.q * df.dx
        Rhs2 = df.assemble(L2)
        [bc.apply(self.S, Rhs2) for bc in self.bcp]
        df.solve(self.S, p.vector(), Rhs2)

        # velocity correction
        L3 = df.dot(u.values, self.v) * df.dx - dtau * df.dot(df.nabla_grad(p), self.v) * df.dx
        Rhs3 = df.assemble(L3)
        [bc.apply(self.M, Rhs3) for bc in self.bcu]
        df.solve(self.M, u.values.vector(), Rhs3)

        return u, p

    def eval_f(self, u, t):
        """
        Routine to evaluate both parts of the right-hand side.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution.
        t : float
            Current time at which the numerical solution is computed.

        Returns
        -------
        f : dtype_f
            The right-hand side divided into two parts.
        """
        self.g.t = t
        conv = -1.0 * df.dot(u.values, df.nabla_grad(u.values))

        f = self.dtype_f(self.V)

        self.K.mult(u.values.vector(), f.impl.values.vector())
        f.expl = self.apply_mass_matrix(self.dtype_u(df.project(conv, self.V)))
        f.expl += self.apply_mass_matrix(self.dtype_u(df.interpolate(self.g, self.V)))

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

        me = self.dtype_u(self.V)
        self.M.mult(u.values.vector(), me.values.vector())

        return me

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

        u0 = df.Expression(('0.0', '0.0'), degree=self.order)
        me = self.dtype_u(df.interpolate(u0, self.V), val=self.V)

        return me

    def fix_residual(self, res):
        """
        Applies homogeneous Dirichlet boundary conditions to the residual

        Parameters
        ----------
        res : dtype_u
              Residual
        """
        self.bc_hom.apply(res.values.vector())
        return None
