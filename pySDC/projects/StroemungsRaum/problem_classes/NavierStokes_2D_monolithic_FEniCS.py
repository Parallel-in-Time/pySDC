import logging
import os

import dolfin as df
import numpy as np

from pySDC.core.problem import Problem
from pySDC.implementations.datatype_classes.fenics_mesh import fenics_mesh


class fenics_NSE_2D_Monolithic(Problem):
    r"""
    Example implementing the forced two-dimensional incompressible Navier-Stokes equations with
    time-dependent Dirichlet boundary conditions

    .. math::
        \frac{\partial u}{\partial t} = - u \cdot \nabla u  + \nu \nabla u - \nabla p + f
                      0 = \nabla \cdot u

    for :math:`x \in \Omega`, where the forcing term :math:`f` is defined by

    .. math::
        f(x, t) = (0, 0).

    This implementation follows a monolithic approach, where velocity and pressure are
    solved simultaneously in a coupled system using a mixed finite element formulation.

    Boundary conditions are applied on subsets of the boundary:
    - no-slip on channel walls and cylinder surface,
    - a time-dependent inflow profile at the inlet,
    - pressure condition at the outflow.

    In this class the problem is implemented in the way that the spatial part is solved using ``FEniCS`` [1]_. Hence, the problem
    is reformulated to the *weak formulation*

    .. math:
        \int_\Omega u_t v\,dx = - \int_\Omega u \cdot \nabla u v\,dx -  \nu \int_\Omega \nabla u \nabla v\,dx + \int_\Omega p \nabla \cdot v\,dx + \int_\Omega f v\,dx
        \int_\Omega \nabla \cdot u q\,dx = 0

    Parameters
    ----------
    t0 : float, optional
        Starting time.
    order : int, optional
        Defines the order of the elements in the function space.
    nu : float, optional
        Diffusion coefficient :math:`\nu`.

    Attributes
    ----------
    V : FunctionSpace
        Defines the function space of the trial and test functions for velocity.
    Q : FunctionSpace
        Defines the function space of the trial and test functions for pressure.
    W : FunctionSpace
        Defines the mixed function space for the coupled velocity-pressure system.
    M : scalar, vector, matrix or higher rank tensor
        Denotes the expression :math:`\int_\Omega u_t v\,dx`.
    Mf : scalar, vector, matrix or higher rank tensor
        Denotes the expression :math:`\int_\Omega u v\,dx + \int_\Omega p q\,dx`.
    g : Expression
        The forcing term :math:`f` in the heat equation.
    bc : DirichletBC
        Denotes the time-dependent Dirichlet boundary conditions.
    bc_hom : DirichletBC
        Denotes the homogeneous Dirichlet boundary conditions, potentially required for fixing the residual
    fix_bc_for_residual: boolean
        flag to indicate that the residual requires special treatment due to boundary conditions

    References
    ----------
    .. [1] The FEniCS Project Version 1.5. M. S. Alnaes, J. Blechta, J. Hake, A. Johansson, B. Kehlet, A. Logg,
        C. Richardson, J. Ring, M. E. Rognes, G. N. Wells. Archive of Numerical Software (2015).
    .. [2] Automated Solution of Differential Equations by the Finite Element Method. A. Logg, K.-A. Mardal, G. N.
        Wells and others. Springer (2012).
    """

    dtype_u = fenics_mesh
    dtype_f = fenics_mesh

    df.set_log_active(False)

    def __init__(self, t0=0.0, order=2, nu=0.001):

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

        # define function spaces for future reference (Taylor-Hood)
        P2 = df.VectorElement("P", mesh.ufl_cell(), order)
        P1 = df.FiniteElement("P", mesh.ufl_cell(), order - 1)
        TH = df.MixedElement([P2, P1])
        self.W = df.FunctionSpace(mesh, TH)
        self.V = df.FunctionSpace(mesh, P2)
        self.Q = df.FunctionSpace(mesh, P1)

        # print the number of DoFs for debugging purposes
        tmp = df.Function(self.W)
        self.logger.debug('DoFs on this level:', len(tmp.vector()[:]))

        super().__init__(self.W)
        self._makeAttributeAndRegister('t0', 'order', 'nu', localVars=locals(), readOnly=True)

        # Trial and test function for the Mixed FE space
        self.u, self.p = df.TrialFunctions(self.W)
        self.v, self.q = df.TestFunctions(self.W)

        # velocity mass matrix
        a_M = df.inner(self.u, self.v) * df.dx
        self.M = df.assemble(a_M)

        # full mass matrix
        a_Mf = df.inner(self.u, self.v) * df.dx + df.inner(self.p, self.q) * df.dx
        self.Mf = df.assemble(a_Mf)

        # define the time-dependent inflow profile as an Expression
        Uin = '4.0*1.5*sin(pi*t/8)*x[1]*(0.41 - x[1]) / pow(0.41, 2)'
        self.u_in = df.Expression((Uin, '0'), pi=np.pi, t=t0, degree=self.order)

        # define boundaries
        inflow = 'near(x[0], 0)'
        outflow = 'near(x[0], 2.2)'
        walls = 'near(x[1], 0) || near(x[1], 0.41)'
        cylinder = 'on_boundary && x[0]>0.1 && x[0]<0.3 && x[1]>0.1 && x[1]<0.3'

        # define boundary conditions
        bc_in = df.DirichletBC(self.W.sub(0), self.u_in, inflow)
        bc_out = df.DirichletBC(self.W.sub(1), 0, outflow)
        bc_walls = df.DirichletBC(self.W.sub(0), (0, 0), walls)
        bc_cylinder = df.DirichletBC(self.W.sub(0), (0, 0), cylinder)
        self.bc = [bc_cylinder, bc_walls, bc_out, bc_in]

        # homogeneous boundary conditions for fixing the residual
        bc_hom_u = df.DirichletBC(self.W.sub(0), df.Constant((0, 0)), 'on_boundary')
        bc_hom_p = df.DirichletBC(self.W.sub(1), df.Constant(0), 'on_boundary')
        self.bc_hom = [bc_hom_u, bc_hom_p]
        self.fix_bc_for_residual = True

        # define measure for drag and lift computation
        Cylinder = df.CompiledSubDomain(cylinder)
        CylinderBoundary = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
        Cylinder.mark(CylinderBoundary, 1)
        self.dsc = df.Measure("ds", domain=mesh, subdomain_data=CylinderBoundary, subdomain_id=1)

        # set forcing term as expression
        self.g = df.Expression(('0', '0'), a=np.pi, b=self.nu, t=self.t0, degree=self.order)

        # initialize XDMF files for velocity and pressure if needed
        self.xdmffile_p = None
        self.xdmffile_u = None

    def solve_system(self, rhs, factor, u0, t):
        r"""
        Dolfin's nonlinear solver for :

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
        # introduce the coupled solution vector for velocity and pressure
        w = self.dtype_u(u0)
        u, p = df.split(w.values)

        # get the SDC right-hand side
        rhs = self.__invert_mass_matrix(rhs)
        rhs_u, rhs_p = df.split(rhs.values)

        # update time in Boundary conditions
        self.u_in.t = t

        # get the forcing term
        self.g.t = t
        g = df.interpolate(self.g, self.V)

        # build the variational form for the coupled system
        F = df.dot(u, self.v) * df.dx
        F += factor * df.dot(df.dot(u, df.nabla_grad(u)), self.v) * df.dx
        F += factor * self.nu * df.inner(df.nabla_grad(u), df.nabla_grad(self.v)) * df.dx
        F -= factor * df.dot(p, df.div(self.v)) * df.dx
        F -= factor * df.dot(g, self.v) * df.dx
        F -= factor * df.dot(df.div(u), self.q) * df.dx
        F -= df.dot(rhs_u, self.v) * df.dx
        F -= df.dot(rhs_p, self.q) * df.dx

        # solve the nonlinear system using Newton's method
        df.solve(F == 0, w.values, self.bc, solver_parameters={"newton_solver": {"absolute_tolerance": 1e-15}})

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

        f = self.dtype_f(self.W)

        u, p = df.split(w.values)

        # get the forcing term
        self.g.t = t
        g = self.dtype_f(df.interpolate(self.g, self.V), val=self.V)

        F = -1.0 * df.dot(df.dot(u, df.nabla_grad(u)), self.v) * df.dx
        F -= self.nu * df.inner(df.nabla_grad(u), df.nabla_grad(self.v)) * df.dx
        F += df.dot(p, df.div(self.v)) * df.dx
        F += df.dot(g.values, self.v) * df.dx
        F += df.dot(df.div(u), self.q) * df.dx

        f.values.vector()[:] = df.assemble(F)

        return f

    def apply_mass_matrix(self, w):
        r"""
        Routine to apply velocity mass matrix.

        Parameters
        ----------
        w : dtype_u
            Current values of the numerical solution.

        Returns
        -------
        me : dtype_u
            The product :math:`M \vec{w}`.
        """

        me = self.dtype_u(self.W)
        self.M.mult(w.values.vector(), me.values.vector())

        return me

    def __invert_mass_matrix(self, w):
        r"""
        Helper routine to invert the full mass matrix Mf.

        Parameters
        ----------
        w : dtype_u
            Current values of the numerical solution.

        Returns
        -------
        me : dtype_u
            The product :math:`Mf^{-1} \vec{w}`.
        """

        me = self.dtype_u(self.W)
        df.solve(self.Mf, me.values.vector(), w.values.vector())
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
        # define the exact solution
        w = df.Function(self.W)

        # assign the exact solution as an Expression
        df.assign(w.sub(0), df.interpolate(df.Expression(('0.0', '0.0'), degree=self.order), self.V))
        df.assign(w.sub(1), df.interpolate(df.Expression('0.0', degree=self.order - 1), self.Q))

        # update time in Boundary conditions
        self.u_in.t = t
        [bc.apply(w.vector()) for bc in self.bc]

        me = self.dtype_u(w, val=self.W)

        return me

    def fix_residual(self, res):
        """
        Applies homogeneous Dirichlet boundary conditions to the residual

        Parameters
        ----------
        res : dtype_u
              Residual
        """
        [bc.apply(res.values.vector()) for bc in self.bc_hom]

        return None
