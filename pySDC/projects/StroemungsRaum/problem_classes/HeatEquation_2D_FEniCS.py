import logging

import dolfin as df
import numpy as np

from pySDC.core.problem import Problem
from pySDC.implementations.datatype_classes.fenics_mesh import fenics_mesh, rhs_fenics_mesh


class fenics_heat2D_mass(Problem):
    r"""
    Example implementing the forced two-dimensional heat equation with Dirichlet boundary conditions

    .. math::
        \frac{\partial u}{\partial t} = \nu \Delta u + f

    for :math:`x \in \Omega:=[0,1]x[0,1]`, where the forcing term :math:`f` is defined by

    .. math::
        f(x, y, t) = -\sin(\pi x)\sin(\pi y) (\sin(t) - 2\nu \pi^2 \cos(t)).

    For initial conditions with constant c and

    .. math::
        u(x, y, 0) = \sin(\pi x)\sin(\pi y) + c

    the exact solution of the problem is given by

    .. math::
        u(x, y, t) = \sin(\pi x)\sin(\pi y)\cos(t) + c.

    In this class the spatial part is solved using ``FEniCS`` [1]_. Hence, the problem is reformulated to
    the *weak formulation*

    .. math:
        \int_\Omega u_t v\,dx = - \nu \int_\Omega \nabla u \nabla v\,dx + \int_\Omega f v\,dx.

    The part containing the forcing term is treated explicitly, where it is interpolated in the function
    space. The other part will be treated in an implicit way.

    Parameters
    ----------
    c_nvars : int, optional
        Spatial resolution.
    t0 : float, optional
        Starting time.
    family : str, optional
        Indicates the family of elements used to create the function space
        for the trail and test functions. The default is ``'CG'``, which are the class
        of Continuous Galerkin, a *synonym* for the Lagrange family of elements, see [2]_.
    order : int, optional
        Defines the order of the elements in the function space.
    nu : float, optional
        Diffusion coefficient :math:`\nu`.
    c: float, optional
        Constant for the Dirichlet boundary condition :math: `c`

    Attributes
    ----------
    V : FunctionSpace
        Defines the function space of the trial and test functions.
    M : scalar, vector, matrix or higher rank tensor
        Denotes the expression :math:`\int_\Omega u_t v\,dx`.
    K : scalar, vector, matrix or higher rank tensor
        Denotes the expression :math:`- \nu \int_\Omega \nabla u \nabla v\,dx`.
    g : Expression
        The forcing term :math:`f` in the heat equation.
    bc : DirichletBC
        Denotes the Dirichlet boundary conditions.

    References
    ----------
    .. [1] The FEniCS Project Version 1.5. M. S. Alnaes, J. Blechta, J. Hake, A. Johansson, B. Kehlet, A. Logg,
        C. Richardson, J. Ring, M. E. Rognes, G. N. Wells. Archive of Numerical Software (2015).
    .. [2] Automated Solution of Differential Equations by the Finite Element Method. A. Logg, K.-A. Mardal, G. N.
        Wells and others. Springer (2012).
    """

    dtype_u = fenics_mesh
    dtype_f = rhs_fenics_mesh

    def __init__(self, c_nvars=64, t0=0.0, family='CG', order=2, nu=0.1, c=0.0):

        # define the Dirichlet boundary
        def Boundary(x, on_boundary):
            return on_boundary

        # set solver and form parameters
        df.parameters["form_compiler"]["optimize"] = True
        df.parameters["form_compiler"]["cpp_optimize"] = True
        df.parameters['allow_extrapolation'] = True

        # set mesh and refinement (for multilevel)
        mesh = df.UnitSquareMesh(c_nvars, c_nvars)

        # define function space for future reference
        self.V = df.FunctionSpace(mesh, family, order)
        tmp = df.Function(self.V)
        self.logger.debug('DoFs on this level:', len(tmp.vector()[:]))

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super().__init__(self.V)
        self._makeAttributeAndRegister('c_nvars', 't0', 'family', 'order', 'nu', 'c', localVars=locals(), readOnly=True)

        # Stiffness term (Laplace)
        u = df.TrialFunction(self.V)
        v = df.TestFunction(self.V)
        a_K = -1.0 * df.inner(df.nabla_grad(u), self.nu * df.nabla_grad(v)) * df.dx

        # Mass term
        a_M = u * v * df.dx

        self.M = df.assemble(a_M)
        self.K = df.assemble(a_K)

        # set boundary values
        self.u_D = df.Expression('sin(a*x[0]) * sin(a*x[1]) * cos(t) + c', c=self.c, a=np.pi, t=t0, degree=self.order)
        self.bc = df.DirichletBC(self.V, self.u_D, Boundary)
        self.bc_hom = df.DirichletBC(self.V, df.Constant(0), Boundary)

        self.fix_bc_for_residual = True

        # set forcing term as expression
        self.g = df.Expression(
            '-sin(a*x[0]) * sin(a*x[1]) * (sin(t) - 2*b*a*a*cos(t))',
            a=np.pi,
            b=self.nu,
            t=self.t0,
            degree=self.order,
        )

    def solve_system(self, rhs, factor, u0, t):
        r"""
        Dolfin's linear solver for :math:`(M - factor A) \vec{u} = \vec{rhs}`.

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
        u : dtype_u
            Solution.
        """

        u = self.dtype_u(u0)
        T = self.M - factor * self.K
        b = self.dtype_u(rhs)

        self.u_D.t = t

        self.bc.apply(T, b.values.vector())
        self.bc.apply(b.values.vector())

        df.solve(T, u.values.vector(), b.values.vector())

        return u

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

        f = self.dtype_f(self.V)

        self.K.mult(u.values.vector(), f.impl.values.vector())

        self.g.t = t
        f.expl = self.apply_mass_matrix(self.dtype_u(df.interpolate(self.g, self.V)))
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
        u0 = df.Expression('sin(a*x[0]) * sin(a*x[1]) * cos(t) + c', c=self.c, a=np.pi, t=t, degree=self.order)
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
