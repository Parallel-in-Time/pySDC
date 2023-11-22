import logging

import dolfin as df
import numpy as np

from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.fenics_mesh import fenics_mesh, rhs_fenics_mesh


# noinspection PyUnusedLocal
class fenics_heat(ptype):
    r"""
    Example implementing the forced one-dimensional heat equation with Dirichlet boundary conditions

    .. math::
        \frac{d u}{d t} = \nu \frac{d^2 u}{d x^2} + f

    for :math:`x \in \Omega:=[0,1]`, where the forcing term :math:`f` is defined by

    .. math::
        f(x, t) = -\cos(\pi x) (\sin(t) - \nu \pi^2 \cos(t)).

    The exact solution of the problem is

    .. math::
        u(x, t) = \cos(\pi x)\cos(t).

    In this class the problem is implemented in the way that the spatial part is solved using ``FEniCS`` [1]_. Hence, the problem
    is reformulated to the *weak formulation*

    .. math:
        \int_\Omega u_t v\,dx = - \nu \int_\Omega \nabla u \nabla v\,dx + \int_\Omega f v\,dx.

    The part containing the forcing term is treated explicitly, where it is interpolated in the function space.
    The other part will be treated in an implicit way.

    Parameters
    ----------
    c_nvars : int, optional
        Spatial resolution, i.e., numbers of degrees of freedom in space.
    t0 : float, optional
        Starting time.
    family : str, optional
        Indicates the family of elements used to create the function space
        for the trail and test functions. The default is ``'CG'``, which are the class
        of Continuous Galerkin, a *synonym* for the Lagrange family of elements, see [2]_.
    order : int, optional
        Defines the order of the elements in the function space.
    refinements : int, optional
        Denotes the refinement of the mesh. ``refinements=2`` refines the mesh by factor :math:`2`.
    nu : float, optional
        Diffusion coefficient :math:`\nu`.

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
        Denotes the Dirichlet boundary conditions (currently not used here).

    References
    ----------
    .. [1] The FEniCS Project Version 1.5. M. S. Alnaes, J. Blechta, J. Hake, A. Johansson, B. Kehlet, A. Logg,
        C. Richardson, J. Ring, M. E. Rognes, G. N. Wells. Archive of Numerical Software (2015).
    .. [2] Automated Solution of Differential Equations by the Finite Element Method. A. Logg, K.-A. Mardal, G. N.
        Wells and others. Springer (2012).
    """

    dtype_u = fenics_mesh
    dtype_f = rhs_fenics_mesh

    def __init__(self, c_nvars=128, t0=0.0, family='CG', order=4, refinements=1, nu=0.1):
        """Initialization routine"""

        # define the Dirichlet boundary
        # def Boundary(x, on_boundary):
        #     return on_boundary

        # set logger level for FFC and dolfin
        logging.getLogger('FFC').setLevel(logging.WARNING)
        logging.getLogger('UFL').setLevel(logging.WARNING)

        # set solver and form parameters
        df.parameters["form_compiler"]["optimize"] = True
        df.parameters["form_compiler"]["cpp_optimize"] = True
        df.parameters['allow_extrapolation'] = True

        # set mesh and refinement (for multilevel)
        mesh = df.UnitIntervalMesh(c_nvars)
        for _ in range(refinements):
            mesh = df.refine(mesh)

        # define function space for future reference
        self.V = df.FunctionSpace(mesh, family, order)
        tmp = df.Function(self.V)
        print('DoFs on this level:', len(tmp.vector()[:]))

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(fenics_heat, self).__init__(self.V)
        self._makeAttributeAndRegister(
            'c_nvars', 't0', 'family', 'order', 'refinements', 'nu', localVars=locals(), readOnly=True
        )

        # Stiffness term (Laplace)
        u = df.TrialFunction(self.V)
        v = df.TestFunction(self.V)
        a_K = -1.0 * df.inner(df.nabla_grad(u), self.nu * df.nabla_grad(v)) * df.dx

        # Mass term
        a_M = u * v * df.dx

        self.M = df.assemble(a_M)
        self.K = df.assemble(a_K)

        # set forcing term as expression
        self.g = df.Expression(
            '-cos(a*x[0]) * (sin(t) - b*a*a*cos(t))',
            a=np.pi,
            b=self.nu,
            t=self.t0,
            degree=self.order,
        )
        # self.g = df.Expression('0', a=np.pi, b=self.nu, t=self.t0,
        #                        degree=self.order)
        # set boundary values
        # bc = df.DirichletBC(self.V, df.Constant(0.0), Boundary)
        #
        # bc.apply(self.M)
        # bc.apply(self.K)

    def solve_system(self, rhs, factor, u0, t):
        r"""
        Dolfin's linear solver for :math:`(M - factor \cdot A) \vec{u} = \vec{rhs}`.

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

        b = self.apply_mass_matrix(rhs)

        u = self.dtype_u(u0)
        df.solve(self.M - factor * self.K, u.values.vector(), b.values.vector())

        return u

    def __eval_fexpl(self, u, t):
        """
        Helper routine to evaluate the explicit part of the right-hand side.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution (not used here).
        t : float
            Current time at which the numerical solution is computed.

        Returns
        -------
        fexpl : dtype_u
            Explicit part of the right-hand side.
        """

        self.g.t = t
        fexpl = self.dtype_u(df.interpolate(self.g, self.V))

        return fexpl

    def __eval_fimpl(self, u, t):
        """
        Helper routine to evaluate the implicit part of the right-hand side.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution.
        t : float
            Current time at which the numerical solution is computed.

        Returns
        -------
        fimpl : dtype_u
            Explicit part of the right-hand side.
        """

        tmp = self.dtype_u(self.V)
        self.K.mult(u.values.vector(), tmp.values.vector())
        fimpl = self.__invert_mass_matrix(tmp)

        return fimpl

    def eval_f(self, u, t):
        """
        Routine to evaluate both parts of the right-hand side of the problem.

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
        f.impl = self.__eval_fimpl(u, t)
        f.expl = self.__eval_fexpl(u, t)
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
            The product :math:`M^{-1} \vec{u}`.
        """

        me = self.dtype_u(self.V)

        b = self.dtype_u(u)

        df.solve(self.M, me.values.vector(), b.values.vector())

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

        u0 = df.Expression('cos(a*x[0]) * cos(t)', a=np.pi, t=t, degree=self.order)
        me = self.dtype_u(df.interpolate(u0, self.V))

        return me


# noinspection PyUnusedLocal
class fenics_heat_mass(fenics_heat):
    r"""
    Example implementing the forced one-dimensional heat equation with Dirichlet boundary conditions

    .. math::
        \frac{d u}{d t} = \nu \frac{d^2 u}{d x^2} + f

    for :math:`x \in \Omega:=[0,1]`, where the forcing term :math:`f` is defined by

    .. math::
        f(x, t) = -\cos(\pi x) (\sin(t) - \nu \pi^2 \cos(t)).

    The exact solution of the problem is

    .. math::
        u(x, t) = \cos(\pi x)\cos(t).

    In this class the problem is implemented in the way that the spatial part is solved using ``FEniCS`` [1]_. Hence, the problem
    is reformulated to the *weak formulation*

    .. math:
        \int_\Omega u_t v\,dx = - \nu \int_\Omega \nabla u \nabla v\,dx + \int_\Omega f v\,dx.

    The forcing term is treated explicitly, and is expressed via the mass matrix resulting from the left-hand side term
    :math:`\int_\Omega u_t v\,dx`, and the other part will be treated in an implicit way.

    Parameters
    ----------
    c_nvars : int, optional
        Spatial resolution, i.e., numbers of degrees of freedom in space.
    t0 : float, optional
        Starting time.
    family : str, optional
        Indicates the family of elements used to create the function space
        for the trail and test functions. The default is ``'CG'``, which are the class
        of Continuous Galerkin, a *synonym* for the Lagrange family of elements, see [2]_.
    order : int, optional
        Defines the order of the elements in the function space.
    refinements : int, optional
        Denotes the refinement of the mesh. ``refinements=2`` refines the mesh by factor :math:`2`.
    nu : float, optional
        Diffusion coefficient :math:`\nu`.

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
        Denotes the Dirichlet boundary conditions (currently not used here).

    References
    ----------
    .. [1] The FEniCS Project Version 1.5. M. S. Alnaes, J. Blechta, J. Hake, A. Johansson, B. Kehlet, A. Logg,
        C. Richardson, J. Ring, M. E. Rognes, G. N. Wells. Archive of Numerical Software (2015).
    .. [2] Automated Solution of Differential Equations by the Finite Element Method. A. Logg, K.-A. Mardal, G. N.
        Wells and others. Springer (2012).
    """

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
        df.solve(self.M - factor * self.K, u.values.vector(), rhs.values.vector())

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
        f.expl = self.dtype_u(df.interpolate(self.g, self.V))
        f.expl = self.apply_mass_matrix(f.expl)

        return f
