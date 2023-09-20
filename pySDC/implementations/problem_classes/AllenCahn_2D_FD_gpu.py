import numpy as np
import cupy as cp
import cupyx.scipy.sparse as csp
from cupyx.scipy.sparse.linalg import cg  # , spsolve, gmres, minres

from pySDC.core.Errors import ProblemError
from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.cupy_mesh import cupy_mesh, imex_cupy_mesh, comp2_cupy_mesh

# http://www.personal.psu.edu/qud2/Res/Pre/dz09sisc.pdf


class allencahn_fullyimplicit(ptype):  # pragma: no cover
    r"""
    Example implementing the two-dimensional Allen-Cahn equation with periodic boundary conditions :math:`u \in [-1, 1]^2`

    .. math::
        \frac{\partial u}{\partial t} = \Delta u + \frac{1}{\varepsilon^2} u (1 - u^\nu)

    for constant parameter :math:`\nu`. Initial condition are circles of the form

    .. math::
        u({\bf x}, 0) = \tanh\left(\frac{r - \sqrt{x_i^2 + y_j^2}}{\sqrt{2}\varepsilon}\right)

    for :math::`i, j=0,..,N-1`, where :math:`N` is the number of spatial grid points. For time-stepping, the problem is
    treated *fully-implicitly*, i.e., the nonlinear system is solved by Newton.

    This class is especially developed for solving it on GPU's using CuPy.

    Parameters
    ----------
    nvars : tuple of int, optional
        Number of unknowns in the problem, e.g. (128, 128).
    nu : float, optional
        Problem parameter :math:`\nu`.
    eps : float, optional
        Scaling parameter :math:`\varepsilon`.
    newton_maxiter : int, optional
        Maximum number of iterations for the Newton solver.
    newton_tol : float, optional
        Tolerance for Newton's method to terminate.
    lin_tol : float, optional
        Tolerance for linear solver to terminate.
    lin_maxiter : int, optional
        Maximum number of iterations for the linear solver.
    radius : float, optional
        Radius of the circles.

    Attributes
    ----------
    A : scipy.spdiags
        Second-order FD discretization of the 2D laplace operator.
    dx : float
        Distance between two spatial nodes (same for both directions).
    xvalues : np.1darray
        Spatial grid points, here both dimensions have the same grid points.
    newton_itercount : int
        Number of iterations of Newton solver.
    lin_itercount
        Number of iterations of linear solver.
    newton_ncalls : int
        Number of calls of Newton solver.
    lin_ncalls : int
        Number of calls of linear solver.
    """

    dtype_u = cupy_mesh
    dtype_f = cupy_mesh

    def __init__(
        self,
        nvars=(128, 128),
        nu=2,
        eps=0.04,
        newton_maxiter=1e-12,
        newton_tol=100,
        lin_tol=1e-8,
        lin_maxiter=100,
        radius=0.25,
    ):
        """Initialization routine"""
        # we assert that nvars looks very particular here.. this will be necessary for coarsening in space later on
        if len(nvars) != 2:
            raise ProblemError('this is a 2d example, got %s' % nvars)
        if nvars[0] != nvars[1]:
            raise ProblemError('need a square domain, got %s' % nvars)
        if nvars[0] % 2 != 0:
            raise ProblemError('the setup requires nvars = 2^p per dimension')

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super().__init__((nvars, None, cp.dtype('float64')))
        self._makeAttributeAndRegister(
            'nvars',
            'nu',
            'eps',
            'newton_maxiter',
            'newton_tol',
            'lin_tol',
            'lin_maxiter',
            'radius',
            localVars=locals(),
            readOnly=True,
        )

        # compute dx and get discretization matrix A
        self.dx = 1.0 / self.nvars[0]
        self.A = self.__get_A(self.nvars, self.dx)
        self.xvalues = cp.array([i * self.dx - 0.5 for i in range(self.nvars[0])])

        self.newton_itercount = 0
        self.lin_itercount = 0
        self.newton_ncalls = 0
        self.lin_ncalls = 0

    @staticmethod
    def __get_A(N, dx):
        """
        Helper function to assemble FD matrix A in sparse format.

        Parameters
        ----------
        N : list
            Number of degrees of freedom.
        dx : float
            Distance between two spatial nodes.

        Returns
        -------
        A : cupyx.scipy.sparse.csr_matrix
            CuPy-matrix A in CSR format.
        """

        stencil = cp.asarray([-2, 1])
        A = stencil[0] * csp.eye(N[0], format='csr')
        for i in range(1, len(stencil)):
            A += stencil[i] * csp.eye(N[0], k=-i, format='csr')
            A += stencil[i] * csp.eye(N[0], k=+i, format='csr')
            A += stencil[i] * csp.eye(N[0], k=N[0] - i, format='csr')
            A += stencil[i] * csp.eye(N[0], k=-N[0] + i, format='csr')
        A = csp.kron(A, csp.eye(N[0])) + csp.kron(csp.eye(N[1]), A)
        A *= 1.0 / (dx**2)
        return A

    # noinspection PyTypeChecker
    def solve_system(self, rhs, factor, u0, t):
        """
        Simple Newton solver.

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side for the nonlinear system.
        factor : float
            Abbrev. for the node-to-node stepsize (or any other factor required).
        u0 : dtype_u
            Initial guess for the iterative solver.
        t : float
            Current time (required here for the BC).

        Returns
        -------
        u : dtype_u
            The solution as mesh.
        """

        u = self.dtype_u(u0).flatten()
        z = self.dtype_u(self.init, val=0.0).flatten()
        nu = self.nu
        eps2 = self.eps**2

        Id = csp.eye(self.nvars[0] * self.nvars[1])

        # start newton iteration
        n = 0
        res = 99
        while n < self.newton_maxiter:
            # form the function g with g(u) = 0
            g = u - factor * (self.A.dot(u) + 1.0 / eps2 * u * (1.0 - u**nu)) - rhs.flatten()

            # if g is close to 0, then we are done
            res = cp.linalg.norm(g, np.inf)

            if res < self.newton_tol:
                break

            # assemble dg
            dg = Id - factor * (self.A + 1.0 / eps2 * csp.diags((1.0 - (nu + 1) * u**nu), offsets=0))

            # newton update: u1 = u0 - g/dg
            # u -= spsolve(dg, g)
            u -= cg(dg, g, x0=z, tol=self.lin_tol)[0]
            # increase iteration count
            n += 1
            # print(n, res)

        # if n == self.newton_maxiter:
        #     raise ProblemError('Newton did not converge after %i iterations, error is %s' % (n, res))

        me = self.dtype_u(self.init)
        me[:] = u.reshape(self.nvars)

        self.newton_ncalls += 1
        self.newton_itercount += n

        return me

    def eval_f(self, u, t):
        """
        Routine to evaluate the right-hand side of the problem.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution.
        t : float
            Current time of the numerical solution is computed (not used here).

        Returns
        -------
        f : dtype_f
            The right-hand side of the problem.
        """
        f = self.dtype_f(self.init)
        v = u.flatten()
        f[:] = (self.A.dot(v) + 1.0 / self.eps**2 * v * (1.0 - v**self.nu)).reshape(self.nvars)

        return f

    def u_exact(self, t):
        """
        Routine to compute the exact solution at time t.

        Parameters
        ----------
        t : float
            Time of the exact solution.

        Returns
        -------
        me : dtype_u
            The exact solution.
        """

        assert t == 0, 'ERROR: u_exact only valid for t=0'
        me = self.dtype_u(self.init, val=0.0)
        mx, my = cp.meshgrid(self.xvalues, self.xvalues)
        me[:] = cp.tanh((self.radius - cp.sqrt(mx**2 + my**2)) / (cp.sqrt(2) * self.eps))
        # print(type(me))
        return me


# noinspection PyUnusedLocal
class allencahn_semiimplicit(allencahn_fullyimplicit):
    r"""
    This class implements the two-dimensional Allen-Cahn equation with periodic boundary conditions :math:`u \in [-1, 1]^2`

    .. math::
        \frac{\partial u}{\partial t} = \Delta u + \frac{1}{\varepsilon^2} u (1 - u^\nu)

    for constant parameter :math:`\nu`. Initial condition are circles of the form

    .. math::
        u({\bf x}, 0) = \tanh\left(\frac{r - \sqrt{x_i^2 + y_j^2}}{\sqrt{2}\varepsilon}\right)

    for :math::`i, j=0,..,N-1`, where :math:`N` is the number of spatial grid points. For time-stepping, the problem is
    treated in a *semi-implicit* way, i.e., the linear system containing the Laplacian is solved by the
    conjugate gradients method, and the system containing the rest of the right-hand side is only evaluated at each time.

    This class is especially developed for solving it on GPUs using CuPy.
    """

    dtype_f = imex_cupy_mesh

    def eval_f(self, u, t):
        """
        Routine to evaluate the right-hand side of the problem.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution.
        t : float
            Current time of the numerical solution is computed.

        Returns
        -------
        f : dtype_f
            The right-hand side of the problem.
        """
        f = self.dtype_f(self.init)
        v = u.flatten()
        f.impl[:] = self.A.dot(v).reshape(self.nvars)
        f.expl[:] = (1.0 / self.eps**2 * v * (1.0 - v**self.nu)).reshape(self.nvars)

        return f

    def solve_system(self, rhs, factor, u0, t):
        r"""
        Simple linear solver for :math:`(I-factor\cdot A)\vec{u}=\vec{rhs}`.

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side for the linear system.
        factor : float
            Abbrev. for the local stepsize (or any other factor required).
        u0 : dtype_u
            Initial guess for the iterative solver.
        t : float
            Current time (e.g. for time-dependent BCs).

        Returns
        -------
        me : dtype_u
            The solution as mesh.
        """

        class context:
            num_iter = 0

        def callback(xk):
            context.num_iter += 1
            return context.num_iter

        me = self.dtype_u(self.init)

        Id = csp.eye(self.nvars[0] * self.nvars[1])

        me[:] = cg(
            Id - factor * self.A,
            rhs.flatten(),
            x0=u0.flatten(),
            tol=self.lin_tol,
            maxiter=self.lin_maxiter,
            callback=callback,
        )[0].reshape(self.nvars)

        self.lin_ncalls += 1
        self.lin_itercount += context.num_iter

        return me


# noinspection PyUnusedLocal
class allencahn_semiimplicit_v2(allencahn_fullyimplicit):
    r"""
    This class implements the two-dimensional Allen-Cahn equation with periodic boundary conditions :math:`u \in [-1, 1]^2`

    .. math::
        \frac{\partial u}{\partial t} = \Delta u + \frac{1}{\varepsilon^2} u (1 - u^\nu)

    for constant parameter :math:`\nu`. Initial condition are circles of the form

    .. math::
        u({\bf x}, 0) = \tanh\left(\frac{r - \sqrt{x_i^2 + y_j^2}}{\sqrt{2}\varepsilon}\right)

    for :math::`i, j=0,..,N-1`, where :math:`N` is the number of spatial grid points. For time-stepping, a special AC-splitting
    is used to get a *semi-implicit* treatment of the problem: The term :math:`\Delta u - \frac{1}{\varepsilon^2} u^{\nu + 1}`
    is handled implicitly and the nonlinear system including this part will be solved by Newton. :math:`\frac{1}{\varepsilon^2} u`
    is only evaluated at each time.

    This class is especially developed for solving it on GPUs using CuPy.
    """

    dtype_f = imex_cupy_mesh

    def eval_f(self, u, t):
        """
        Routine to evaluate the right-hand side of the problem.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution.
        t : float
            Current time of the numerical solution is computed.

        Returns
        -------
        f : dtype_f
            The right-hand side of the problem.
        """
        f = self.dtype_f(self.init)
        v = u.flatten()
        f.impl[:] = (self.A.dot(v) - 1.0 / self.eps**2 * v ** (self.nu + 1)).reshape(self.nvars)
        f.expl[:] = (1.0 / self.eps**2 * v).reshape(self.nvars)

        return f

    def solve_system(self, rhs, factor, u0, t):
        """
        Simple Newton solver.

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side for the nonlinear system.
        factor : float
            Abbrev. for the node-to-node stepsize (or any other factor required).
        u0 : dtype_u
            Initial guess for the iterative solver.
        t : float
            Current time (required here for the BC).

        Returns
        -------
        me : dtype_u
            The solution as mesh.
        """

        u = self.dtype_u(u0).flatten()
        z = self.dtype_u(self.init, val=0.0).flatten()
        nu = self.nu
        eps2 = self.eps**2

        Id = csp.eye(self.nvars[0] * self.nvars[1])

        # start newton iteration
        n = 0
        res = 99
        while n < self.newton_maxiter:
            # form the function g with g(u) = 0
            g = u - factor * (self.A.dot(u) - 1.0 / eps2 * u ** (nu + 1)) - rhs.flatten()

            # if g is close to 0, then we are done
            res = cp.linalg.norm(g, np.inf)

            if res < self.newton_tol:
                break

            # assemble dg
            dg = Id - factor * (self.A - 1.0 / eps2 * csp.diags(((nu + 1) * u**nu), offsets=0))

            # newton update: u1 = u0 - g/dg
            # u -= spsolve(dg, g)
            u -= cg(dg, g, x0=z, tol=self.lin_tol)[0]
            # increase iteration count
            n += 1
            # print(n, res)

        # if n == self.newton_maxiter:
        #     raise ProblemError('Newton did not converge after %i iterations, error is %s' % (n, res))

        me = self.dtype_u(self.init)
        me[:] = u.reshape(self.nvars)

        self.newton_ncalls += 1
        self.newton_itercount += n

        return me


# noinspection PyUnusedLocal
class allencahn_multiimplicit(allencahn_fullyimplicit):
    r"""
    Example implementing the two-dimensional Allen-Cahn equation with periodic boundary conditions :math:`u \in [-1, 1]^2`

    .. math::
        \frac{\partial u}{\partial t} = \Delta u + \frac{1}{\varepsilon^2} u (1 - u^\nu)

    for constant parameter :math:`\nu`. Initial condition are circles of the form

    .. math::
        u({\bf x}, 0) = \tanh\left(\frac{r - \sqrt{x_i^2 + y_j^2}}{\sqrt{2}\varepsilon}\right)

    for :math::`i, j=0,..,N-1`, where :math:`N` is the number of spatial grid points. For time-stepping, the problem is
    treated in *multi-implicit* fashion, i.e., the nonlinear system containing the second order term is solved by the
    conjugate gradients method, and the system containing the rest of the right-hand side will be solved by Newton's method.

    This class is especially developed for solving it on GPUs using CuPy.
    """

    dtype_f = comp2_cupy_mesh

    def eval_f(self, u, t):
        """
        Routine to evaluate the right-hand side of the problem.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution.
        t : float
            Current time of the numerical solution is computed (not used here).

        Returns
        -------
        f : dtype_f
            The right-hand side of the problem.
        """
        f = self.dtype_f(self.init)
        v = u.flatten()
        f.comp1[:] = self.A.dot(v).reshape(self.nvars)
        f.comp2[:] = (1.0 / self.eps**2 * v * (1.0 - v**self.nu)).reshape(self.nvars)

        return f

    def solve_system_1(self, rhs, factor, u0, t):
        r"""
        Simple linear solver for :math:`(I-factor\cdot A)\vec{u}=\vec{rhs}`.

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side for the linear system.
        factor : float
            Abbrev. for the local stepsize (or any other factor required).
        u0 : dtype_u
            Initial guess for the iterative solver.
        t : float
            Current time (e.g. for time-dependent BCs).

        Returns
        -------
        me : dtype_u
            The solution as mesh.
        """

        class context:
            num_iter = 0

        def callback(xk):
            context.num_iter += 1
            return context.num_iter

        me = self.dtype_u(self.init)

        Id = csp.eye(self.nvars[0] * self.nvars[1])

        me[:] = cg(
            Id - factor * self.A,
            rhs.flatten(),
            x0=u0.flatten(),
            tol=self.lin_tol,
            maxiter=self.lin_maxiter,
            callback=callback,
        )[0].reshape(self.nvars)

        self.lin_ncalls += 1
        self.lin_itercount += context.num_iter

        return me

    def solve_system_2(self, rhs, factor, u0, t):
        """
        Simple Newton solver.

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side for the nonlinear system
        factor : float
            Abbrev. for the node-to-node stepsize (or any other factor required).
        u0 : dtype_u
            Initial guess for the iterative solver.
        t : float
            Current time (required here for the BC).

        Returns
        -------
        me : dtype_u
            The solution as mesh.
        """

        u = self.dtype_u(u0).flatten()
        z = self.dtype_u(self.init, val=0.0).flatten()
        nu = self.nu
        eps2 = self.eps**2

        Id = csp.eye(self.nvars[0] * self.nvars[1])

        # start newton iteration
        n = 0
        res = 99
        while n < self.newton_maxiter:
            # form the function g with g(u) = 0
            g = u - factor * (1.0 / eps2 * u * (1.0 - u**nu)) - rhs.flatten()

            # if g is close to 0, then we are done
            res = cp.linalg.norm(g, np.inf)

            if res < self.newton_tol:
                break

            # assemble dg
            dg = Id - factor * (1.0 / eps2 * csp.diags((1.0 - (nu + 1) * u**nu), offsets=0))

            # newton update: u1 = u0 - g/dg
            # u -= spsolve(dg, g)
            u -= cg(dg, g, x0=z, tol=self.lin_tol)[0]
            # increase iteration count
            n += 1
            # print(n, res)

        # if n == self.newton_maxiter:
        #     raise ProblemError('Newton did not converge after %i iterations, error is %s' % (n, res))

        me = self.dtype_u(self.init)
        me[:] = u.reshape(self.nvars)

        self.newton_ncalls += 1
        self.newton_itercount += n

        return me


# noinspection PyUnusedLocal
class allencahn_multiimplicit_v2(allencahn_fullyimplicit):
    r"""
    This class implements the two-dimensional Allen-Cahn equation with periodic boundary conditions :math:`u \in [-1, 1]^2`

    .. math::
        \frac{\partial u}{\partial t} = \Delta u + \frac{1}{\varepsilon^2} u (1 - u^\nu)

    for constant parameter :math:`\nu`. The initial condition has the form of circles

    .. math::
        u({\bf x}, 0) = \tanh\left(\frac{r - \sqrt{x_i^2 + y_j^2}}{\sqrt{2}\varepsilon}\right)

    for :math::`i, j=0,..,N-1`, where :math:`N` is the number of spatial grid points. For time-stepping, a special AC-splitting
    is used here to get another kind of *semi-implicit* treatment of the problem: The term :math:`\Delta u - \frac{1}{\varepsilon^2} u^{\nu + 1}`
    is handled implicitly and the nonlinear system including this part will be solved by Newton. :math:`\frac{1}{\varepsilon^2} u`
    is solved by a linear solver provided by a scipy routine.

    This class is especially developed for solving it on GPUs using CuPy.
    """

    dtype_f = comp2_cupy_mesh

    def eval_f(self, u, t):
        """
        Routine to evaluate the right-hand side of the problem.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution.
        t : float
            Current time of the numerical solution is computed (not used here).

        Returns
        -------
        f : dtype_f
            The right-hand side of the problem.
        """
        f = self.dtype_f(self.init)
        v = u.flatten()
        f.comp1[:] = (self.A.dot(v) - 1.0 / self.eps**2 * v ** (self.nu + 1)).reshape(self.nvars)
        f.comp2[:] = (1.0 / self.eps**2 * v).reshape(self.nvars)

        return f

    def solve_system_1(self, rhs, factor, u0, t):
        """
        Simple Newton solver.

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side for the nonlinear system.
        factor : float
            Abbrev. for the node-to-node stepsize (or any other factor required).
        u0 : dtype_u
            Initial guess for the iterative solver.
        t : float
            Current time (required here for the BC).

        Returns
        -------
        me : dtype_u
            The solution as mesh.
        """

        u = self.dtype_u(u0).flatten()
        z = self.dtype_u(self.init, val=0.0).flatten()
        nu = self.nu
        eps2 = self.eps**2

        Id = csp.eye(self.nvars[0] * self.nvars[1])

        # start newton iteration
        n = 0
        res = 99
        while n < self.newton_maxiter:
            # form the function g with g(u) = 0
            g = u - factor * (self.A.dot(u) - 1.0 / eps2 * u ** (nu + 1)) - rhs.flatten()

            # if g is close to 0, then we are done
            res = cp.linalg.norm(g, np.inf)

            if res < self.newton_tol:
                break

            # assemble dg
            dg = Id - factor * (self.A - 1.0 / eps2 * csp.diags(((nu + 1) * u**nu), offsets=0))

            # newton update: u1 = u0 - g/dg
            # u -= spsolve(dg, g)
            u -= cg(dg, g, x0=z, tol=self.lin_tol)[0]
            # increase iteration count
            n += 1
            # print(n, res)

        # if n == self.newton_maxiter:
        #     raise ProblemError('Newton did not converge after %i iterations, error is %s' % (n, res))

        me = self.dtype_u(self.init)
        me[:] = u.reshape(self.nvars)

        self.newton_ncalls += 1
        self.newton_itercount += n

        return me

    def solve_system_2(self, rhs, factor, u0, t):
        r"""
        Simple linear solver for :math:`(I-factor\cdot A)\vec{u}=\vec{rhs}`.

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side for the linear system.
        factor : float
            Abbrev. for the local stepsize (or any other factor required).
        u0 : dtype_u
            Initial guess for the iterative solver.
        t : float
            Current time (e.g. for time-dependent BCs).

        Returns
        -------
        me : dtype_u
            The solution as mesh.
        """

        me = self.dtype_u(self.init)

        me[:] = (1.0 / (1.0 - factor * 1.0 / self.eps**2) * rhs).reshape(self.nvars)
        return me
