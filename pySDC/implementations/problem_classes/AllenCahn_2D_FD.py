import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg

from pySDC.core.Errors import ParameterError, ProblemError
from pySDC.core.Problem import ptype, WorkCounter
from pySDC.helpers import problem_helper
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh, comp2_mesh


# http://www.personal.psu.edu/qud2/Res/Pre/dz09sisc.pdf


# noinspection PyUnusedLocal
class allencahn_fullyimplicit(ptype):
    r"""
    Example implementing the two-dimensional Allen-Cahn equation with periodic boundary conditions :math:`u \in [-1, 1]^2`

    .. math::
        \frac{\partial u}{\partial t} = \Delta u + \frac{1}{\varepsilon^2} u (1 - u^\nu)

    for constant parameter :math:`\nu`. Initial condition are circles of the form

    .. math::
        u({\bf x}, 0) = \tanh\left(\frac{r - \sqrt{x_i^2 + y_j^2}}{\sqrt{2}\varepsilon}\right)

    for :math:`i, j=0,..,N-1`, where :math:`N` is the number of spatial grid points. For time-stepping, the problem is
    treated *fully-implicitly*, i.e., the nonlinear system is solved by Newton.

    Parameters
    ----------
    nvars : tuple of int, optional
        Number of unknowns in the problem, e.g. ``nvars=(128, 128)``.
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
    order : int, optional
        Order of the finite difference matrix.

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

    dtype_u = mesh
    dtype_f = mesh

    def __init__(
        self,
        nvars=(128, 128),
        nu=2,
        eps=0.04,
        newton_maxiter=200,
        newton_tol=1e-12,
        lin_tol=1e-8,
        lin_maxiter=100,
        inexact_linear_ratio=None,
        radius=0.25,
        order=2,
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
        super().__init__((nvars, None, np.dtype('float64')))
        self._makeAttributeAndRegister(
            'nvars',
            'nu',
            'eps',
            'radius',
            'order',
            localVars=locals(),
            readOnly=True,
        )
        self._makeAttributeAndRegister(
            'newton_maxiter',
            'newton_tol',
            'lin_tol',
            'lin_maxiter',
            'inexact_linear_ratio',
            localVars=locals(),
            readOnly=False,
        )

        # compute dx and get discretization matrix A
        self.dx = 1.0 / self.nvars[0]
        self.A, _ = problem_helper.get_finite_difference_matrix(
            derivative=2,
            order=self.order,
            stencil_type='center',
            dx=self.dx,
            size=self.nvars[0],
            dim=2,
            bc='periodic',
        )
        self.xvalues = np.array([i * self.dx - 0.5 for i in range(self.nvars[0])])

        self.newton_itercount = 0
        self.lin_itercount = 0
        self.newton_ncalls = 0
        self.lin_ncalls = 0

        self.work_counters['newton'] = WorkCounter()
        self.work_counters['rhs'] = WorkCounter()
        self.work_counters['linear'] = WorkCounter()

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
        A : scipy.sparse.csc_matrix
            Matrix in CSC format.
        """

        stencil = [1, -2, 1]
        zero_pos = 2

        dstencil = np.concatenate((stencil, np.delete(stencil, zero_pos - 1)))
        offsets = np.concatenate(
            (
                [N[0] - i - 1 for i in reversed(range(zero_pos - 1))],
                [i - zero_pos + 1 for i in range(zero_pos - 1, len(stencil))],
            )
        )
        doffsets = np.concatenate((offsets, np.delete(offsets, zero_pos - 1) - N[0]))

        A = sp.diags(dstencil, doffsets, shape=(N[0], N[0]), format='csc')
        A = sp.kron(A, sp.eye(N[0])) + sp.kron(sp.eye(N[1]), A)
        A *= 1.0 / (dx**2)
        return A

    # noinspection PyTypeChecker
    def solve_system(self, rhs, factor, u0, t):
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

        Id = sp.eye(self.nvars[0] * self.nvars[1])

        # start newton iteration
        n = 0
        res = 99
        while n < self.newton_maxiter:
            # form the function g with g(u) = 0
            g = u - factor * (self.A.dot(u) + 1.0 / eps2 * u * (1.0 - u**nu)) - rhs.flatten()

            # if g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)

            # do inexactness in the linear solver
            if self.inexact_linear_ratio:
                self.lin_tol = res * self.inexact_linear_ratio

            if res < self.newton_tol:
                break

            # assemble dg
            dg = Id - factor * (self.A + 1.0 / eps2 * sp.diags((1.0 - (nu + 1) * u**nu), offsets=0))

            # newton update: u1 = u0 - g/dg
            # u -= spsolve(dg, g)
            u -= cg(
                dg, g, x0=z, tol=self.lin_tol, maxiter=self.lin_maxiter, atol=0, callback=self.work_counters['linear']
            )[0]
            # increase iteration count
            n += 1
            # print(n, res)

            self.work_counters['newton']()

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

        self.work_counters['rhs']()
        return f

    def u_exact(self, t, u_init=None, t_init=None):
        r"""
        Routine to compute the exact solution at time :math:`t`.

        Parameters
        ----------
        t : float
            Time of the exact solution.

        Returns
        -------
        me : dtype_u
            The exact solution.
        """
        me = self.dtype_u(self.init, val=0.0)
        if t > 0:

            def eval_rhs(t, u):
                return self.eval_f(u.reshape(self.init[0]), t).flatten()

            me[:] = self.generate_scipy_reference_solution(eval_rhs, t, u_init, t_init)

        else:
            for i in range(self.nvars[0]):
                for j in range(self.nvars[1]):
                    r2 = self.xvalues[i] ** 2 + self.xvalues[j] ** 2
                    me[i, j] = np.tanh((self.radius - np.sqrt(r2)) / (np.sqrt(2) * self.eps))

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

    for :math:`i, j=0,..,N-1`, where :math:`N` is the number of spatial grid points. For time-stepping, the problem is
    treated in a *semi-implicit* way, i.e., the linear system containing the Laplacian is solved by the conjugate gradients
    method, and the system containing the rest of the right-hand side is only evaluated at each time.
    """

    dtype_f = imex_mesh

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
        f.impl[:] = self.A.dot(v).reshape(self.nvars)
        f.expl[:] = (1.0 / self.eps**2 * v * (1.0 - v**self.nu)).reshape(self.nvars)

        self.work_counters['rhs']()
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
            self.work_counters['linear']()
            return context.num_iter

        me = self.dtype_u(self.init)

        Id = sp.eye(self.nvars[0] * self.nvars[1])

        me[:] = cg(
            Id - factor * self.A,
            rhs.flatten(),
            x0=u0.flatten(),
            tol=self.lin_tol,
            maxiter=self.lin_maxiter,
            atol=0,
            callback=callback,
        )[0].reshape(self.nvars)

        self.lin_ncalls += 1
        self.lin_itercount += context.num_iter

        return me

    def u_exact(self, t, u_init=None, t_init=None):
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
        me = self.dtype_u(self.init, val=0.0)
        if t > 0:

            def eval_rhs(t, u):
                f = self.eval_f(u.reshape(self.init[0]), t)
                return (f.impl + f.expl).flatten()

            me[:] = self.generate_scipy_reference_solution(eval_rhs, t, u_init, t_init)
        else:
            me[:] = super().u_exact(t, u_init, t_init)
        return me


# noinspection PyUnusedLocal
class allencahn_semiimplicit_v2(allencahn_fullyimplicit):
    r"""
    This class implements the two-dimensional Allen-Cahn (AC) equation with periodic boundary conditions :math:`u \in [-1, 1]^2`

    .. math::
        \frac{\partial u}{\partial t} = \Delta u + \frac{1}{\varepsilon^2} u (1 - u^\nu)

    for constant parameter :math:`\nu`. Initial condition are circles of the form

    .. math::
        u({\bf x}, 0) = \tanh\left(\frac{r - \sqrt{x_i^2 + y_j^2}}{\sqrt{2}\varepsilon}\right)

    for :math:`i, j=0,..,N-1`, where :math:`N` is the number of spatial grid points. For time-stepping, a special AC-splitting
    is used to get a *semi-implicit* treatment of the problem: The term :math:`\Delta u - \frac{1}{\varepsilon^2} u^{\nu + 1}`
    is handled implicitly and the nonlinear system including this part will be solved by Newton. :math:`\frac{1}{\varepsilon^2} u`
    is only evaluated at each time.
    """

    dtype_f = imex_mesh

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

        Id = sp.eye(self.nvars[0] * self.nvars[1])

        # start newton iteration
        n = 0
        res = 99
        while n < self.newton_maxiter:
            # form the function g with g(u) = 0
            g = u - factor * (self.A.dot(u) - 1.0 / eps2 * u ** (nu + 1)) - rhs.flatten()

            # if g is close to 0, then we are done
            # res = np.linalg.norm(g, np.inf)
            res = np.linalg.norm(g, np.inf)

            if res < self.newton_tol:
                break

            # assemble dg
            dg = Id - factor * (self.A - 1.0 / eps2 * sp.diags(((nu + 1) * u**nu), offsets=0))

            # newton update: u1 = u0 - g/dg
            # u -= spsolve(dg, g)
            u -= cg(dg, g, x0=z, tol=self.lin_tol, atol=0)[0]
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

    for :math:`i, j=0,..,N-1`, where :math:`N` is the number of spatial grid points. For time-stepping, the problem is
    treated in *multi-implicit* fashion, i.e., the linear system containing the Laplacian is solved by the conjugate gradients
    method, and the system containing the rest of the right-hand side will be solved by Newton's method.
    """

    dtype_f = comp2_mesh

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

        Id = sp.eye(self.nvars[0] * self.nvars[1])

        me[:] = cg(
            Id - factor * self.A,
            rhs.flatten(),
            x0=u0.flatten(),
            tol=self.lin_tol,
            maxiter=self.lin_maxiter,
            atol=0,
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

        Id = sp.eye(self.nvars[0] * self.nvars[1])

        # start newton iteration
        n = 0
        res = 99
        while n < self.newton_maxiter:
            # form the function g with g(u) = 0
            g = u - factor * (1.0 / eps2 * u * (1.0 - u**nu)) - rhs.flatten()

            # if g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)

            if res < self.newton_tol:
                break

            # assemble dg
            dg = Id - factor * (1.0 / eps2 * sp.diags((1.0 - (nu + 1) * u**nu), offsets=0))

            # newton update: u1 = u0 - g/dg
            # u -= spsolve(dg, g)
            u -= cg(dg, g, x0=z, tol=self.lin_tol, atol=0)[0]
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
    This class implements the two-dimensional Allen-Cahn (AC) equation with periodic boundary conditions :math:`u \in [-1, 1]^2`

    .. math::
        \frac{\partial u}{\partial t} = \Delta u + \frac{1}{\varepsilon^2} u (1 - u^\nu)

    for constant parameter :math:`\nu`. The initial condition has the form of circles

    .. math::
        u({\bf x}, 0) = \tanh\left(\frac{r - \sqrt{x_i^2 + y_j^2}}{\sqrt{2}\varepsilon}\right)

    for :math:`i, j=0,..,N-1`, where :math:`N` is the number of spatial grid points. For time-stepping, a special AC-splitting
    is used here to get another kind of *semi-implicit* treatment of the problem: The term :math:`\Delta u - \frac{1}{\varepsilon^2} u^{\nu + 1}`
    is handled implicitly and the nonlinear system including this part will be solved by Newton. :math:`\frac{1}{\varepsilon^2} u`
    is solved by a linear solver provided by a ``SciPy`` routine.
    """

    dtype_f = comp2_mesh

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
        ------
        me : dtype_u
            The solution as mesh.
        """

        u = self.dtype_u(u0).flatten()
        z = self.dtype_u(self.init, val=0.0).flatten()
        nu = self.nu
        eps2 = self.eps**2

        Id = sp.eye(self.nvars[0] * self.nvars[1])

        # start newton iteration
        n = 0
        res = 99
        while n < self.newton_maxiter:
            # form the function g with g(u) = 0
            g = u - factor * (self.A.dot(u) - 1.0 / eps2 * u ** (nu + 1)) - rhs.flatten()

            # if g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)

            if res < self.newton_tol:
                break

            # assemble dg
            dg = Id - factor * (self.A - 1.0 / eps2 * sp.diags(((nu + 1) * u**nu), offsets=0))

            # newton update: u1 = u0 - g/dg
            # u -= spsolve(dg, g)
            u -= cg(
                dg,
                g,
                x0=z,
                tol=self.lin_tol,
                atol=0,
            )[0]
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
