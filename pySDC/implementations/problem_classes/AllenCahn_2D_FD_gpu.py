import time

import numpy as np
import cupy as cp
import cupyx.scipy.sparse as csp
from cupyx.scipy.sparse.linalg import spsolve, gmres, cg, minres


from pySDC.core.Errors import ParameterError, ProblemError
from pySDC.core.Problem import ptype
# from pySDC.implementations.datatype_classes.cupy_mesh import cupy_mesh, imex_cupy_mesh, comp2_cupy_mesh
from pySDC.implementations.datatype_classes.cupy_mesh import cupy_mesh, imex_cupy_mesh, comp2_cupy_mesh

# http://www.personal.psu.edu/qud2/Res/Pre/dz09sisc.pdf


# noinspection PyUnusedLocal
class allencahn_fullyimplicit(ptype):
    """
    Example implementing the Allen-Cahn equation in 2D with finite differences and periodic BC

    Attributes:
        A: second-order FD discretization of the 2D laplace operator
        dx: distance between two spatial nodes (same for both directions)
    """

    def __init__(self, problem_params, dtype_u=cupy_mesh, dtype_f=cupy_mesh):
        """
        Initialization routine

        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: mesh data type (will be passed parent class)
            dtype_f: mesh data type (will be passed parent class)
        """

        # these parameters will be used later, so assert their existence
        essential_keys = ['nvars', 'nu', 'eps', 'newton_maxiter', 'newton_tol', 'lin_tol', 'lin_maxiter', 'radius']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        # we assert that nvars looks very particular here.. this will be necessary for coarsening in space later on
        if len(problem_params['nvars']) != 2:
            raise ProblemError('this is a 2d example, got %s' % problem_params['nvars'])
        if problem_params['nvars'][0] != problem_params['nvars'][1]:
            raise ProblemError('need a square domain, got %s' % problem_params['nvars'])
        if problem_params['nvars'][0] % 2 != 0:
            raise ProblemError('the setup requires nvars = 2^p per dimension')

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        # super(allencahn_fullyimplicit, self).__init__((problem_params['nvars'], None, np.dtype('float64')),
        #                                               dtype_u, dtype_f, problem_params)
        super(allencahn_fullyimplicit, self).__init__((problem_params['nvars'], None, cp.dtype('float64')),
                                                      dtype_u, dtype_f, problem_params)

        # compute dx and get discretization matrix A
        self.dx = 1.0 / self.params.nvars[0]
        self.A = self.__get_A(self.params.nvars, self.dx)
        self.xvalues = cp.array([i * self.dx - 0.5 for i in range(self.params.nvars[0])])

        self.newton_itercount = 0
        self.lin_itercount = 0
        self.newton_ncalls = 0
        self.lin_ncalls = 0

    @staticmethod
    def __get_A(N, dx):
        """
        Helper function to assemble FD matrix A in sparse format

        Args:
            N (list): number of dofs
            dx (float): distance between two spatial nodes

        Returns:
            scipy.sparse.csc_matrix: matrix A in CSC format
        """

        stencil = cp.asarray([-2, 1])
        A = stencil[0] * csp.eye(N[0], format='csr')  # TODO: is this the right format? Was: csc
        for i in range(1, len(stencil)):
            A += stencil[i] * csp.eye(N[0], k=-i, format='csr')
            A += stencil[i] * csp.eye(N[0], k=+i, format='csr')
            A += stencil[i] * csp.eye(N[0], k=N[0] - i, format='csr')
            A += stencil[i] * csp.eye(N[0], k=-N[0] + i, format='csr')
        A = csp.kron(A, csp.eye(N[0])) + csp.kron(csp.eye(N[1]), A)
        A *= 1.0 / (dx ** 2)
        return A

    # noinspection PyTypeChecker
    def solve_system(self, rhs, factor, u0, t):
        """
        Simple Newton solver

        Args:
            rhs (dtype_f): right-hand side for the nonlinear system
            factor (float): abbrev. for the node-to-node stepsize (or any other factor required)
            u0 (dtype_u): initial guess for the iterative solver
            t (float): current time (required here for the BC)

        Returns:
            dtype_u: solution u
        """

        u = self.dtype_u(u0).flatten()
        z = self.dtype_u(self.init, val=0.0).flatten()
        nu = self.params.nu
        eps2 = self.params.eps ** 2

        Id = csp.eye(self.params.nvars[0] * self.params.nvars[1])

        # start newton iteration
        n = 0
        res = 99
        while n < self.params.newton_maxiter:

            # form the function g with g(u) = 0
            g = u - factor * (self.A.dot(u) + 1.0 / eps2 * u * (1.0 - u ** nu)) - rhs.flatten()

            # if g is close to 0, then we are done
            res = cp.linalg.norm(g, np.inf)

            if res < self.params.newton_tol:
                break

            # assemble dg
            dg = Id - factor * (self.A + 1.0 / eps2 * csp.diags((1.0 - (nu + 1) * u ** nu), offsets=0))

            # newton update: u1 = u0 - g/dg
            # u -= spsolve(dg, g)
            u -= cg(dg, g, x0=z, tol=self.params.lin_tol)[0]
            # increase iteration count
            n += 1
            # print(n, res)

        # if n == self.params.newton_maxiter:
        #     raise ProblemError('Newton did not converge after %i iterations, error is %s' % (n, res))

        me = self.dtype_u(self.init)
        me[:] = u.reshape(self.params.nvars)

        self.newton_ncalls += 1
        self.newton_itercount += n

        return me

    def eval_f(self, u, t):
        """
        Routine to evaluate the RHS

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            dtype_f: the RHS
        """
        f = self.dtype_f(self.init)
        v = u.flatten()
        f[:] = (self.A.dot(v) + 1.0 / self.params.eps ** 2 * v * (1.0 - v ** self.params.nu)).reshape(self.params.nvars)

        return f

    def u_exact(self, t):
        """
        Routine to compute the exact solution at time t

        Args:
            t (float): current time

        Returns:
            dtype_u: exact solution
        """

        assert t == 0, 'ERROR: u_exact only valid for t=0'
        me = self.dtype_u(self.init, val=0.0)
        mx, my = cp.meshgrid(self.xvalues, self.xvalues)
        me[:] = cp.tanh((self.params.radius - cp.sqrt(mx ** 2 + my ** 2)) / (cp.sqrt(2) * self.params.eps))
        # print(type(me))
        return me


# noinspection PyUnusedLocal
class allencahn_semiimplicit(allencahn_fullyimplicit):
    """
    Example implementing the Allen-Cahn equation in 2D with finite differences, SDC standard splitting
    """

    def __init__(self, problem_params, dtype_u=cupy_mesh, dtype_f=imex_cupy_mesh):
        """
        Initialization routine

        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: mesh data type (will be passed parent class)
            dtype_f: mesh data type with implicit and explicit parts (will be passed parent class)
        """

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(allencahn_semiimplicit, self).__init__(problem_params, dtype_u, dtype_f)
        self.f_im = 0
        self.f_ex = 0

    def eval_f(self, u, t):
        """
        Routine to evaluate the RHS

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            dtype_f: the RHS
        """
        f = self.dtype_f(self.init)
        start = time.perf_counter()
        v = u.flatten()
        f.impl[:] = self.A.dot(v).reshape(self.params.nvars)
        ende = time.perf_counter()
        self.f_im += ende - start
        start = time.perf_counter()
        f.expl[:] = (1.0 / self.params.eps ** 2 * v * (1.0 - v ** self.params.nu)).reshape(self.params.nvars)
        ende = time.perf_counter()
        self.f_ex += ende - start
        return f

    def solve_system(self, rhs, factor, u0, t):
        """
        Simple linear solver for (I-factor*A)u = rhs

        Args:
            rhs (dtype_f): right-hand side for the linear system
            factor (float): abbrev. for the local stepsize (or any other factor required)
            u0 (dtype_u): initial guess for the iterative solver
            t (float): current time (e.g. for time-dependent BCs)

        Returns:
            dtype_u: solution as mesh
        """

        class context:
            num_iter = 0

        def callback(xk):
            context.num_iter += 1
            return context.num_iter

        me = self.dtype_u(self.init)

        Id = csp.eye(self.params.nvars[0] * self.params.nvars[1])

        me[:] = cg(Id - factor * self.A, rhs.flatten(), x0=u0.flatten(), tol=self.params.lin_tol,
                   maxiter=self.params.lin_maxiter, callback=callback)[0].reshape(self.params.nvars)

        self.lin_ncalls += 1
        self.lin_itercount += context.num_iter

        return me


# noinspection PyUnusedLocal
class allencahn_semiimplicit_v2(allencahn_fullyimplicit):
    """
    Example implementing the Allen-Cahn equation in 2D with finite differences, AC splitting
    """

    def __init__(self, problem_params, dtype_u=cupy_mesh, dtype_f=imex_cupy_mesh):
        """
        Initialization routine

        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: mesh data type (will be passed parent class)
            dtype_f: mesh data type with implicit and explicit parts (will be passed parent class)
        """

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(allencahn_semiimplicit_v2, self).__init__(problem_params, dtype_u, dtype_f)

    def eval_f(self, u, t):
        """
        Routine to evaluate the RHS

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            dtype_f: the RHS
        """
        f = self.dtype_f(self.init)
        v = u.flatten()
        f.impl[:] = (self.A.dot(v) - 1.0 / self.params.eps ** 2 * v ** (self.params.nu + 1)).reshape(self.params.nvars)
        f.expl[:] = (1.0 / self.params.eps ** 2 * v).reshape(self.params.nvars)

        return f

    def solve_system(self, rhs, factor, u0, t):
        """
        Simple Newton solver

        Args:
            rhs (dtype_f): right-hand side for the nonlinear system
            factor (float): abbrev. for the node-to-node stepsize (or any other factor required)
            u0 (dtype_u): initial guess for the iterative solver
            t (float): current time (required here for the BC)

        Returns:
            dtype_u: solution u
        """

        u = self.dtype_u(u0).flatten()
        z = self.dtype_u(self.init, val=0.0).flatten()
        nu = self.params.nu
        eps2 = self.params.eps ** 2

        Id = csp.eye(self.params.nvars[0] * self.params.nvars[1])

        # start newton iteration
        n = 0
        res = 99
        while n < self.params.newton_maxiter:

            # form the function g with g(u) = 0
            g = u - factor * (self.A.dot(u) - 1.0 / eps2 * u ** (nu + 1)) - rhs.flatten()

            # if g is close to 0, then we are done
            res = cp.linalg.norm(g, np.inf)

            if res < self.params.newton_tol:
                break

            # assemble dg
            dg = Id - factor * (self.A - 1.0 / eps2 * csp.diags(((nu + 1) * u ** nu), offsets=0))

            # newton update: u1 = u0 - g/dg
            # u -= spsolve(dg, g)
            u -= cg(dg, g, x0=z, tol=self.params.lin_tol)[0]
            # increase iteration count
            n += 1
            # print(n, res)

        # if n == self.params.newton_maxiter:
        #     raise ProblemError('Newton did not converge after %i iterations, error is %s' % (n, res))

        me = self.dtype_u(self.init)
        me[:] = u.reshape(self.params.nvars)

        self.newton_ncalls += 1
        self.newton_itercount += n

        return me


# noinspection PyUnusedLocal
class allencahn_multiimplicit(allencahn_fullyimplicit):
    """
    Example implementing the Allen-Cahn equation in 2D with finite differences, SDC standard splitting
    """

    def __init__(self, problem_params, dtype_u=cupy_mesh, dtype_f=comp2_cupy_mesh):
        """
        Initialization routine

        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: mesh data type (will be passed parent class)
            dtype_f: mesh data type with 2 components (will be passed parent class)
        """

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(allencahn_multiimplicit, self).__init__(problem_params, dtype_u, dtype_f)

    def eval_f(self, u, t):
        """
        Routine to evaluate the RHS

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            dtype_f: the RHS
        """
        f = self.dtype_f(self.init)
        v = u.flatten()
        f.comp1[:] = self.A.dot(v).reshape(self.params.nvars)
        f.comp2[:] = (1.0 / self.params.eps ** 2 * v * (1.0 - v ** self.params.nu)).reshape(self.params.nvars)

        return f

    def solve_system_1(self, rhs, factor, u0, t):
        """
        Simple linear solver for (I-factor*A)u = rhs

        Args:
            rhs (dtype_f): right-hand side for the linear system
            factor (float): abbrev. for the local stepsize (or any other factor required)
            u0 (dtype_u): initial guess for the iterative solver
            t (float): current time (e.g. for time-dependent BCs)

        Returns:
            dtype_u: solution as mesh
        """

        class context:
            num_iter = 0

        def callback(xk):
            context.num_iter += 1
            return context.num_iter

        me = self.dtype_u(self.init)

        Id = csp.eye(self.params.nvars[0] * self.params.nvars[1])

        me[:] = cg(Id - factor * self.A, rhs.flatten(), x0=u0.flatten(), tol=self.params.lin_tol,
                   maxiter=self.params.lin_maxiter, callback=callback)[0].reshape(self.params.nvars)

        self.lin_ncalls += 1
        self.lin_itercount += context.num_iter

        return me

    def solve_system_2(self, rhs, factor, u0, t):
        """
        Simple Newton solver

        Args:
            rhs (dtype_f): right-hand side for the nonlinear system
            factor (float): abbrev. for the node-to-node stepsize (or any other factor required)
            u0 (dtype_u): initial guess for the iterative solver
            t (float): current time (required here for the BC)

        Returns:
            dtype_u: solution u
        """

        u = self.dtype_u(u0).flatten()
        z = self.dtype_u(self.init, val=0.0).flatten()
        nu = self.params.nu
        eps2 = self.params.eps ** 2

        Id = csp.eye(self.params.nvars[0] * self.params.nvars[1])

        # start newton iteration
        n = 0
        res = 99
        while n < self.params.newton_maxiter:

            # form the function g with g(u) = 0
            g = u - factor * (1.0 / eps2 * u * (1.0 - u ** nu)) - rhs.flatten()

            # if g is close to 0, then we are done
            res = cp.linalg.norm(g, np.inf)

            if res < self.params.newton_tol:
                break

            # assemble dg
            dg = Id - factor * (1.0 / eps2 * csp.diags((1.0 - (nu + 1) * u ** nu), offsets=0))

            # newton update: u1 = u0 - g/dg
            # u -= spsolve(dg, g)
            u -= cg(dg, g, x0=z, tol=self.params.lin_tol)[0]
            # increase iteration count
            n += 1
            # print(n, res)

        # if n == self.params.newton_maxiter:
        #     raise ProblemError('Newton did not converge after %i iterations, error is %s' % (n, res))

        me = self.dtype_u(self.init)
        me[:] = u.reshape(self.params.nvars)

        self.newton_ncalls += 1
        self.newton_itercount += n

        return me


# noinspection PyUnusedLocal
class allencahn_multiimplicit_v2(allencahn_fullyimplicit):
    """
    Example implementing the Allen-Cahn equation in 2D with finite differences, AC splitting
    """

    def __init__(self, problem_params, dtype_u=cupy_mesh, dtype_f=comp2_cupy_mesh):
        """
        Initialization routine

        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: mesh data type (will be passed parent class)
            dtype_f: mesh data type with 2 components (will be passed parent class)
        """

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(allencahn_multiimplicit_v2, self).__init__(problem_params, dtype_u, dtype_f)

    def eval_f(self, u, t):
        """
        Routine to evaluate the RHS

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            dtype_f: the RHS
        """
        f = self.dtype_f(self.init)
        v = u.flatten()
        f.comp1[:] = (self.A.dot(v) - 1.0 / self.params.eps ** 2 * v ** (self.params.nu + 1)).reshape(self.params.nvars)
        f.comp2[:] = (1.0 / self.params.eps ** 2 * v).reshape(self.params.nvars)

        return f

    def solve_system_1(self, rhs, factor, u0, t):
        """
        Simple Newton solver

        Args:
            rhs (dtype_f): right-hand side for the nonlinear system
            factor (float): abbrev. for the node-to-node stepsize (or any other factor required)
            u0 (dtype_u): initial guess for the iterative solver
            t (float): current time (required here for the BC)

        Returns:
            dtype_u: solution u
        """

        u = self.dtype_u(u0).flatten()
        z = self.dtype_u(self.init, val=0.0).flatten()
        nu = self.params.nu
        eps2 = self.params.eps ** 2

        Id = csp.eye(self.params.nvars[0] * self.params.nvars[1])

        # start newton iteration
        n = 0
        res = 99
        while n < self.params.newton_maxiter:

            # form the function g with g(u) = 0
            g = u - factor * (self.A.dot(u) - 1.0 / eps2 * u ** (nu + 1)) - rhs.flatten()

            # if g is close to 0, then we are done
            res = cp.linalg.norm(g, np.inf)

            if res < self.params.newton_tol:
                break

            # assemble dg
            dg = Id - factor * (self.A - 1.0 / eps2 * csp.diags(((nu + 1) * u ** nu), offsets=0))

            # newton update: u1 = u0 - g/dg
            # u -= spsolve(dg, g)
            u -= cg(dg, g, x0=z, tol=self.params.lin_tol)[0]
            # increase iteration count
            n += 1
            # print(n, res)

        # if n == self.params.newton_maxiter:
        #     raise ProblemError('Newton did not converge after %i iterations, error is %s' % (n, res))

        me = self.dtype_u(self.init)
        me[:] = u.reshape(self.params.nvars)

        self.newton_ncalls += 1
        self.newton_itercount += n

        return me

    def solve_system_2(self, rhs, factor, u0, t):
        """
        Simple linear solver for (I-factor*A)u = rhs

        Args:
            rhs (dtype_f): right-hand side for the linear system
            factor (float): abbrev. for the local stepsize (or any other factor required)
            u0 (dtype_u): initial guess for the iterative solver
            t (float): current time (e.g. for time-dependent BCs)

        Returns:
            dtype_u: solution as mesh
        """

        me = self.dtype_u(self.init)

        me[:] = (1.0 / (1.0 - factor * 1.0 / self.params.eps ** 2) * rhs).reshape(self.params.nvars)
        return me
