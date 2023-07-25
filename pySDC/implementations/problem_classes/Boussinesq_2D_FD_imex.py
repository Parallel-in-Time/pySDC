import numpy as np
from scipy.sparse.linalg import gmres

from pySDC.core.Errors import ParameterError
from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh
from pySDC.implementations.problem_classes.boussinesq_helpers.build2DFDMatrix import get2DMesh
from pySDC.implementations.problem_classes.boussinesq_helpers.buildBoussinesq2DMatrix import getBoussinesq2DMatrix
from pySDC.implementations.problem_classes.boussinesq_helpers.buildBoussinesq2DMatrix import getBoussinesq2DUpwindMatrix
from pySDC.implementations.problem_classes.boussinesq_helpers.helper_classes import Callback, logging
from pySDC.implementations.problem_classes.boussinesq_helpers.unflatten import unflatten


# noinspection PyUnusedLocal
class boussinesq_2d_imex(ptype):
    r"""
    This class implements the two-dimensional Boussinesq equations for different boundary conditions with

    .. math::
        \frac{\partial u}{\partial t} + U \frac{\partial u}{\partial x} + \frac{\partial p}{\partial x} = 0,

    .. math::
        \frac{\partial w}{\partial t} + U \frac{\partial w}{\partial x} + \frac{\partial p}{\partial z} = 0,

    .. math::
        \frac{\partial b}{\partial t} + U \frac{\partial b}{\partial x} + N^2 w = 0,

    .. math::
        \frac{\partial p}{\partial t} + U \frac{\partial p}{\partial x} + c^2 (\frac{\partial u}{\partial x} + \frac{\partial w}{\partial z}) = 0.

    They can be derived from the linearized Euler equations by a transformation of variables [1]_.

    Parameters
    ----------
    nvars : list, optional
        List of number of unknowns nvars.
    c_s : float, optional
        Acoustic velocity :math:`c_s`.
    u_adv : float, optional
        Advection speed :math:`U`.
    Nfreq : float, optional
        Stability frequency.
    x_bounds : list, optional
        Domain in x-direction.
    z_bounds : list, optional
        Domain in z-direction.
    order_upwind : int, optional
        Order of upwind scheme for discretization.
    order : int, optional
        Order for discretization.
    gmres_maxiter : int, optional
        Maximum number of iterations for GMRES solver.
    gmres_restart : int, optional
        Number of iterations between restarts in GMRES solver.
    gmres_tol_limit : float, optional
        Tolerance for GMRES solver to terminate.

    Attributes
    ----------
    N : list
        List of number of unknowns nvars.
    bc_hor : list
        Contains type of boundary conditions for both boundaries for both dimensions.
    bc_ver :
        Contains type of boundary conditions for both boundaries for both dimemsions, e.g. 'neumann' or 'dirichlet'.
    xx : np.ndarray
        List of np.ndarrays for mesh in x-direction.
    zz : np.ndarray
        List of np.ndarrays for mesh in z-direction.
    h : float
        Mesh size.
    Id : sp.sparse.eye
        Identity matrix for the equation of appropriate size.
    M : np.ndarray
        Boussinesq 2D Matrix.
    D_upwind : sp.csc_matrix
        Boussinesq 2D Upwind matrix for discretization.
    gmres_logger : object
        Logger for GMRES solver.

    References
    ----------
    .. [1] D. R. Durran. Numerical Methods for Fluid Dynamics. Texts Appl. Math. 32. Springer-Verlag, New York (2010)
        http://dx.doi.org/10.1007/978-1-4419-6412-0.
    """

    dtype_u = mesh
    dtype_f = imex_mesh

    def __init__(
        self,
        nvars=None,
        c_s=0.3,
        u_adv=0.02,
        Nfreq=0.01,
        x_bounds=None,
        z_bounds=None,
        order_upw=5,
        order=4,
        gmres_maxiter=500,
        gmres_restart=10,
        gmres_tol_limit=1e-5,
    ):
        """Initialization routine"""

        if nvars is None:
            nvars = [(4, 300, 30)]

        if x_bounds is None:
            x_bounds = [(-150.0, 150.0)]

        if z_bounds is None:
            z_bounds = [(0.0, 10.0)]

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super().__init__(init=(nvars, None, np.dtype('float64')))
        self._makeAttributeAndRegister(
            'nvars',
            'c_s',
            'u_adv',
            'Nfreq',
            'x_bounds',
            'z_bounds',
            'order_upw',
            'order',
            'gmres_maxiter',
            'gmres_restart',
            'gmres_tol_limit',
            localVars=locals(),
            readOnly=True,
        )

        self.N = [self.nvars[1], self.nvars[2]]

        self.bc_hor = [
            ['periodic', 'periodic'],
            ['periodic', 'periodic'],
            ['periodic', 'periodic'],
            ['periodic', 'periodic'],
        ]
        self.bc_ver = [
            ['neumann', 'neumann'],
            ['dirichlet', 'dirichlet'],
            ['dirichlet', 'dirichlet'],
            ['neumann', 'neumann'],
        ]

        self.xx, self.zz, self.h = get2DMesh(self.N, self.x_bounds, self.z_bounds, self.bc_hor[0], self.bc_ver[0])

        self.Id, self.M = getBoussinesq2DMatrix(
            self.N, self.h, self.bc_hor, self.bc_ver, self.c_s, self.Nfreq, self.order
        )
        self.D_upwind = getBoussinesq2DUpwindMatrix(self.N, self.h[0], self.u_adv, self.order_upw)

        self.gmres_logger = logging()

    def solve_system(self, rhs, factor, u0, t):
        r"""
        Simple linear solver for :math:`(I - factor A) \vec{u} = \vec{rhs}` using GMRES.

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side for the nonlinear system.
        factor : float
            Abbrev. for the node-to-node stepsize (or any other factor required).
        u0 : dtype_u
            Initial guess for the iterative solver (not used here so far).
        t : float
            Current time (e.g. for time-dependent BCs).

        Returns
        -------
        me : dtype_u
            The solution as mesh.
        """

        b = rhs.flatten()
        cb = Callback()

        sol, info = gmres(
            self.Id - factor * self.M,
            b,
            x0=u0.flatten(),
            tol=self.gmres_tol_limit,
            restart=self.gmres_restart,
            maxiter=self.gmres_maxiter,
            atol=0,
            callback=cb,
        )
        # If this is a dummy call with factor==0.0, do not log because it should not be counted as a solver call
        if factor != 0.0:
            self.gmres_logger.add(cb.getcounter())
        me = self.dtype_u(self.init)
        me[:] = unflatten(sol, 4, self.N[0], self.N[1])

        return me

    def __eval_fexpl(self, u, t):
        """
        Helper routine to evaluate the explicit part of the right-hand side.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution.
        t : float
            Current time at which the numerical solution is computed (not used here).

        Returns
        -------
        fexpl : dtype_u
            Explicit part of right-hand side.
        """

        # Evaluate right hand side
        fexpl = self.dtype_u(self.init)
        temp = u.flatten()
        temp = self.D_upwind.dot(temp)
        fexpl[:] = unflatten(temp, 4, self.N[0], self.N[1])

        return fexpl

    def __eval_fimpl(self, u, t):
        """
        Helper routine to evaluate the implicit part of the right-hand side.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution.
        t : float
            Current time at which the numerical solution is computed (not used here).

        Returns
        -------
        fexpl : dtype_u
            Implicit part of right-hand side.
        """

        temp = u.flatten()
        temp = self.M.dot(temp)
        fimpl = self.dtype_u(self.init)
        fimpl[:] = unflatten(temp, 4, self.N[0], self.N[1])

        return fimpl

    def eval_f(self, u, t):
        """
        Routine to evaluate both parts of the RHS

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            dtype_f: the RHS divided into two parts
        """

        f = self.dtype_f(self.init)
        f.impl = self.__eval_fimpl(u, t)
        f.expl = self.__eval_fexpl(u, t)
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

        dtheta = 0.01
        H = 10.0
        a = 5.0
        x_c = -50.0

        me = self.dtype_u(self.init)
        me[0, :, :] = 0.0 * self.xx
        me[1, :, :] = 0.0 * self.xx
        # me[2,:,:] = 0.0*self.xx
        # me[3,:,:] = np.exp(-0.5*(self.xx-0.0)**2.0/0.15**2.0)*np.exp(-0.5*(self.zz-0.5)**2/0.15**2)
        # me[2,:,:] = np.exp(-0.5*(self.xx-0.0)**2.0/0.05**2.0)*np.exp(-0.5*(self.zz-0.5)**2/0.2**2)
        me[2, :, :] = dtheta * np.sin(np.pi * self.zz / H) / (1.0 + np.square(self.xx - x_c) / (a * a))
        me[3, :, :] = 0.0 * self.xx
        return me
