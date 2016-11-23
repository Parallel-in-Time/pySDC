import numpy as np
from scipy.sparse.linalg import gmres

from pySDC.implementations.problem_classes.boussinesq_helpers.build2DFDMatrix import get2DMesh
from pySDC.implementations.problem_classes.boussinesq_helpers.buildBoussinesq2DMatrix import getBoussinesq2DMatrix
from pySDC.implementations.problem_classes.boussinesq_helpers.buildBoussinesq2DMatrix import getBoussinesq2DUpwindMatrix
from pySDC.implementations.problem_classes.boussinesq_helpers.unflatten import unflatten
from pySDC.implementations.problem_classes.boussinesq_helpers.helper_classes import Callback, logging

from pySDC.core.Problem import ptype
from pySDC.core.Errors import ParameterError


# noinspection PyUnusedLocal
class boussinesq_2d_imex(ptype):
    """
    Example implementing the 2D Boussinesq equation for different boundary conditions
    """

    def __init__(self, problem_params, dtype_u, dtype_f):
        """
        Initialization routine

        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: particle data type (will be passed parent class)
            dtype_f: acceleration data type (will be passed parent class)
        """

        # these parameters will be used later, so assert their existence
        essential_keys = ['nvars', 'c_s', 'u_adv', 'Nfreq', 'x_bounds', 'z_bounds', 'order_upw', 'order',
                          'gmres_maxiter', 'gmres_restart', 'gmres_tol_limit']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(boussinesq_2d_imex, self).__init__(problem_params['nvars'], dtype_u, dtype_f, problem_params)

        self.N = [self.params.nvars[1], self.params.nvars[2]]

        self.bc_hor = [['periodic', 'periodic'], ['periodic', 'periodic'], ['periodic', 'periodic'],
                       ['periodic', 'periodic']]
        self.bc_ver = [['neumann', 'neumann'], ['dirichlet', 'dirichlet'], ['dirichlet', 'dirichlet'],
                       ['neumann', 'neumann']]

        self.xx, self.zz, self.h = get2DMesh(self.N, self.params.x_bounds, self.params.z_bounds,
                                             self.bc_hor[0], self.bc_ver[0])

        self.Id, self.M = getBoussinesq2DMatrix(self.N, self.h, self.bc_hor, self.bc_ver, self.params.c_s,
                                                self.params.Nfreq, self.params.order)
        self.D_upwind = getBoussinesq2DUpwindMatrix(self.N, self.h[0], self.params.u_adv, self.params.order_upw)

        self.gmres_logger = logging()

    def solve_system(self, rhs, factor, u0, t):
        """
        Simple linear solver for (I-dtA)u = rhs using GMRES

        Args:
            rhs (dtype_f): right-hand side for the nonlinear system
            factor (float): abbrev. for the node-to-node stepsize (or any other factor required)
            u0 (dtype_u): initial guess for the iterative solver (not used here so far)
            t (float): current time (e.g. for time-dependent BCs)

        Returns:
            dtype_u: solution as mesh
        """

        b = rhs.values.flatten()
        cb = Callback()

        sol, info = gmres(self.Id - factor * self.M, b, x0=u0.values.flatten(), tol=self.params.gmres_tol_limit,
                          restart=self.params.gmres_restart, maxiter=self.params.gmres_maxiter, callback=cb)
        # If this is a dummy call with factor==0.0, do not log because it should not be counted as a solver call
        if factor != 0.0:
            self.gmres_logger.add(cb.getcounter())
        me = self.dtype_u(self.init)
        me.values = unflatten(sol, 4, self.N[0], self.N[1])

        return me

    def __eval_fexpl(self, u, t):
        """
        Helper routine to evaluate the explicit part of the RHS

        Args:
            u (dtype_u): current values (not used here)
            t (float): current time

        Returns:
            explicit part of RHS
        """

        # Evaluate right hand side
        fexpl = self.dtype_u(self.init)
        temp = u.values.flatten()
        temp = self.D_upwind.dot(temp)
        fexpl.values = unflatten(temp, 4, self.N[0], self.N[1])

        return fexpl

    def __eval_fimpl(self, u, t):
        """
        Helper routine to evaluate the implicit part of the RHS

        Args:
            u (dtype_u): current values
            t (float): current time (not used here)

        Returns:
            implicit part of RHS
        """

        temp = u.values.flatten()
        temp = self.M.dot(temp)
        fimpl = self.dtype_u(self.init)
        fimpl.values = unflatten(temp, 4, self.N[0], self.N[1])

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
        Routine to compute the exact solution at time t

        Args:
            t (float): current time

        Returns:
            dtype_u: exact solution
        """

        dtheta = 0.01
        H = 10.0
        a = 5.0
        x_c = -50.0

        me = self.dtype_u(self.init)
        me.values[0, :, :] = 0.0 * self.xx
        me.values[1, :, :] = 0.0 * self.xx
        # me.values[2,:,:] = 0.0*self.xx
        # me.values[3,:,:] = np.exp(-0.5*(self.xx-0.0)**2.0/0.15**2.0)*np.exp(-0.5*(self.zz-0.5)**2/0.15**2)
        # me.values[2,:,:] = np.exp(-0.5*(self.xx-0.0)**2.0/0.05**2.0)*np.exp(-0.5*(self.zz-0.5)**2/0.2**2)
        me.values[2, :, :] = dtheta * np.sin(np.pi * self.zz / H) / (1.0 + np.square(self.xx - x_c) / (a * a))
        me.values[3, :, :] = 0.0 * self.xx
        return me
