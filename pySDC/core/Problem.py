#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description
-----------

Module containing the base Problem class for pySDC
"""

import logging

from pySDC.core.Common import RegisterParams


class ptype(RegisterParams):
    """
    Prototype class for problems, just defines the attributes essential to get started.

    Parameters
    ----------
    init : list of args
        Argument(s) used to initialize data types.
    dtype_u : type
        Variable data type. Should generate a data variable using dtype_u(init).
    dtype_f : type
        RHS data type. Should generate a data variable using dtype_f(init).

    Attributes
    ----------
    logger: logging.Logger
        custom logger for problem-related logging.
    """

    def __init__(self, init, dtype_u, dtype_f):
        # set up logger
        self.logger = logging.getLogger('problem')

        # pass initialization parameter and data types
        self.init = init
        self.dtype_u = dtype_u
        self.dtype_f = dtype_f

    @property
    def u_init(self):
        """Generate a data variable for u"""
        return self.dtype_u(self.init)

    @property
    def f_init(self):
        """Generate a data variable for RHS"""
        return self.dtype_f(self.init)

    def eval_f(self, u, t):
        """
        Abstract interface to RHS computation of the ODE

        Parameters
        ----------
        u : dtype_u
            Current values.
        t : float
            Current time.

        Returns
        -------
        f : dtype_f
            The RHS values.
        """
        raise NotImplementedError('ERROR: problem has to implement eval_f(self, u, t)')

    def apply_mass_matrix(self, u):
        """
        Abstract interface to apply mass matrix (only needed for FEM)
        """
        raise NotImplementedError('ERROR: if you want a mass matrix, implement apply_mass_matrix(u)')

    def generate_scipy_reference_solution(self, eval_rhs, t, u_init=None, t_init=None):
        """
        Compute a reference solution using `scipy.solve_ivp` with very small tolerances.
        Keep in mind that scipy needs the solution to be a one dimensional array. If you are solving something higher
        dimensional, you need to make sure the function `eval_rhs` takes a flattened one-dimensional version as an input
        and output, but reshapes to whatever the problem needs for evaluation.

        Args:
            eval_rhs (function): Function evaluate the full right hand side. Must have signature `eval_rhs(float: t, numpy.1darray: u)`
            t (float): current time
            u_init (pySDC.implementations.problem_classes.Lorenz.dtype_u): initial conditions for getting the exact solution
            t_init (float): the starting time

        Returns:
            numpy.ndarray: exact solution
        """
        import numpy as np
        from scipy.integrate import solve_ivp

        tol = 100 * np.finfo(float).eps
        u_init = self.u_exact(t=0) if u_init is None else u_init
        t_init = 0 if t_init is None else t_init

        u_shape = u_init.shape
        return solve_ivp(eval_rhs, (t_init, t), u_init.flatten(), rtol=tol, atol=tol).y[:, -1].reshape(u_shape)
