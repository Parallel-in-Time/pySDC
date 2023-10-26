#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description
-----------

Module containing the base Problem class for pySDC
"""

import logging

from pySDC.core.Common import RegisterParams


class WorkCounter(object):
    """
    Utility class for counting iterations.

    Contains one attribute `niter` initialized to zero during
    instantiation, which can be incremented by calling object as
    a function, e.g

    >>> count = WorkCounter()  # => niter = 0
    >>> count()                # => niter = 1
    >>> count()                # => niter = 2
    """

    def __init__(self):
        self.niter = 0

    def __call__(self, *args, **kwargs):
        # *args and **kwargs are necessary for gmres
        self.niter += 1

    def decrement(self):
        self.niter -= 1


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

    logger = logging.getLogger('problem')
    dtype_u = None
    dtype_f = None

    def __init__(self, init):
        self.work_counters = {}  # Dictionary to store WorkCounter objects
        self.init = init  # Initialization parameter to instantiate data types

    @property
    def u_init(self):
        """Generate a data variable for u"""
        return self.dtype_u(self.init)

    @property
    def f_init(self):
        """Generate a data variable for RHS"""
        return self.dtype_f(self.init)

    @classmethod
    def get_default_sweeper_class(cls):
        raise NotImplementedError(f'No default sweeper class implemented for {cls} problem!')

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

    def apply_mass_matrix(self, u):  # pragma: no cover
        """Default mass matrix : identity"""
        return u

    def generate_scipy_reference_solution(self, eval_rhs, t, u_init=None, t_init=None, **kwargs):
        """
        Compute a reference solution using `scipy.solve_ivp` with very small tolerances.
        Keep in mind that scipy needs the solution to be a one dimensional array. If you are solving something higher
        dimensional, you need to make sure the function `eval_rhs` takes a flattened one-dimensional version as an input
        and output, but reshapes to whatever the problem needs for evaluation.

        The keyword arguments will be passed to `scipy.solve_ivp`. You should consider passing `method='BDF'` for stiff
        problems and to accelerate that you can pass a function that evaluates the Jacobian with arguments `jac(t, u)`
        as `jac=jac`.

        Args:
            eval_rhs (function): Function evaluate the full right hand side. Must have signature `eval_rhs(float: t, numpy.1darray: u)`
            t (float): current time
            u_init (pySDC.implementations.problem_classes.Lorenz.dtype_u): initial conditions for getting the exact solution
            t_init (float): the starting time

        Returns:
            numpy.ndarray: Reference solution
        """
        import numpy as np
        from scipy.integrate import solve_ivp

        tol = 100 * np.finfo(float).eps
        u_init = self.u_exact(t=0) if u_init is None else u_init * 1.0
        t_init = 0 if t_init is None else t_init

        u_shape = u_init.shape
        return (
            solve_ivp(eval_rhs, (t_init, t), u_init.flatten(), rtol=tol, atol=tol, **kwargs).y[:, -1].reshape(u_shape)
        )
