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
