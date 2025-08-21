#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions building 1D periodic problems with Dedalus
"""
import numpy as np
import dedalus.public as d3


def buildAdvDiffProblem(nX, listK, nu=0):
    r"""
    Build an 1D periodic advection-diffusion problem

    .. math::
        \frac{\partial u}{\partial t}
        + \frac{\partial u}{\partial x}
        - \nu\frac{\partial^2 u}{\partial x^2}
        = 0

    for :math:`x \in [0, 2\pi]` with Dedalus space discretization on :math:`N_x` points,
    where all linear terms (left of = sign) are treated implicitly.
    It uses an initial solution of the form :

    .. math::
        u_0 = \sum_{i=0}^{N-1} \cos(k[i]x)

    with the list of :math:`k[...]` given as argument.

    Parameters
    ----------
    nX : int
        Number of point in space :math:`N_x`.
    listK : list[int]
        List of :math:`N` sinusoidal frequencies `k` with unitary amplitude
        used to create the initial solution.
    nu : float, optional
        Diffusion coefficient :math:`\nu`. The default is 0.

    Returns
    -------
    pData: dict
        Dictionary containing :

        - `problem` : the Dedalus problem
        - `x` : the x grid (1D np.ndarray)
        - `u` : the Dedalus field object for the solution
        - `u0` : the Dedalus field object for the initial solution
    """
    # -- build space grid
    coords = d3.CartesianCoordinates('x')
    dist = d3.Distributor(coords, dtype=np.float64)
    xbasis = d3.RealFourier(coords['x'], size=nX, bounds=(0, 2*np.pi))
    u = dist.Field(name='u', bases=xbasis)

    # -- build problem
    dx = lambda f: d3.Differentiate(f, coords['x'])
    problem = d3.IVP([u], namespace=locals())
    problem.add_equation("dt(u) + dx(u) - nu*dx(dx(u)) = 0")

    # -- initial solution
    x = xbasis.local_grid(dist, scale=1)
    u0 = np.sum([np.cos(k*x) for k in listK], axis=0)
    np.copyto(u['g'], u0)
    u0 = u.copy()   # store initial field into a copy

    pData = {
        "problem": problem,
        "x": x,
        "u": u,
        "u0": u0
        }
    return pData


def buildKdVBurgerProblem(nX, xEnd, nu=1e-4, b=2e-4, n=20, dtype=np.float64, dealias=3/2):
    r"""
    Build an 1D periodic KdV Burger problem

    .. math::
        \frac{\partial u}{\partial t}
        - \nu\frac{\partial^2 u}{\partial x^2}
        - b\frac{\partial^3 u}{\partial x^3}
        = u\frac{\partial u}{\partial x}

    for :math:`x \in [0, L_x]` with Dedalus space discretization on :math:`N_x` points,
    where all linear terms (left of = sign) are treated implicitly.
    It uses an initial solution of the form :

    .. math::
        u_0 = \frac{1}{2n}\ln\left(
            1 + \frac{\cosh(n)^2}{\cosh(n(x-0.2L_x))^2}
            \right)

    with :math:`n[...]` given as argument.

    Parameters
    ----------
    nX : int
        Number of point in space :math:`N_x`.
    xEnd : TYPE
        Length of the domain :math:`L_x`.
    nu : float, optional
        Diffusion coefficient :math:`\nu`. The default is 1e-4.
    b : float, optional
        Hyperdiffusion coefficient :math:`b`.. The default is 2e-4.
    n : int, optional
        Initial solution parameter :math:`n`. The default is 20.
    dtype : np.dtype, optional
        Dtype used for the solution. The default is np.float64.
    dealias : float, optional
        Dealiasing coefficient for space discretization. The default is 3/2.

    Returns
    -------
    pData : dict
        Dictionary containing :

        - `problem` : the Dedalus problem
        - `x` : the x grid (1D np.ndarray)
        - `u` : the Dedalus field object for the solution
    """
    # -- build space grid
    xcoord = d3.Coordinate('x')
    dist = d3.Distributor(xcoord, dtype=dtype)
    xbasis = d3.RealFourier(xcoord, size=nX, bounds=(0, xEnd), dealias=dealias)
    u = dist.Field(name='u', bases=xbasis)

    # -- build problem
    dx = lambda A: d3.Differentiate(A, xcoord)
    problem = d3.IVP([u], namespace=locals())
    problem.add_equation("dt(u) - nu*dx(dx(u)) - b*dx(dx(dx(u))) = - u*dx(u)")

    # -- initial solution
    x = dist.local_grid(xbasis)
    u['g'] = np.log(1 + np.cosh(n)**2/np.cosh(n*(x-0.2*xEnd))**2) / (2*n)

    pData = {
        "problem": problem,
        "x": x,
        "u": u
        }
    return pData
