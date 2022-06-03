import logging
import numpy as np
from scipy.special import factorial

from pySDC.helpers.pysdc_helper import FrozenClass


# short helper class to add params as attributes
class _Pars(FrozenClass):
    def __init__(self, pars):

        for k, v in pars.items():
            setattr(self, k, v)

        self._freeze()


class ptype(object):
    """
    Prototype class for problems, just defines the attributes essential to get started

    Attributes:
        logger: custom logger for problem-related logging
        params (__Pars): parameter object containing the custom parameters passed by the user
        init: number of degrees-of-freedom (whatever this may represent)
        dtype_u: variable data type
        dtype_f: RHS data type
    """

    def __init__(self, init, dtype_u, dtype_f, params):
        """
        Initialization routine

        Args:
            init: number of degrees-of-freedom (whatever this may represent)
            dtype_u: variable data type
            dtype_f: RHS data type
            params (dict): set or parameters
        """

        self.params = _Pars(params)

        # set up logger
        self.logger = logging.getLogger('problem')

        # pass initialization parameter and data types
        self.init = init
        self.dtype_u = dtype_u
        self.dtype_f = dtype_f

    def eval_f(self, u, t):
        """
        Abstract interface to RHS computation of the ODE
        """
        raise NotImplementedError('ERROR: problem has to implement eval_f(self, u, t)')

    def apply_mass_matrix(self, u):
        """
        Abstract interface to apply mass matrix (only needed for FEM)
        """
        raise NotImplementedError('ERROR: if you want a mass matrix, implement apply_mass_matrix(u)')


def get_finite_difference_stencil(derivative, order, type=None, steps=None):
    """
    Derive general finite difference stencils from Taylor expansions
    """
    if steps is not None:
        n = len(steps)
    elif type == 'center':
        n = order + derivative - (derivative + 1) % 2 // 1
        steps = np.arange(n) - (n) // 2
    elif type == 'forward':
        n = order + derivative
        steps = np.arange(n)
    elif type == 'backward':
        n = order + derivative
        steps = -np.arange(n)
    else:
        raise ValueError(f'Stencil must be of type "center", "forward" or "backward", not {type}. If you want something\
else, you can also give specific steps.')

    # the index of the position around which we Taylor expand
    zero_pos = np.argmin(abs(steps)) + 1

    # make a matrix that contains the Taylor coefficients
    A = np.zeros((n, n))
    idx = np.arange(n)
    inv_facs = 1. / factorial(idx)
    for i in range(0, n):
        A[i, :] = steps**idx[i] * inv_facs[i]

    # make a right hand side vector that is zero everywhere except at the postition of the desired derivative
    sol = np.zeros(n)
    sol[derivative] = 1.

    # solve the linear system for the finite difference coefficients
    coeff = np.linalg.solve(A, sol)

    return coeff, zero_pos, steps
