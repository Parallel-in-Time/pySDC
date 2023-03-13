import logging

from pySDC.helpers.pysdc_helper import FrozenClass


# short helper class to add params as attributes
class _Pars(FrozenClass):
    def __init__(self, pars):
        for k, v in pars.items():
            setattr(self, k, v)

        self._freeze()


class WorkCounter(object):
    """
    Class for counting iterations
    """

    def __init__(self):
        self.niter = 0

    def __call__(self, *args, **kwargs):
        self.niter += 1


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
        self.work_counters = {}

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
            numpy.ndarray: exact solution
        """
        import numpy as np
        from scipy.integrate import solve_ivp

        tol = 100 * np.finfo(float).eps
        u_init = self.u_exact(t=0) if u_init is None else u_init
        t_init = 0 if t_init is None else t_init

        u_shape = u_init.shape
        return (
            solve_ivp(eval_rhs, (t_init, t), u_init.flatten(), rtol=tol, atol=tol, **kwargs).y[:, -1].reshape(u_shape)
        )
