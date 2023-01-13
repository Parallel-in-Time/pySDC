import logging


class ReadOnlyError(Exception):
    """Exception class thrown when setting read-only parameters"""

    def __init__(self, name):
        super().__init__(f'cannot set read-only attribute {name}')


class RegisterParams(object):
    """Base class to register parameters"""

    def _register(self, *names, readOnly=False):
        if readOnly:
            if not hasattr(self, '_readOnly'):
                self._readOnly = set()
            self._readOnly = self._readOnly.union(names)
        else:
            if not hasattr(self, '_parNames'):
                self._parNames = set()
            self._parNames = self._parNames.union(names)

    @property
    def params(self):
        return {name: getattr(self, name) for name in self._readOnly.union(self._parNames)}

    def __setattr__(self, name, value):
        try:
            if name in self._readOnly:
                raise ReadOnlyError(name)
        except AttributeError:
            pass
        super().__setattr__(name, value)


class ptype(RegisterParams):
    """
    Prototype class for problems, just defines the attributes essential to get started

    Attributes:
        logger: custom logger for problem-related logging
        params (__Pars): parameter object containing the custom parameters passed by the user
        init: number of degrees-of-freedom (whatever this may represent)
        dtype_u: variable data type
        dtype_f: RHS data type
    """

    def __init__(self, init, dtype_u, dtype_f):
        """
        Initialization routine.
        Add the problem parameters as keyword arguments.

        Args:
            init: number of degrees-of-freedom (whatever this may represent)
            dtype_u: variable data type
            dtype_f: RHS data type
        """
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
