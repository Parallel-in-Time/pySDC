import numpy as np

from pmesh.pm import ParticleMesh

from pySDC.core.Errors import ParameterError, ProblemError
from pySDC.core.Problem import ptype
from pySDC.playgrounds.pmesh.PMESH_datatype import pmesh_datatype, rhs_imex_pmesh


class allencahn_imex(ptype):
    """
    Example implementing Allen-Cahn equation in 2-3D using PMESH for solving linear parts, IMEX time-stepping

    PMESH: https://github.com/rainwoodman/pmesh

    Attributes:
        xvalues: grid points in space
        dx: mesh width
    """

    def __init__(self, problem_params, dtype_u=pmesh_datatype, dtype_f=rhs_imex_pmesh):
        """
        Initialization routine

        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: pmesh data type (will be passed to parent class)
            dtype_f: pmesh data type wuth implicit and explicit parts (will be passed to parent class)
        """

        if 'L' not in problem_params:
            problem_params['L'] = 1.0
        if 'init_type' not in problem_params:
            problem_params['init_type'] = 'circle'
        if 'comm' not in problem_params:
            problem_params['comm'] = None

        # these parameters will be used later, so assert their existence
        essential_keys = ['nvars', 'nu', 'eps', 'L', 'radius']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        if not (isinstance(problem_params['nvars'], tuple) and len(problem_params['nvars']) > 1):
            raise ProblemError('Need at least two dimensions')

        # Creating ParticleMesh structure
        self.pm = ParticleMesh(BoxSize=problem_params['L'], Nmesh=list(problem_params['nvars']), dtype='f8',
                               plan_method='measure', comm=problem_params['comm'])

        # create test RealField to get the local dimensions (there's probably a better way to do that)
        tmp = self.pm.create(type='real')

        # invoke super init, passing the communicator and the local dimensions as init
        super(allencahn_imex, self).__init__(init=(self.pm.comm, tmp.value.shape), dtype_u=dtype_u, dtype_f=dtype_f,
                                             params=problem_params)

        # Need this for diagnostics
        self.dx = self.params.L / problem_params['nvars'][0]
        self.dy = self.params.L / problem_params['nvars'][1]
        self.xvalues = [i * self.dx - problem_params['L'] / 2 for i in range(problem_params['nvars'][0])]
        self.yvalues = [i * self.dy - problem_params['L'] / 2 for i in range(problem_params['nvars'][1])]

    def eval_f(self, u, t):
        """
        Routine to evaluate the RHS

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            dtype_f: the RHS
        """

        def Laplacian(k, v):
            k2 = sum(ki ** 2 for ki in k)
            return -k2 * v

        f = self.dtype_f(self.init)
        tmp_u = self.pm.create(type='real', value=u.values)
        f.impl.values = tmp_u.r2c().apply(Laplacian, out=Ellipsis).c2r(out=Ellipsis).value
        if self.params.eps > 0:
            f.expl.values = 1.0 / self.params.eps ** 2 * u.values * (1.0 - u.values ** self.params.nu)
        return f

    def solve_system(self, rhs, factor, u0, t):
        """
        Simple FFT solver for the diffusion part

        Args:
            rhs (dtype_f): right-hand side for the linear system
            factor (float) : abbrev. for the node-to-node stepsize (or any other factor required)
            u0 (dtype_u): initial guess for the iterative solver (not used here so far)
            t (float): current time (e.g. for time-dependent BCs)

        Returns:
            dtype_u: solution as mesh
        """

        def linear_solve(k, v):
            k2 = sum(ki ** 2 for ki in k)
            return 1.0 / (1.0 + factor * k2) * v

        me = self.dtype_u(self.init)
        tmp_rhs = self.pm.create(type='real', value=rhs.values)
        me.values = tmp_rhs.r2c().apply(linear_solve, out=Ellipsis).c2r(out=Ellipsis).value
        return me

    def u_exact(self, t):
        """
        Routine to compute the exact solution at time t

        Args:
            t (float): current time

        Returns:
            dtype_u: exact solution
        """

        def circle(i, v):
            r = [ii * (Li / ni) - 0.5 * Li for ii, ni, Li in zip(i, v.Nmesh, v.BoxSize)]
            r2 = sum(ri ** 2 for ri in r)
            return np.tanh((self.params.radius - np.sqrt(r2)) / (np.sqrt(2) * self.params.eps))

        def checkerboard(i, v):
            r = [ii * (Li / ni) for ii, ni, Li in zip(i, v.Nmesh, v.BoxSize)]
            return np.sin(2 * np.pi * r[0]) * np.sin(2 * np.pi * r[1])

        assert t == 0, 'ERROR: u_exact only valid for t=0'
        me = self.dtype_u(self.init)
        if self.params.init_type == 'circle':
            tmp_u = self.pm.create(type='real', value=0.0)
            me.values = tmp_u.apply(circle, kind='index').value
        elif self.params.init_type == 'checkerboard':
            tmp_u = self.pm.create(type='real', value=0.0)
            me.values = tmp_u.apply(checkerboard, kind='index').value
        else:
            raise NotImplementedError('type of initial value not implemented, got %s' % self.params.init_type)

        return me


class allencahn_imex_stab(allencahn_imex):
    """
    Example implementing Allen-Cahn equation in 2-3D using PMESH for solving linear parts, IMEX time-stepping with
    stabilized splitting
    """

    def eval_f(self, u, t):
        """
        Routine to evaluate the RHS

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            dtype_f: the RHS
        """

        def Laplacian(k, v):
            k2 = sum(ki ** 2 for ki in k) + 2.0 / self.params.eps ** 2
            return -k2 * v

        f = self.dtype_f(self.init)
        tmp_u = self.pm.create(type='real', value=u.values)
        f.impl.values = tmp_u.r2c().apply(Laplacian, out=Ellipsis).c2r(out=Ellipsis).value
        if self.params.eps > 0:
            f.expl.values = 1.0 / self.params.eps ** 2 * u.values * (1.0 - u.values ** self.params.nu) + \
                2.0 / self.params.eps ** 2 * u.values
        return f

    def solve_system(self, rhs, factor, u0, t):
        """
        Simple FFT solver for the diffusion part

        Args:
            rhs (dtype_f): right-hand side for the linear system
            factor (float) : abbrev. for the node-to-node stepsize (or any other factor required)
            u0 (dtype_u): initial guess for the iterative solver (not used here so far)
            t (float): current time (e.g. for time-dependent BCs)

        Returns:
            dtype_u: solution as mesh
        """

        def linear_solve(k, v):
            k2 = sum(ki ** 2 for ki in k) + 2.0 / self.params.eps ** 2
            return 1.0 / (1.0 + factor * k2) * v

        me = self.dtype_u(self.init)
        tmp_rhs = self.pm.create(type='real', value=rhs.values)
        me.values = tmp_rhs.r2c().apply(linear_solve, out=Ellipsis).c2r(out=Ellipsis).value

        return me
