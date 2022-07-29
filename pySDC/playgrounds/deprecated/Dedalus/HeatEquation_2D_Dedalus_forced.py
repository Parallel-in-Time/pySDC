import numpy as np

from dedalus import public as de

from pySDC.core.Problem import ptype
from pySDC.core.Errors import ParameterError, ProblemError

from pySDC.playgrounds.Dedalus.dedalus_field import dedalus_field, rhs_imex_dedalus_field


class heat2d_dedalus_forced(ptype):
    """
    Example implementing the forced 2D heat equation with periodic BC in [0,1], discretized using Dedalus
    """

    def __init__(self, problem_params, dtype_u=dedalus_field, dtype_f=rhs_imex_dedalus_field):
        """
        Initialization routine

        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: mesh data type (will be passed parent class)
            dtype_f: mesh data type (will be passed parent class)
        """

        if 'comm' not in problem_params:
            problem_params['comm'] = None

        # these parameters will be used later, so assert their existence
        essential_keys = ['nvars', 'nu', 'freq', 'comm']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        # we assert that nvars looks very particular here.. this will be necessary for coarsening in space later on
        if problem_params['freq'] % 2 != 0:
            raise ProblemError('setup requires freq to be an equal number')

        xbasis = de.Fourier('x', problem_params['nvars'][0], interval=(0, 1), dealias=1)
        ybasis = de.Fourier('y', problem_params['nvars'][1], interval=(0, 1), dealias=1)
        domain = de.Domain([xbasis, ybasis], grid_dtype=np.float64, comm=problem_params['comm'])

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(heat2d_dedalus_forced, self).__init__(
            init=domain, dtype_u=dtype_u, dtype_f=dtype_f, params=problem_params
        )

        self.x = self.init.grid(0, scales=1)
        self.y = self.init.grid(1, scales=1)
        self.rhs = self.dtype_u(self.init, val=0.0)
        self.problem = de.IVP(domain=self.init, variables=['u'])
        self.problem.parameters['nu'] = self.params.nu
        self.problem.add_equation("dt(u) - nu * dx(dx(u)) - nu * dy(dy(u)) = 0")
        self.solver = self.problem.build_solver(de.timesteppers.SBDF1)
        self.u = self.solver.state['u']

    def eval_f(self, u, t):
        """
        Routine to evaluate both parts of the RHS

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            dtype_f: the RHS with two parts
        """

        f = self.dtype_f(self.init)
        f.impl.values = (
            self.params.nu * de.operators.differentiate(u.values, x=2)
            + self.params.nu * de.operators.differentiate(u.values, y=2)
        ).evaluate()
        f.expl.values['g'] = (
            -np.sin(np.pi * self.params.freq * self.x)
            * np.sin(np.pi * self.params.freq * self.y)
            * (np.sin(t) - 2.0 * self.params.nu * (np.pi * self.params.freq) ** 2 * np.cos(t))
        )
        return f

    def solve_system(self, rhs, factor, u0, t):
        """
        Simple linear solver for (I-factor*A)u = rhs

        Args:
            rhs (dtype_f): right-hand side for the linear system
            factor (float): abbrev. for the local stepsize (or any other factor required)
            u0 (dtype_u): initial guess for the iterative solver
            t (float): current time (e.g. for time-dependent BCs)

        Returns:
            dtype_u: solution as mesh
        """

        # u = self.solver.state['u']
        self.u['g'] = rhs.values['g']
        self.u['c'] = rhs.values['c']

        self.solver.step(factor)

        me = self.dtype_u(self.init)
        me.values['g'] = self.u['g']

        return me

    def u_exact(self, t):
        """
        Routine to compute the exact solution at time t

        Args:
            t (float): current time

        Returns:
            dtype_u: exact solution
        """

        me = self.dtype_u(self.init)
        me.values['g'] = (
            np.sin(np.pi * self.params.freq * self.x) * np.sin(np.pi * self.params.freq * self.y) * np.cos(t)
        )
        return me
