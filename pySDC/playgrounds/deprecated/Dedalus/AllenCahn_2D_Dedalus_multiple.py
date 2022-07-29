import numpy as np

from dedalus import public as de

from pySDC.core.Problem import ptype
from pySDC.core.Errors import ParameterError

from pySDC.playgrounds.Dedalus.dedalus_field import dedalus_field, rhs_imex_dedalus_field


class allencahn2d_dedalus(ptype):
    """
    Example implementing the 2D Allen-Cahn equation with periodic BC in [-L/2,L/2]^2, discretized using Dedalus
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
        if 'L' not in problem_params:
            problem_params['L'] = 1.0
        if 'init_type' not in problem_params:
            problem_params['init_type'] = 'circle'

        # these parameters will be used later, so assert their existence
        essential_keys = ['nvars', 'nu', 'eps', 'L', 'radius', 'comm', 'init_type']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        L = problem_params['L']
        xbasis = de.Fourier('x', problem_params['nvars'][0], interval=(-L / 2, L / 2), dealias=1)
        ybasis = de.Fourier('y', problem_params['nvars'][1], interval=(-L / 2, L / 2), dealias=1)
        domain = de.Domain([xbasis, ybasis], grid_dtype=np.float64, comm=problem_params['comm'])

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(allencahn2d_dedalus, self).__init__(init=domain, dtype_u=dtype_u, dtype_f=dtype_f, params=problem_params)

        self.x = self.init.grid(0, scales=1)
        self.y = self.init.grid(1, scales=1)
        self.rhs = self.dtype_u(self.init)
        linear_problem = de.IVP(domain=self.init, variables=['u'])
        linear_problem.add_equation("dt(u) - dx(dx(u)) - dy(dy(u)) = 0")
        self.solver = linear_problem.build_solver(de.timesteppers.SBDF1)
        self.u = self.solver.state['u']
        self.f = domain.new_field()
        self.fxx = xbasis.Differentiate(xbasis.Differentiate(self.f)) + ybasis.Differentiate(
            ybasis.Differentiate(self.f)
        )

        self.dx = L / len(self.x)
        self.nvars = (len(self.x), len(self.y))

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
        self.f['g'] = u.values['g']
        self.f['c'] = u.values['c']
        f.impl.values = self.fxx.evaluate()

        if self.params.eps > 0:
            f.expl.values['g'] = 1.0 / self.params.eps**2 * u.values['g'] * (1.0 - u.values['g'] ** self.params.nu)
        else:
            raise NotImplementedError()
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
            dtype_u: solution as Dedalus field
        """

        self.u['g'] = rhs.values['g']
        self.u['c'] = rhs.values['c']

        self.solver.step(factor)

        me = self.dtype_u(self.init)
        me.values['g'] = self.u['g']

        # uxx = (de.operators.differentiate(self.u, x=2) + de.operators.differentiate(self.u, y=2)).evaluate()
        # print(np.amax(abs(self.u['g'] - factor * uxx['g'] - rhs.values['g'])))

        return me

    def u_exact(self, t):
        """
        Routine to compute the exact solution at time t

        Args:
            t (float): current time

        Returns:
            dtype_u: exact solution
        """

        assert t == 0, 'ERROR: u_exact only valid for t=0'
        me = self.dtype_u(self.init, val=0.0)
        if self.params.init_type == 'circle':
            # xv, yv = np.meshgrid(self.x, self.y, indexing='ij')
            me.values['g'] = np.tanh(
                (self.params.radius - np.sqrt(self.x**2 + self.y**2)) / (np.sqrt(2) * self.params.eps)
            )
        elif self.params.init_type == 'checkerboard':
            me.values['g'] = np.sin(2.0 * np.pi * self.x) * np.sin(2.0 * np.pi * self.y)
        elif self.params.init_type == 'random':
            me.values['g'] = np.random.uniform(-1, 1, self.init)
        else:
            raise NotImplementedError('type of initial value not implemented, got %s' % self.params.init_type)

        return me
