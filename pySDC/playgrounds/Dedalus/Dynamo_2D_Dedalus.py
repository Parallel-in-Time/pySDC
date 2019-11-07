import numpy as np
from scipy import signal

from dedalus import public as de

from pySDC.core.Problem import ptype
from pySDC.core.Errors import ParameterError

from pySDC.playgrounds.Dedalus.dedalus_field import dedalus_field, rhs_imex_dedalus_field


class dynamo_2d_dedalus(ptype):
    """
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
        essential_keys = ['nvars', 'Rm', 'kz', 'comm', 'initial']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        xbasis = de.Fourier('x', problem_params['nvars'][0], interval=(0, 2 * np.pi), dealias=1)
        ybasis = de.Fourier('y', problem_params['nvars'][1], interval=(0, 2 * np.pi), dealias=1)
        domain = de.Domain([xbasis, ybasis], grid_dtype=np.complex128, comm=problem_params['comm'])

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(dynamo_2d_dedalus, self).__init__(init=(domain, 2), dtype_u=dtype_u, dtype_f=dtype_f,
                                                params=problem_params)

        self.x = self.init[0].grid(0, scales=1)
        self.y = self.init[0].grid(1, scales=1)

        self.rhs = self.dtype_u(self.init, val=0.0)
        self.problem = de.IVP(domain=self.init[0], variables=['b_x', 'b_y'])
        self.problem.parameters['Rm'] = self.params.Rm
        self.problem.parameters['kz'] = self.params.kz
        self.problem.add_equation("dt(b_x) - (1/Rm) * ( dx(dx(b_x)) + dy(dy(b_x)) - kz ** 2 * (b_x) ) = 0")
        self.problem.add_equation("dt(b_y) - (1/Rm) * ( dx(dx(b_y)) + dy(dy(b_y)) - kz ** 2 * (b_y) ) = 0")
        self.solver = self.problem.build_solver(de.timesteppers.SBDF1)
        self.b_x = self.solver.state['b_x']
        self.b_y = self.solver.state['b_y']

        self.u_x = domain.new_field()
        self.u_y = domain.new_field()
        self.u_z = domain.new_field()
        self.dx_uy = domain.new_field()
        self.dy_ux = domain.new_field()

        self.u_x['g'] = np.cos(self.y)
        self.u_y['g'] = np.sin(self.x)
        self.u_z['g'] = np.cos(self.x) + np.sin(self.y)
        self.dx_uy['g'] = self.u_y.differentiate(x=1)['g']
        self.dy_ux['g'] = self.u_x.differentiate(y=1)['g']

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

        dx_b_x = u.values[0].differentiate(x=1)
        dy_b_x = u.values[0].differentiate(y=1)
        dx_b_y = u.values[1].differentiate(x=1)
        dy_b_y = u.values[1].differentiate(y=1)

        f.expl.values[0] = (-self.u_x * dx_b_x - self.u_y * dy_b_x - self.u_z * 1j * self.params.kz * u.values[0] +
                            u.values[1] * self.dy_ux).evaluate()
        f.expl.values[1] = (-self.u_x * dx_b_y - self.u_y * dy_b_y - self.u_z * 1j * self.params.kz * u.values[1] +
                            u.values[0] * self.dx_uy).evaluate()

        f.impl.values[0] = (1.0 / self.params.Rm * (u.values[0].differentiate(x=2) + u.values[0].differentiate(y=2) -
                                                    self.params.kz ** 2 * u.values[0])).evaluate()
        f.impl.values[1] = (1.0 / self.params.Rm * (u.values[1].differentiate(x=2) + u.values[1].differentiate(y=2) -
                                                    self.params.kz ** 2 * u.values[1])).evaluate()

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

        self.b_x['g'] = rhs.values[0]['g']
        self.b_x['c'] = rhs.values[0]['c']
        self.b_y['g'] = rhs.values[1]['g']
        self.b_y['c'] = rhs.values[1]['c']

        self.solver.step(factor)

        me = self.dtype_u(self.init)
        me.values[0]['g'] = self.b_x['g']
        me.values[0]['c'] = self.b_x['c']
        me.values[1]['g'] = self.b_y['g']
        me.values[1]['c'] = self.b_y['c']

        return me

    def u_exact(self, t):
        """
        Routine to compute the exact solution at time t

        Args:
            t (float): current time

        Returns:
            dtype_u: exact solution
        """

        xvar_loc = self.x.shape[0]
        yvar_loc = self.y.shape[1]

        np.random.seed(0)

        me = self.dtype_u(self.init)

        if self.params.initial == 'random':

            me.values[0]['g'] = np.random.uniform(low=-1e-5, high=1e-5, size=(xvar_loc, yvar_loc))
            me.values[1]['g'] = np.random.uniform(low=-1e-5, high=1e-5, size=(xvar_loc, yvar_loc))

        elif self.params.initial == 'low-res':

            me.values[0]['g'] = np.random.uniform(low=-1e-5, high=1e-5, size=(xvar_loc, yvar_loc))
            me.values[1]['g'] = np.random.uniform(low=-1e-5, high=1e-5, size=(xvar_loc, yvar_loc))

            me.values[0].set_scales(4.0 / self.params.nvars[0])
            # Need to do that because otherwise Dedalus tries to be clever..
            tmpx = me.values[0]['g']
            me.values[1].set_scales(4.0 / self.params.nvars[1])
            # Need to do that because otherwise Dedalus tries to be clever..
            tmpy = me.values[1]['g']

            me.values[0].set_scales(1)
            me.values[1].set_scales(1)


        return me
