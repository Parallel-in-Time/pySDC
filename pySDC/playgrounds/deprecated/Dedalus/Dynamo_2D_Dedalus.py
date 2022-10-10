import numpy as np

from dedalus import public as de

from pySDC.core.Problem import ptype
from pySDC.core.Errors import ParameterError

from pySDC.playgrounds.Dedalus.dedalus_field import dedalus_field, rhs_imex_dedalus_field


class dynamo_2d_dedalus(ptype):
    """ """

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
        super(dynamo_2d_dedalus, self).__init__(
            init=(domain, 2), dtype_u=dtype_u, dtype_f=dtype_f, params=problem_params
        )

        self.x = self.init[0].grid(0, scales=1)
        self.y = self.init[0].grid(1, scales=1)

        imp_var = ['b_x', 'b_y']
        self.problem_imp = de.IVP(domain=self.init[0], variables=imp_var)
        self.problem_imp.parameters['Rm'] = self.params.Rm
        self.problem_imp.parameters['kz'] = self.params.kz
        self.problem_imp.add_equation("dt(b_x) - (1/Rm) * ( dx(dx(b_x)) + dy(dy(b_x)) - kz ** 2 * (b_x) ) = 0")
        self.problem_imp.add_equation("dt(b_y) - (1/Rm) * ( dx(dx(b_y)) + dy(dy(b_y)) - kz ** 2 * (b_y) ) = 0")
        self.solver_imp = self.problem_imp.build_solver(de.timesteppers.SBDF1)
        self.imp_var = []
        for l in range(self.init[1]):
            self.imp_var.append(self.solver_imp.state[imp_var[l]])

        exp_var = ['b_x', 'b_y']
        self.problem_exp = de.IVP(domain=self.init[0], variables=exp_var)
        self.problem_exp.parameters['Rm'] = self.params.Rm
        self.problem_exp.parameters['kz'] = self.params.kz
        self.problem_exp.substitutions['u_x'] = 'cos(y)'
        self.problem_exp.substitutions['u_y'] = 'sin(x)'
        self.problem_exp.substitutions['u_z'] = 'cos(x) + sin(y)'
        self.problem_exp.add_equation("dt(b_x) = -u_x * dx(b_x) - u_y * dy(b_x) - u_z*1j*kz *b_x  + b_y * dy(u_x)")
        self.problem_exp.add_equation("dt(b_y) = -u_x * dx(b_y) - u_y * dy(b_y) - u_z*1j*kz *b_y  + b_x * dx(u_y)")
        self.solver_exp = self.problem_exp.build_solver(de.timesteppers.SBDF1)
        self.exp_var = []
        for l in range(self.init[1]):
            self.exp_var.append(self.solver_exp.state[exp_var[l]])

    def eval_f(self, u, t):
        """
        Routine to evaluate both parts of the RHS

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            dtype_f: the RHS with two parts
        """

        pseudo_dt = 1e-05

        f = self.dtype_f(self.init)

        for l in range(self.init[1]):
            self.exp_var[l]['g'] = u.values[l]['g']
            self.exp_var[l]['c'] = u.values[l]['c']

        self.solver_exp.step(pseudo_dt)

        for l in range(self.init[1]):
            self.exp_var[l].set_scales(1)
            f.expl.values[l]['g'] = 1.0 / pseudo_dt * (self.exp_var[l]['g'] - u.values[l]['g'])
            f.expl.values[l]['c'] = 1.0 / pseudo_dt * (self.exp_var[l]['c'] - u.values[l]['c'])

        for l in range(self.init[1]):
            self.imp_var[l]['g'] = u.values[l]['g']
            self.imp_var[l]['c'] = u.values[l]['c']

        self.solver_imp.step(pseudo_dt)

        for l in range(self.init[1]):
            # self.imp_var[l].set_scales(1)
            f.impl.values[l]['g'] = 1.0 / pseudo_dt * (self.imp_var[l]['g'] - u.values[l]['g'])
            f.impl.values[l]['c'] = 1.0 / pseudo_dt * (self.imp_var[l]['c'] - u.values[l]['c'])

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

        me = self.dtype_u(self.init)

        for l in range(self.init[1]):
            self.imp_var[l]['g'] = rhs.values[l]['g']
            self.imp_var[l]['c'] = rhs.values[l]['c']

        self.solver_imp.step(factor)

        for l in range(self.init[1]):
            # self.imp_var[l].set_scales(1)
            me.values[l]['g'][:] = self.imp_var[l]['g']
            me.values[l]['c'][:] = self.imp_var[l]['c']

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
