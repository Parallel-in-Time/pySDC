import numpy as np

from dedalus import public as de

from pySDC.core.Problem import ptype
from pySDC.core.Errors import ParameterError

from pySDC.playgrounds.Dedalus.dedalus_field import dedalus_field, rhs_imex_dedalus_field


class rayleighbenard_2d_dedalus(ptype):
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
        essential_keys = ['nvars', 'Ra', 'Pr', 'comm', 'initial']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        xbasis = de.Fourier('x', problem_params['nvars'][0], interval=(0, 2), dealias=3 / 2)
        zbasis = de.Chebyshev('z', problem_params['nvars'][1], interval=(-1 / 2, +1 / 2), dealias=3 / 2)
        domain = de.Domain([xbasis, zbasis], grid_dtype=np.float64, comm=problem_params['comm'])

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(rayleighbenard_2d_dedalus, self).__init__(init=(domain, 6), dtype_u=dtype_u, dtype_f=dtype_f,
                                                        params=problem_params)

        self.x = self.init[0].grid(0, scales=1)
        self.z = self.init[0].grid(1, scales=1)

        self.rhs = self.dtype_u(self.init, val=0.0)
        imp_var = ['T', 'u', 'w', 'Tz', 'uz', 'wz', 'p']
        self.problem_imp = de.IVP(domain=self.init[0], variables=imp_var)

        self.problem_imp.parameters['Ra'] = self.params.Ra
        self.problem_imp.parameters['Pr'] = self.params.Pr
        self.problem_imp.add_equation("dx(u) + wz = 0 ")
        self.problem_imp.add_equation("        dt(T) - ( dx(dx(T)) + dz(Tz) )        = 0")
        self.problem_imp.add_equation(" dt(u) - ( dx(dx(u)) + dz(uz) ) +dx(p) = 0")  # need to look at Pr
        self.problem_imp.add_equation(" dt(w) - ( dx(dx(w)) + dz(wz) ) +dz(p) = 0")  # Need to look at Pr

        self.problem_imp.add_equation("Tz - dz(T) = 0")
        self.problem_imp.add_equation("uz - dz(u) = 0")
        self.problem_imp.add_equation("wz - dz(w) = 0")

        # Boundary conditions.
        self.problem_imp.add_bc("left(T) = 1")
        self.problem_imp.add_bc("left(u) = 0")
        self.problem_imp.add_bc("left(w) = 0")
        self.problem_imp.add_bc("right(T) = 0")
        self.problem_imp.add_bc("right(u) = 0")
        self.problem_imp.add_bc("right(w) = 0", condition="(nx != 0)")
        self.problem_imp.add_bc("left(p) = 0", condition="(nx == 0)")
        self.solver_imp = self.problem_imp.build_solver(de.timesteppers.SBDF1)
        self.imp_var = []
        for var in imp_var:
            self.imp_var.append(self.solver_imp.state[var])

        exp_var = ['T', 'u', 'w', 'Tz', 'uz', 'wz']
        self.problem_exp = de.IVP(domain=self.init[0], variables=exp_var)

        self.problem_exp.parameters['Ra'] = self.params.Ra
        self.problem_exp.parameters['Pr'] = self.params.Pr

        self.problem_exp.add_equation("dt(T)    = -  (u * dx(T) + w * dz(T) )")
        self.problem_exp.add_equation("dt(u)    = -  (u * dx(u) + w * dz(u) )")  # Need to look at pr
        self.problem_exp.add_equation("dt(w)    = -  (u * dx(w) + w * dz(w) ) + Ra*T ")  # need to look at pr

        self.problem_exp.add_equation("Tz    = dz(T)")
        self.problem_exp.add_equation("uz    = dz(u)")
        self.problem_exp.add_equation("wz    = dz(w)")
        self.solver_exp = self.problem_exp.build_solver(de.timesteppers.SBDF1)
        self.exp_var = []
        for var in exp_var:
            self.exp_var.append(self.solver_exp.state[var])

    def eval_f(self, u, t):
        """
        Routine to evaluate both parts of the RHS

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            dtype_f: the RHS with two parts
        """

        pseudo_dt = 1E-05

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
        zvar_loc = self.z.shape[1]

        np.random.seed(0)

        me = self.dtype_u(self.init)

        if self.params.initial == 'random':

            for l in range(self.init[1]):
                me.values[l]['g'] = np.zeros((xvar_loc, zvar_loc))

            me.values[0]['g'] = -self.z + 0.5 + np.random.random(size=(xvar_loc, zvar_loc)) * 1e-2

        elif self.params.initial == 'low-res':


            for l in range(self.init[1]):
                me.values[l]['g'] = np.zeros((xvar_loc, zvar_loc))

            me.values[0]['g'] = np.random.uniform(low=-1e-5, high=1e-5, size=(xvar_loc, zvar_loc))

            me.values[0].set_scales(4.0 / self.params.nvars[0])
            # Need to do that because otherwise Dedalus tries to be clever..
            tmpx = me.values[0]['g']

            me.values[0].set_scales(1)

            me.values[0]['g'] += 0.5 - self.z

        return me
