import numpy as np

from pySDC.core.sweeper import Sweeper
from pySDC.core.errors import ParameterError


class rush_larsen(Sweeper):
    r"""
    The time steppers cardiac codes actually run, as one-node sweepers: each "sweep" is one full time step, so use them
    with ``num_nodes=1``, ``maxiter=1`` and every residual computation skipped.

    The gating variables :math:`z_e` relax exponentially, :math:`z_e' = \lambda(u)(z_e - z_{e,\infty}(u))`. With
    :math:`\lambda` and :math:`z_{e,\infty}` frozen at a state :math:`u^*` this is solved exactly (Rush-Larsen); every
    other ionic variable, the potential's ionic current and the stimulus are stepped explicitly, and the diffusion of
    the potential is implicit.

    ``scheme``:

    * ``'RL1'``: first-order IMEX Rush-Larsen, implicit Euler diffusion and Rush-Larsen reaction from the step start.
      One ionic evaluation and one diffusion solve per step.
    * ``'RL2'``: second-order Strang splitting, reaction over half steps with the second-order Rush-Larsen method
      (exponential midpoint, Marsh, Ziaratgahi and Spiteri, IEEE TBME 59, 2012) and Crank-Nicolson diffusion.
      ``splitting='RDR'`` costs four ionic evaluations and one solve per step, ``'DRD'`` two ionic evaluations and two
      solves.
    """

    def __init__(self, params, level):
        params.setdefault('scheme', 'RL1')
        params.setdefault('splitting', 'DRD')
        super().__init__(params, level)
        if self.coll.num_nodes != 1 or not self.coll.right_is_node:
            raise ParameterError('rush_larsen performs whole time steps and needs a single node at the right end')
        if self.params.scheme not in ['RL1', 'RL2'] or self.params.splitting not in ['RDR', 'DRD']:
            raise ParameterError(f'unknown scheme {self.params.scheme} or splitting {self.params.splitting}')

    def predict(self):
        # nothing to evaluate: the step computes what it needs itself
        L = self.level
        L.u[1] = L.prob.dtype_u(L.u[0])
        L.status.unlocked = True
        L.status.updated = True

    def reaction(self, u0, ustar, h, t):
        """
        One Rush-Larsen step of length h for the ionic model alone, from u0 with the right-hand side frozen at ustar
        """
        P = self.level.prob
        f = P.eval_f_nonstiff(ustar, t, P.dtype_u(init=P.init, val=0.0), zero_untouched_indeces=True)
        lmbda = P.dtype_u(init=P.init, val=0.0)
        yinf = P.dtype_u(init=P.init, val=0.0)
        P.eval_lmbda_yinf_exp(ustar, lmbda, yinf)
        P.work_counters['exp'].niter += len(P.rhs_exp_indeces)

        u = P.dtype_u(u0)
        u[P.rhs_nonstiff_indeces] += h * f[P.rhs_nonstiff_indeces]
        i = P.rhs_exp_indeces
        u[i] = yinf[i] + (u0[i] - yinf[i]) * np.exp(h * lmbda[i])
        return u

    def reaction_rl2(self, u0, h, t):
        """Second-order Rush-Larsen (exponential midpoint) for the ionic model alone"""
        return self.reaction(u0, self.reaction(u0, u0, h / 2, t), h, t + h / 2)

    def diffusion_cn(self, u, h, t):
        """Crank-Nicolson for the diffusion of the potential, other variables untouched"""
        P = self.level.prob
        rhs = P.dtype_u(u)
        lap = P.dtype_u(init=P.init, val=0.0)
        P.eval_f_stiff(u, t, lap, zero_untouched_indeces=True)
        rhs[0] += h / 2 * lap[0]
        return P.solve_system(rhs, h / 2, u, t + h)

    def update_nodes(self):
        L = self.level
        P = L.prob
        assert L.status.unlocked
        dt, t = L.dt, L.time

        if self.params.scheme == 'RL1':
            u = self.reaction(L.u[0], L.u[0], dt, t)
            L.u[1] = P.solve_system(u, dt, u, t + dt)
        elif self.params.splitting == 'RDR':
            u = self.reaction_rl2(L.u[0], dt / 2, t)
            u = self.diffusion_cn(u, dt, t)
            L.u[1] = self.reaction_rl2(u, dt / 2, t + dt / 2)
        else:
            u = self.diffusion_cn(L.u[0], dt / 2, t)
            u = self.reaction_rl2(u, dt, t)
            L.u[1] = self.diffusion_cn(u, dt / 2, t + dt / 2)

        L.status.updated = True

    def compute_end_point(self):
        self.level.uend = self.level.prob.dtype_u(self.level.u[1])

    def compute_residual(self, stage=''):
        # a time stepper has no collocation residual
        self.level.status.residual = 0.0
        self.level.status.updated = False
