import numpy as np
from mpi4py import MPI

from pySDC.implementations.sweeper_classes.generic_implicit_MPI import SweeperMPI
from pySDC.projects.Monodomain.sweeper_classes.exponential_runge_kutta.imexexp_parallel import imexexp_parallel


class imexexp_parallel_MPI(SweeperMPI, imexexp_parallel):
    r"""
    The node-parallel exponential SDC sweep of ``imexexp_parallel`` with one collocation node per MPI rank.

    Rank r holds node r only. The collocation integral to node m is the sum of every rank's weighted right-hand side,
    reduced onto rank m (as in pySDC's ``imex_1st_order_MPI``); the exponential part uses the per-cell weights of the
    exponential quadrature. The integral computed for the residual check is kept and reused by the next sweep, so each
    sweep reduces once. The residual is the serial ESDC's relative one, maximized over the ranks.
    """

    def integrate(self):
        """Integral from the step start to this rank's node, at the current iterate"""
        L = self.level
        P = L.prob
        M, r = self.coll.num_nodes, self.rank
        self.compute_lambda_phi_Qmat_exp()

        f = L.f[r + 1]
        q_exp = f.exp[P.rhs_exp_indeces] + self.lmbda * (L.u[0][P.rhs_exp_indeces] - L.u[r + 1][P.rhs_exp_indeces])
        me = P.dtype_u(init=P.init, val=0.0)
        for m in range(M):
            contrib = P.dtype_u(init=P.init, val=0.0)
            contrib[P.rhs_stiff_indeces] = self.coll.Qmat[m + 1, r + 1] * f.impl[P.rhs_stiff_indeces]
            contrib[P.rhs_nonstiff_indeces] += self.coll.Qmat[m + 1, r + 1] * f.expl[P.rhs_nonstiff_indeces]
            contrib[P.rhs_exp_indeces] += self.Qmat_exp[m, r] * q_exp
            contrib *= L.dt
            self.comm.Reduce(contrib, me if m == r else None, op=MPI.SUM, root=m)
        return me

    def update_nodes(self):
        L = self.level
        P = L.prob
        assert L.status.unlocked
        r = self.rank

        integral = self._integral if getattr(self, '_integral', None) is not None else self.integrate()
        self._integral = None
        a = L.dt * self.q[r]
        rhs = L.u[0] + integral
        if L.tau[r] is not None:
            rhs += L.tau[r]
        rhs[P.rhs_stiff_indeces] -= a * L.f[r + 1].impl[P.rhs_stiff_indeces]
        L.u[r + 1] = P.solve_system(rhs, a, L.u[r + 1], L.time + L.dt * self.coll.nodes[r])
        P.eval_f(L.u[r + 1], L.time + L.dt * self.coll.nodes[r], fh=L.f[r + 1])
        L.status.updated = True
        return None

    def predict(self):
        self._integral = None
        super().predict()

    def compute_residual(self, stage=''):
        L = self.level
        if stage in self.params.skip_residual_computation:
            L.status.residual = 0.0 if L.status.residual is None else L.status.residual
            return None
        r = self.rank
        self._integral = self.integrate()  # reused by the next sweep
        res = self._integral + L.u[0] - L.u[r + 1]
        if L.tau[r] is not None:
            res += L.tau[r]
        L.status.residual = self.comm.allreduce(self.rel_norm(res, L.u[0]), op=MPI.MAX)
        L.status.updated = False
        return None
