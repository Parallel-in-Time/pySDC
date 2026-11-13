r"""
Explicit SDC for SDEs with smooth Brownian bridges (SDC-SBB), after Binkowski, Christian, Götschel, Minion: "Toward
Spectral Deferred Correction Methods for Stochastic Differential Equations" (arXiv:2610.23693).

On each step, the SDE is replaced by the random ODE :math:`x' = \underline{a}(t, x) + \sum_r b_r(t, x)\, dB_r/dt` of
:mod:`pySDC.projects.SDC4SDEs.problem_classes`, and explicit SDC is applied to it: a sweep is

.. math::

    x^{k+1}_{m+1} = x_0 + I_{m+1}(x^k) + \Delta T \sum_{j \leq m} \tilde{q}_{m+1, j}
        \left(f(\tau_j, x^{k+1}_j) - f(\tau_j, x^k_j)\right),

with the right-hand side :math:`f` of the random ODE, the explicit Euler matrix :math:`\tilde{q}`, and the integral
:math:`I_m` of :math:`f` from the start of the step to node :math:`m`. The drift and the linear part of the bridge
are integrated with the collocation matrix, as in SDC for ODEs. The cosine terms of the bridge oscillate too fast for
that, so they are integrated exactly against the Lagrange polynomial through the diffusion at the nodes.
"""

import numpy as np

from pySDC.core.errors import ParameterError
from pySDC.core.sweeper import Sweeper


def lagrange_cosine_matrices(nodes, n_terms):
    r"""
    Integrals :math:`\int_0^{\tau_m} \ell_j(s) \cos(k \pi s)\, ds` of the Lagrange polynomials :math:`\ell_j` through
    the nodes :math:`\tau_j` in [0, 1], for k = 1, ..., n_terms.

    Args:
        nodes (numpy.ndarray): the M nodes in [0, 1]
        n_terms (int): number of cosine terms

    Returns:
        numpy.ndarray: shape (n_terms, M + 1, M + 1), with zeros in the first row and column like ``coll.Qmat``
    """
    M = len(nodes)
    # Gauss-Legendre with this many points integrates polynomials of degree M - 1 times cos(k pi s), k <= n_terms,
    # on intervals of length at most 1 to machine precision
    x, w = np.polynomial.legendre.leggauss(n_terms + M + 30)
    Q = np.zeros((n_terms, M + 1, M + 1))
    k = np.arange(1, n_terms + 1)
    for m in range(1, M + 1):
        s, ws = (x + 1) * nodes[m - 1] / 2, w * nodes[m - 1] / 2
        lagrange = np.array(
            [np.prod([(s - nodes[i]) / (nodes[j] - nodes[i]) for i in range(M) if i != j], axis=0) for j in range(M)]
        )
        Q[:, m, 1:] = np.cos(np.pi * np.outer(k, s)) @ (ws * lagrange).T
    return Q


class SBBSweeper(Sweeper):
    r"""
    Explicit SDC for the random ODE of an :class:`~pySDC.projects.SDC4SDEs.problem_classes.SDEProblem`.

    The sweeper needs a node at the right end of the step (e.g. ``quad_type='LOBATTO'``, as in the paper). It
    supports these initial guesses (``initial_guess``):

    - ``'spread'``: the initial value at all nodes
    - ``'euler'``: explicit Euler for the random ODE (default)
    - ``'euler_maruyama'``: one Euler-Maruyama step for the Itô form of the SDE over the whole step, interpolated
      linearly to the nodes

    Attributes:
        QE: explicit Euler integration matrix
    """

    def __init__(self, params, level):
        params = {'QE': 'EE', 'initial_guess': 'euler', **params}
        if params['initial_guess'] not in ['spread', 'euler', 'euler_maruyama']:
            raise ParameterError(f'initial_guess {params["initial_guess"]!r} not implemented')
        super().__init__(params, level)
        # shortcut: the end point is copied from the last node, add the integral over the step for other nodes
        if self.params.do_coll_update:
            raise ParameterError('SBBSweeper needs a node at the right end of the step and do_coll_update=False')
        self.QE = self.get_Qdelta_explicit(qd_type=self.params.QE)
        self.Qcos = lagrange_cosine_matrices(self.coll.nodes, 0)

    def node_times(self):
        """Times of the left end and of the nodes of the current step."""
        L = self.level
        return L.time + L.dt * np.concatenate([[0.0], self.coll.nodes])

    def predict(self):
        """Compute the bridge of the step, the integration weights for the diffusion, and the initial guess."""
        L, P = self.level, self.level.prob
        M, dt, t = self.coll.num_nodes, L.dt, self.node_times()
        P.set_step(L.time, dt)

        if self.Qcos.shape[0] != P.n_bridge_terms:
            self.Qcos = lagrange_cosine_matrices(self.coll.nodes, P.n_bridge_terms)
        # weights of the diffusion b_r in the integrals, per sample: shape (R, n_samples, M, M)
        Q, Qcos = self.coll.Qmat[1:, 1:], self.Qcos[:, 1:, 1:]
        self.noise_weights = np.sqrt(dt) * (
            np.multiply.outer(P.eta0, Q) + np.sqrt(2.0) * np.tensordot(P.xi, Qcos, axes=([2], [0]))
        )
        self.dB = [P.dB(tm)[..., None] for tm in t]

        L.f[0] = P.eval_f(L.u[0], t[0])
        if self.params.initial_guess == 'euler':
            for m in range(M):
                L.u[m + 1] = L.u[m] + (t[m + 1] - t[m]) * self.rhs(m)
                self.eval_f(m + 1, t[m + 1])
        else:
            if self.params.initial_guess == 'euler_maruyama':
                P.work_counters['diffusion']()
                du = dt * P.ito_drift(L.u[0], t[0]) + (P.dW[..., None] * P.diffusion(L.u[0], t[0])).sum(axis=0)
            else:
                du = 0.0 * L.u[0]
            for m in range(M):
                L.u[m + 1] = L.u[0] + self.coll.nodes[m] * du
                self.eval_f(m + 1, t[m + 1])

        L.status.unlocked = True
        L.status.updated = True

    def eval_f(self, m, t):
        """Evaluate the right-hand side at node m, or copy it from the left end if node m is there."""
        L = self.level
        if m == 1 and self.coll.left_is_node and L.tau[0] is None:
            L.f[1] = L.prob.dtype_f(L.f[0])
        else:
            L.f[m] = L.prob.eval_f(L.u[m], t)

    def rhs(self, m):
        """Right-hand side of the random ODE at node m (0 is the left end) from the current L.f[m]."""
        return self.level.f[m][0] + (self.dB[m] * self.level.f[m][1:]).sum(axis=0)

    def integrate(self):
        """Integrals of the right-hand side from the start of the step to the nodes."""
        L, P = self.level, self.level.prob
        F = np.stack([np.asarray(f) for f in L.f[1:]], axis=2)  # shape (1 + R, n_samples, M, ndim)
        integral = L.dt * self.coll.Qmat[1:, 1:] @ F[0] + (self.noise_weights @ F[1:]).sum(axis=0)
        me = []
        for m in range(self.coll.num_nodes):
            me.append(P.dtype_u(P.init))
            me[-1][:] = integral[:, m]
        return me

    def update_nodes(self):
        """One sweep over all nodes."""
        L = self.level
        assert L.status.unlocked
        M, t = self.coll.num_nodes, self.node_times()

        integral = self.integrate()
        rhs_old = [None] + [self.rhs(m) for m in range(1, M + 1)]
        for m in range(M):
            L.u[m + 1] = L.u[0] + integral[m]
            if L.tau[m] is not None:
                L.u[m + 1] += L.tau[m]
            for j in range(1, m + 1):
                L.u[m + 1] += L.dt * self.QE[m + 1, j] * (self.rhs(j) - rhs_old[j])
            self.eval_f(m + 1, t[m + 1])

        L.status.updated = True

    def compute_end_point(self):
        """The value at the last node, which is the right end of the step."""
        self.level.uend = self.level.prob.dtype_u(self.level.u[-1])
