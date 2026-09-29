import numpy as np

from pySDC.implementations.problem_classes.HeatEquation_Chebychev import Heat1DUltraspherical
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit


class Heat1DTimeDepBCs(Heat1DUltraspherical):
    r"""
    Heat equation :math:`u_t = \nu u_{xx}` on :math:`(-1, 1)` with time-dependent Dirichlet data
    :math:`u(-1, t) = a\cos(f_t t)` and :math:`u(1, t) = b\cos(f_t t)`, in three ways of imposing it.

    - ``'pointwise'``: every stage takes the data at its node, :math:`u_B(\tau_m) = g(\tau_m)`.
      This loses order: RADAU-RIGHT with M nodes falls from :math:`2M-1` to :math:`M+1`; both remedies
      below reach :math:`M+2`. See README.rst for the measurements.
    - ``'lifted'``: solve for :math:`v = u - E` with the lift :math:`E(x, t) = (g_-(t)(1-x) + g_+(t)(1+x))/2`,
      which is linear in x, so :math:`v_t = \nu v_{xx} - E_t` with homogeneous data. The state is
      ``v``; use :meth:`to_u` to get ``u`` back.
    - ``'differentiated'``: impose the data on its derivative and recover the stage values by the
      collocation quadrature, :math:`u_B(\tau_m) = g(t_0) + \Delta t \sum_j Q_{mj} \dot g(\tau_j)`.
      Needs the :class:`generic_implicit_diffbc` sweeper, which hands over the step.

    This is the spectral sibling of ``playgrounds/FEniCS/order_reduction`` and of the Taylor-Green
    study in ``projects/StroemungsRaum``: it has no spatial error floor above roundoff, so orders
    can be read off down to about 1e-12. Idea and first version by Thomas Baumann, pull request #634.

    Args:
        ft (float): Frequency of the boundary data in time; 0 gives constant data
        bc_mode (str): ``'pointwise'``, ``'lifted'`` or ``'differentiated'``
        kwargs: Forwarded to :class:`Heat1DUltraspherical`, with ``a`` and ``b`` the data at t=0; the
            initial mode ``f=4`` decays fast enough that constant data shows the design order
    """

    def __init__(self, ft=1.0, bc_mode='pointwise', nvars=128, a=1, b=2, f=4, nu=1e-2, **kwargs):
        assert bc_mode in ['pointwise', 'lifted', 'differentiated'], f'Unknown {bc_mode=}'
        super().__init__(nvars=nvars, a=a, b=b, f=f, nu=nu, **kwargs)
        self._makeAttributeAndRegister('ft', 'bc_mode', localVars=locals(), readOnly=True)
        self._node_times = None

    def g(self, t):
        """Boundary data (left, right) at time `t`"""
        return self.a * np.cos(self.ft * t), self.b * np.cos(self.ft * t)

    def g_dot(self, t):
        """Time derivative of the boundary data (left, right) at time `t`"""
        return -self.ft * self.a * np.sin(self.ft * t), -self.ft * self.b * np.sin(self.ft * t)

    def lift(self, t, derivative=False):
        """The lift E(x, t), or its time derivative, in the same space as the solution"""
        left, right = self.g_dot(t) if derivative else self.g(t)
        E = self.u_init
        E[0] = (left * (1 - self.x) + right * (1 + self.x)) / 2
        return self.transform(E) if self.spectral_space else E

    def to_u(self, u, t):
        """The solution `u` from the state, which is `v = u - E` in lifted mode"""
        return u + self.lift(t) if self.bc_mode == 'lifted' else u

    def prepare_step(self, t0, dt, coll):
        """Build the differentiated boundary values of all stages; called by `generic_implicit_diffbc`"""
        self._node_times = t0 + dt * coll.nodes
        rates = np.array([self.g_dot(t) for t in self._node_times])
        self._node_bcs = np.array(self.g(t0)) + dt * coll.Qmat[1:, 1:] @ rates

    def eval_f(self, u, t, *args, **kwargs):
        f = super().eval_f(u, t, *args, **kwargs)
        if self.bc_mode == 'lifted':
            f -= self.lift(t, derivative=True)
        return f

    def solve_system(self, rhs, dt, u0, t, *args, **kwargs):
        """Put this node's boundary values into the BC rows, then solve as `Heat1DUltraspherical` does"""
        if self.bc_mode == 'pointwise':
            left, right = self.g(t)
        elif self.bc_mode == 'lifted':
            left, right = 0, 0
            rhs = rhs - dt * self.lift(t, derivative=True)
        else:
            assert self._node_times is not None, 'differentiated BCs need the generic_implicit_diffbc sweeper'
            node = np.flatnonzero(self._node_times == t)
            assert node.size == 1, f'{t=} is no node of the prepared step {self._node_times}'
            left, right = self._node_bcs[node[0]]

        self.spectral.rhs_BCs_hat[0, -1] = left
        self.spectral.rhs_BCs_hat[0, -2] = right
        return super().solve_system(rhs, dt, u0, t, *args, **kwargs)

    def u_exact(self, t=0):
        """Initial conditions; there is no closed-form solution for t > 0"""
        assert t == 0, 'Heat1DTimeDepBCs only knows its initial conditions'
        u = super().u_exact(0)
        return u - self.lift(0) if self.bc_mode == 'lifted' else u


class generic_implicit_diffbc(generic_implicit):
    """`generic_implicit` that hands each step to the problem's `prepare_step` before predicting"""

    def predict(self):
        L = self.level
        L.prob.prepare_step(L.time, L.dt, self.coll)
        return super().predict()
