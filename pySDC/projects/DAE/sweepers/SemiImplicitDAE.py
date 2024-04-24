from pySDC.core.Errors import ParameterError
from pySDC.projects.DAE.sweepers.fully_implicit_DAE import fully_implicit_DAE


class SemiImplicitDAE(fully_implicit_DAE):
    r"""
    Custom sweeper class to implement SDC for solving semi-explicit DAEs of the form

    .. math::
        u' = f(u, z, t),

    .. math::
        0 = g(u, z, t)

    with :math:`u(t), u'(t) \in\mathbb{R}^{N_d}` the differential variables and their derivates,
    algebraic variables :math:`z(t) \in\mathbb{R}^{N_a}`, :math:`f(u, z, t) \in \mathbb{R}^{N_d}`,
    and :math:`g(u, z, t) \in \mathbb{R}^{N_a}`. :math:`N = N_d + N_a` is the dimension of the whole
    system of DAEs.

    It solves a collocation problem of the form

    .. math::
        U = f(\vec{U}_0 + \Delta t (\mathbf{Q} \otimes \mathbf{I}_{n_d}) \vec{U}, \vec{z}, \tau),

    .. math::
        0 = g(\vec{U}_0 + \Delta t (\mathbf{Q} \otimes \mathbf{I}_{n_d}) \vec{U}, \vec{z}, \tau),

    where
    
    - :math:`\tau=(\tau_1,..,\tau_M) in \mathbb{R}^M` the vector of collocation nodes,
    - :math:`\vec{U}_0 = (u_0,..,u_0) \in \mathbb{R}^{MN_d}` the vector of initial condition spread to each node,
    - spectral integration matrix :math:`\mathbf{Q} \in \mathbb{R}^{M \times M}`,
    - :math:`\vec{U}=(U_1,..,U_M) \in \mathbb{R}^{MN_d}` the vector of unknown derivatives of differential variables
      :math:`U_m \approx U(\tau_m) = u'(\tau_m) \in \mathbb{R}^{N_d}`,
    - :math:`\vec{z}=(z_1,..,z_M) \in \mathbb{R}^{MN_a}` the vector of unknown algebraic variables
      :math:`z_m \approx z(\tau_m) \in \mathbb{R}^{N_a}`,
    - and identity matrix :math:`\mathbf{I}_{N_d} \in \mathbb{R}^{N_d \times N_d}`.

    This sweeper treats the differential and the algebraic variables differently by only integrating the differential
    components. Solving the nonlinear system, :math:`{U,z}` are the unknowns.

    The sweeper implementation is based on the ideas mentioned in the KDC publication [1]_.

    Parameters
    ----------
    params : dict
        Parameters passed to the sweeper.

    Attributes
    ----------
    QI : np.2darray
        Implicit Euler integration matrix.

    References
    ----------
    .. [1] J. Huang, J. Jun, M. L. Minion. Arbitrary order Krylov deferred correction methods for differential algebraic
       equation. J. Comput. Phys. Vol. 221 No. 2 (2007).

    Note
    ----
    The right-hand side of the problem DAE classes using this sweeper has to be exactly implemented in the way, the
    semi-explicit DAE is defined. Define :math:`\vec{x}=(y, z)^T`, :math:`F(\vec{x})=(f(\vec{x}), g(\vec{x}))`, and the
    matrix

    .. math::
        A = \begin{matrix}
            I & 0 \\
            0 & 0
        \end{matrix}

    then, the problem can be reformulated as

    .. math::
        A\vec{x}' = F(\vec{x}).

    Then, setting :math:`F_{new}(\vec{x}, \vec{x}') = A\vec{x}' - F(\vec{x})` defines a DAE of fully-implicit form

    .. math::
        0 = F_{new}(\vec{x}, \vec{x}').

    Hence, the method ``eval_f`` of problem DAE classes of semi-explicit form implements the right-hand side in the way of
    returning :math:`F(\vec{x})`, whereas ``eval_f`` of problem classes of fully-implicit form return the right-hand side
    :math:`F_{new}(\vec{x}, \vec{x}')`.
    """

    def __init__(self, params):
        """Initialization routine"""

        if 'QI' not in params:
            params['QI'] = 'IE'

        # call parent's initialization routine
        super().__init__(params)

        msg = f"Quadrature type {self.params.quad_type} is not implemented yet. Use 'RADAU-RIGHT' instead!"
        if self.coll.left_is_node:
            raise ParameterError(msg)

        self.QI = self.get_Qdelta_implicit(coll=self.coll, qd_type=self.params.QI)

    def integrate(self):
        r"""
        Returns the solution by integrating its gradient (fundamental theorem of calculus) at each collocation node.
        ``level.f`` stores the gradient of solution ``level.u``.

        Returns
        -------
        me : list of lists
            Integral of the gradient at each collocation node.
        """

        # get current level and problem description
        L = self.level
        P = L.prob
        M = self.coll.num_nodes

        me = []
        for m in range(1, M + 1):
            # new instance of dtype_u, initialize values with 0
            me.append(P.dtype_u(P.init, val=0.0))
            for j in range(1, M + 1):
                me[-1].diff[:] += L.dt * self.coll.Qmat[m, j] * L.f[j].diff[:]

        return me

    def update_nodes(self):
        r"""
        Updates the values of solution ``u`` and their gradient stored in ``f``.
        """

        L = self.level
        P = L.prob

        # only if the level has been touched before
        assert L.status.unlocked
        M = self.coll.num_nodes

        integral = self.integrate()
        # build the rest of the known solution u_0 + del_t(Q - Q_del)U_k
        for m in range(1, M + 1):
            for j in range(1, m + 1):
                integral[m - 1].diff[:] -= L.dt * self.QI[m, j] * L.f[j].diff[:]
            integral[m - 1].diff[:] += L.u[0].diff

        # do the sweep
        for m in range(1, M + 1):
            u_approx = P.dtype_u(integral[m - 1])
            for j in range(1, m):
                u_approx.diff[:] += L.dt * self.QI[m, j] * L.f[j].diff[:]

            def implSystem(unknowns):
                """
                Build implicit system to solve in order to find the unknowns.

                Parameters
                ----------
                unknowns : dtype_u
                    Unknowns of the system.

                Returns
                -------
                sys :
                    System to be solved as implicit function.
                """

                unknowns_mesh = P.dtype_f(unknowns)

                local_u_approx = P.dtype_u(u_approx)
                local_u_approx.diff[:] += L.dt * self.QI[m, m] * unknowns_mesh.diff[:]
                local_u_approx.alg[:] = unknowns_mesh.alg[:]

                sys = P.eval_f(local_u_approx, unknowns_mesh, L.time + L.dt * self.coll.nodes[m - 1])
                return sys

            u0 = P.dtype_u(P.init)
            u0.diff[:], u0.alg[:] = L.f[m].diff[:], L.u[m].alg[:]
            u_new = P.solve_system(implSystem, u0, L.time + L.dt * self.coll.nodes[m - 1])
            # ---- update U' and z ----
            L.f[m].diff[:] = u_new.diff[:]
            L.u[m].alg[:] = u_new.alg[:]

        # Update solution approximation
        integral = self.integrate()
        for m in range(M):
            L.u[m + 1].diff[:] = L.u[0].diff[:] + integral[m].diff[:]

        # indicate presence of new values at this level
        L.status.updated = True

        return None
