from mpi4py import MPI

from pySDC.implementations.sweeper_classes.generic_implicit_MPI import SweeperMPI
from pySDC.implementations.sweeper_classes.imex_1st_order_mass import imex_1st_order_mass


class imex_1st_order_mass_MPI(SweeperMPI, imex_1st_order_mass):
    """
    Node-parallel IMEX-SDC sweeper for problems M u' = f(u) with a mass matrix M, one collocation node per MPI rank.
    The node-parallel counterpart of
    :class:`~pySDC.implementations.sweeper_classes.imex_1st_order_mass.imex_1st_order_mass`: it needs a diagonal QI and
    QE = 'PIC', and the end point is a copy of the last node.
    """

    def integrate(self, last_only=False):
        """
        Integrate the right-hand side, dt (Q (f_I + f_E))_m, for the node this rank administers.

        The sum over the nodes is a reduction of load vectors, so the buffers are load vectors too, never a zero
        ``dtype_u``.

        Args:
            last_only (bool): Integrate only the last node for the residual or all of them

        Returns:
            dtype_f component: the integral at this rank's node, a load vector like f
        """
        L = self.level
        f = L.f[self.rank + 1].impl + L.f[self.rank + 1].expl

        me = None
        for m in [self.coll.num_nodes - 1] if last_only else range(self.coll.num_nodes):
            contribution = L.dt * self.coll.Qmat[m + 1, self.rank + 1] * f
            recvBuf = type(contribution)(contribution) if m == self.rank else None
            self.comm.Reduce(contribution, recvBuf, root=m, op=MPI.SUM)
            me = recvBuf if m == self.rank else me

        # with last_only, the other ranks get nothing; give them a zero of the right kind
        return 0.0 * f if me is None else me

    def _mass_u0(self):
        """M u0 on the finest level; below it, u0 is already the load vector."""
        L = self.level
        return L.prob.apply_mass_matrix(L.u[0]) if L.level_index == 0 else L.u[0]

    def update_nodes(self):
        """
        Update the value at this rank's collocation node, one sweep

        Returns:
            None
        """
        L = self.level
        P = L.prob
        m = self.rank

        # only if the level has been touched before
        assert L.status.unlocked

        # M u0 + dt Q F(u^k) - dt QI_mm FI(u^k)_m + tau; QE is zero on the diagonal
        rhs = self.integrate()
        rhs -= L.dt * self.QI[m + 1, m + 1] * L.f[m + 1].impl
        rhs += self._mass_u0()
        if L.tau[m] is not None:
            rhs += L.tau[m]

        t = L.time + L.dt * self.coll.nodes[m]
        L.u[m + 1] = P.solve_system(rhs, L.dt * self.QI[m + 1, m + 1], L.u[m + 1], t)
        L.f[m + 1] = P.eval_f(L.u[m + 1], t)

        # indicate presence of new values at this level
        L.status.updated = True

        return None

    def node_residual(self, last_only=False):
        """
        The residual at this rank's node, M u0 + dt (QF)_m + tau_m - M u_m, a load vector.

        Args:
            last_only (bool): Only the residual at the last node is needed, the other ranks may return anything

        Returns:
            dtype_f component: the residual
        """
        L = self.level
        P = L.prob
        res = self.integrate(last_only=last_only)
        res += self._mass_u0() - P.apply_mass_matrix(L.u[self.rank + 1])
        if L.tau[self.rank] is not None:
            res += L.tau[self.rank]
        # Due to different boundary conditions we might have to fix the residual
        if getattr(P, 'fix_bc_for_residual', False):
            P.fix_residual(res)
        return res

    def compute_end_point(self):
        """
        Copy the last node to the end point on every rank; below the finest level, hand it on as a load vector, as
        :meth:`~pySDC.implementations.sweeper_classes.imex_1st_order_mass.imex_1st_order_mass.compute_end_point` does.

        Returns:
            None
        """
        L = self.level
        P = L.prob
        if not self.coll.right_is_node or self.params.do_coll_update:
            raise NotImplementedError('Mass matrix sweeper expect u_M = u_end')

        # the buffer is solution data on every rank: below the finest level u[0] would be a load vector
        root = self.comm.size - 1
        L.uend = P.dtype_u(L.u[-1]) if self.rank == root else P.u_init
        self.comm.Bcast(L.uend, root=root)
        if L.level_index > 0:
            L.uend = P.apply_mass_matrix(L.uend)
        return None
