from mpi4py import MPI

from pySDC.projects.DAE.sweepers.fullyImplicitDAEMPI import SweeperDAEMPI
from pySDC.projects.DAE.sweepers.semiImplicitDAE import SemiImplicitDAE


class SemiImplicitDAEMPI(SweeperDAEMPI, SemiImplicitDAE):
    r"""
    Custom sweeper class to implement the fully-implicit SDC parallelized across the nodes for solving fully-implicit DAE problems of the form

    .. math::
        u' = f(u, z, t),

    .. math::
        0 = g(u, z, t)

    More detailed description can be found in ``semiImplicitDAE.py``. To parallelize SDC across the method the idea is to use a diagonal :math:`\mathbf{Q}_\Delta` to have solves of the implicit system on each node that can be done in parallel since they are fully decoupled from values of previous nodes. First such diagonal :math:`\mathbf{Q}_\Delta` were developed in [1]_. Years later intensive theory about the topic was developed in [2]_. For the DAE case these ideas were basically transferred.

    Note
    ----
    For more details of implementing a sweeper enabling parallelization across the method we refer to documentation in [generic_implicit_MPI.py](https://github.com/Parallel-in-Time/pySDC/blob/master/pySDC/implementations/sweeper_classes/generic_implicit_MPI.py) and [fullyImplicitDAEMPI.py](https://github.com/Parallel-in-Time/pySDC/blob/master/pySDC/projects/DAE/sweepers/fullyImplicitDAEMPI.py).

    For parallelization across the method for semi-explicit DAEs, differential and algebraic parts have to be treated separately. In ``integrate()`` these parts needs to be collected separately, otherwise information would get lost.

    Reference
    ---------
    .. [1] R. Speck. Parallelizing spectral deferred corrections across the method. Comput. Vis. Sci. 19, No. 3-4, 75-83 (2018).
    .. [2] G. Čaklović,  T. Lunet, S. Götschel, D. Ruprecht. Improving Efficiency of Parallel Across the Method Spectral Deferred Corrections. Preprint, arXiv:2403.18641 [math.NA] (2024).
    """

    def integrate(self, last_only=False):
        """
        Integrates the gradient. Note that here only the differential part is integrated, i.e., the
        integral over the algebraic part is zero. ``me`` serves as buffer, and root process ``m``
        stores the result of integral at node :math:`\tau_m` there.

        Parameters
        ----------
        last_only : bool, optional
            Integrate only the last node for the residual or all of them.

        Returns
        -------
        me : list of dtype_u
            Containing the integral as values.
        """

        L = self.level
        P = L.prob

        me = P.dtype_u(P.init, val=0.0)
        for m in [self.coll.num_nodes - 1] if last_only else range(self.coll.num_nodes):
            integral = P.dtype_u(P.init, val=0.0)
            integral.diff[:] = L.dt * self.coll.Qmat[m + 1, self.rank + 1] * L.f[self.rank + 1].diff[:]
            recvBuf = me[:] if m == self.rank else None
            self.comm.Reduce(integral, recvBuf, root=m, op=MPI.SUM)

        return me

    def update_nodes(self):
        r"""
        Updates values of ``u`` and ``f`` at collocation nodes. This correspond to a single iteration of the
        preconditioned Richardson iteration in **"ordinary"** SDC.
        """

        L = self.level
        P = L.prob

        # only if the level has been touched before
        assert L.status.unlocked

        integral = self.integrate()
        integral.diff[:] -= L.dt * self.QI[self.rank + 1, self.rank + 1] * L.f[self.rank + 1].diff[:]
        integral.diff[:] += L.u[0].diff[:]

        u_approx = P.dtype_u(integral)

        u0 = P.dtype_u(P.init)
        u0.diff[:], u0.alg[:] = L.f[self.rank + 1].diff[:], L.u[self.rank + 1].alg[:]

        u_new = P.solve_system(
            SemiImplicitDAE.F,
            u_approx,
            L.dt * self.QI[self.rank + 1, self.rank + 1],
            u0,
            L.time + L.dt * self.coll.nodes[self.rank],
        )

        L.f[self.rank + 1].diff[:] = u_new.diff[:]
        L.u[self.rank + 1].alg[:] = u_new.alg[:]

        integral = self.integrate()
        L.u[self.rank + 1].diff[:] = L.u[0].diff[:] + integral.diff[:]

        # indicate presence of new values at this level
        L.status.updated = True

        return None
