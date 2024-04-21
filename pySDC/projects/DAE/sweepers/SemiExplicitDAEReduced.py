import numpy as np

from pySDC.core.Errors import ParameterError
from pySDC.projects.DAE.sweepers.SemiExplicitDAE import SemiExplicitDAE


class SemiExplicitDAEReduced(SemiExplicitDAE):
    r"""
    TODO: Write docu
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

    def update_nodes(self):
        r"""
        Updates the values of solution ``u`` and their gradient stored in ``f``.
        """

        L = self.level
        P = L.prob

        # only if the level has been touched before
        assert L.status.unlocked
        M = self.coll.num_nodes

        u_0 = L.u[0]
        integral = self.integrate()
        # build the rest of the known solution u_0 + del_t(Q - Q_del)U_k
        for m in range(1, M + 1):
            for j in range(1, M + 1):
                integral[m - 1].diff -= L.dt * self.QI[m, j] * L.f[j].diff
            integral[m - 1].diff += u_0.diff

        # do the sweep
        for m in range(1, M + 1):
            u_approx = P.dtype_u(integral[m - 1])
            for j in range(1, m):
                u_approx.diff += L.dt * self.QI[m, j] * L.f[j].diff

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

                local_u_approx = P.dtype_u(u_approx)

                local_u_approx.diff += L.dt * self.QI[m, m] * unknowns.diff
                local_u_approx.alg = unknowns.alg
                sys = P.eval_f(local_u_approx, unknowns, L.time + L.dt * self.coll.nodes[m - 1])

                return sys

            u0 = P.dtype_u(P.init)
            u0.diff[:] = L.f[m].diff
            u_new = P.solve_system(implSystem, u0, L.time + L.dt * self.coll.nodes[m - 1])
            # ---- update U' and z ----
            L.f[m].diff[:] = u_new.diff

            G = P.eval_G(u_new, L.time + L.dt * self.coll.nodes[m - 1])
            L.u[m].alg[:] = G.alg[:]

        # Update solution approximation
        integral = self.integrate()
        for m in range(M):
            L.u[m + 1].diff = u_0.diff + integral[m].diff

        # indicate presence of new values at this level
        L.status.updated = True

        return None