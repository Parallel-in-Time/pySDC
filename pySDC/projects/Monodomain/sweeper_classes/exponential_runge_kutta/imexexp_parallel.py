import numpy as np

from pySDC.projects.Monodomain.sweeper_classes.exponential_runge_kutta.imexexp_1st_order import imexexp_1st_order


class imexexp_parallel(imexexp_1st_order):
    r"""
    Exponential SDC whose node updates within a sweep are independent, so the M nodes can be computed in parallel.

    The collocation problem is the one of ``imexexp_1st_order`` (exponential collocation for the gating variables,
    standard collocation for the rest). Each node is updated from the previous iterate only:

    .. math::
        u_m^{k+1} = u_0 + I_m(u^k) + \Delta t\, q_m \big(f_I(u_m^{k+1}) - f_I(u_m^k)\big),

    with :math:`I_m` the (exponential) collocation integral from the step start to node m and :math:`q_m` the diagonal
    MIN-SR-S coefficients for the implicit diffusion. The explicit ionic terms and the gates' deviation from their
    rate frozen at the step start enter through :math:`I_m(u^k)` only (Picard). The critical path of a sweep is one
    ionic evaluation and one diffusion solve, against M of each for the serial sweep.
    """

    def __init__(self, params, level):
        super().__init__(params, level)
        self.q = np.diag(self.get_Qdelta_implicit(qd_type='MIN-SR-S'))[1:]

    def update_nodes(self):
        L = self.level
        P = L.prob
        assert L.status.unlocked
        M = self.coll.num_nodes

        integral = self.integrate()  # integrals from the step start, at the previous iterate
        for m in range(M):
            if L.tau[m] is not None:
                integral[m] += L.tau[m]

        new = []
        for m in range(M):  # independent of each other: this loop is the parallel part
            a = L.dt * self.q[m]
            rhs = L.u[0] + integral[m]
            rhs[P.rhs_stiff_indeces] -= a * L.f[m + 1].impl[P.rhs_stiff_indeces]
            new.append(P.solve_system(rhs, a, L.u[m + 1], L.time + L.dt * self.coll.nodes[m]))
        for m in range(M):
            L.u[m + 1] = new[m]
            P.eval_f(L.u[m + 1], L.time + L.dt * self.coll.nodes[m], fh=L.f[m + 1])

        L.status.updated = True
        return None
