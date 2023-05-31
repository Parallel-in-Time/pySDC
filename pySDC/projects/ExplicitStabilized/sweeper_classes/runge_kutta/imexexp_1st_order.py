import numpy as np

from pySDC.core.Sweeper import sweeper
from pySDC.core.Errors import CollocationError

class imexexp_1st_order(sweeper):
    """
    Custom sweeper class, implements Sweeper.py

    First-order IMEXEXP sweeper using implicit/explicit/exponential Euler as base integrator
    In the cardiac electrphysiology community this is known as Rush-Larsen scheme.

    Attributes:
        QI: implicit Euler integration matrix
        QE: explicit Euler integration matrix
    """

    def __init__(self, params):
        """
        Initialization routine for the custom sweeper

        Args:
            params: parameters for the sweeper
        """

        if 'QI' not in params:
            params['QI'] = 'IE'
        if 'QE' not in params:
            params['QE'] = 'EE'

        # call parent's initialization routine
        super(imexexp_1st_order, self).__init__(params)

        # IMEX integration matrices
        self.QI = self.get_Qdelta_implicit(coll=self.coll, qd_type=self.params.QI)
        self.QE = self.get_Qdelta_explicit(coll=self.coll, qd_type=self.params.QE)

    def integrate(self):
        """
        Integrates the right-hand side (here impl + expl + exp)

        Returns:
            list of dtype_u: containing the integral as values
        """

        # get current level and problem description
        L = self.level
        P = L.prob

        me = []

        # integrate RHS over all collocation nodes
        for m in range(1, self.coll.num_nodes + 1):
            me.append(L.dt * self.coll.Qmat[m, 1] * (L.f[1].impl + L.f[1].expl + L.f[1].exp))
            # me.append(L.dt * self.coll.Qmat[m, 1] * (L.f[1].impl + L.f[1].expl + P.phi_one_f_eval(L.u[1],L.dt*self.coll.nodes[0],L.time)))
            # new instance of dtype_u, initialize values with 0
            for j in range(2, self.coll.num_nodes + 1):
                me[m - 1] += L.dt * self.coll.Qmat[m, j] * (L.f[j].impl + L.f[j].expl + L.f[j].exp)        
                # me[m - 1] += L.dt * self.coll.Qmat[m, j] * (L.f[j].impl + L.f[j].expl + P.phi_one_f_eval(L.u[j],L.dt*self.coll.nodes[j-1],L.time))        

        return me

    def update_nodes(self):
        """
        Update the u- and f-values at the collocation nodes -> corresponds to a single sweep over all nodes

        Returns:
            None
        """

        # get current level and problem description
        L = self.level
        P = L.prob

        # only if the level has been touched before
        assert L.status.unlocked

        # get number of collocation nodes for easier access
        M = self.coll.num_nodes

        integral = [P.dtype_u(P.init,val=0.) for _ in range(M)]
        for m in range(1,M + 1):
            for j in range(1, M + 1):
                integral[m - 1] += L.dt * self.coll.Smat[m, j] * (L.f[j].impl + L.f[j].expl + L.f[j].exp)      
                # integral[m - 1] += L.dt * self.coll.Smat[m, j] * (L.f[j].impl + L.f[j].expl + P.phi_one_f_eval(L.u[j],L.dt*self.coll.nodes[j-1],L.time))        

        if L.tau[0] is not None:
            integral[0] += L.tau[0]
        for m in range(1,M):
            if L.tau[m] is not None:
                integral[m] += L.tau[m]-L.tau[m-1]

        for m in range(M):
            # integral[m] -= L.dt*self.coll.delta_m[m]*(L.f[m].expl+L.f[m+1].impl+P.phi_one_eval(L.f[m].exp,L.dt*self.coll.delta_m[m],L.time))
            integral[m] -= L.dt*self.coll.delta_m[m]*(L.f[m].expl+L.f[m+1].impl+P.phi_one_f_eval(L.u[m],L.dt*self.coll.delta_m[m],L.time))

        # do the sweep
        for m in range(M):

            # rhs = L.u[m] + integral[m] + L.dt*self.coll.delta_m[m]*(L.f[m].expl+P.phi_one_eval(L.f[m].exp,L.dt*self.coll.delta_m[m],L.time))
            rhs = L.u[m] + integral[m] + L.dt*self.coll.delta_m[m]*(L.f[m].expl+P.phi_one_f_eval(L.u[m],L.dt*self.coll.delta_m[m],L.time))

            # implicit solve with prefactor stemming from QI
            L.u[m + 1] = P.solve_system(
                rhs, L.dt * self.QI[m + 1, m + 1], L.u[m + 1], L.time + L.dt * self.coll.nodes[m]
            )

            # update function values
            L.f[m + 1] = P.eval_f(L.u[m + 1], L.time + L.dt * self.coll.nodes[m])

        # # indicate presence of new values at this level
        L.status.updated = True

        return None

    def compute_end_point(self):
        """
        Compute u at the right point of the interval

        The value uend computed here is a full evaluation of the Picard formulation unless do_full_update==False

        Returns:
            None
        """

        # get current level and problem description
        L = self.level
        P = L.prob

        # check if Mth node is equal to right point and do_coll_update is false, perform a simple copy
        if self.coll.right_is_node and not self.params.do_coll_update:
            # a copy is sufficient
            L.uend = P.dtype_u(L.u[-1])
        else:
            raise CollocationError('This option is not implemented yet.')
            # start with u0 and add integral over the full interval (using coll.weights)
            L.uend = P.dtype_u(L.u[0])
            for m in range(self.coll.num_nodes):
                L.uend += L.dt * self.coll.weights[m] * (L.f[m + 1].impl + L.f[m + 1].expl)
            # add up tau correction of the full interval (last entry)
            if L.tau[-1] is not None:
                L.uend += L.tau[-1]

        return None
