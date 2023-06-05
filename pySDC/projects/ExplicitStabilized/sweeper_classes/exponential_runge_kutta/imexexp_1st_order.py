import numpy as np

from pySDC.core.Sweeper import sweeper
from pySDC.core.Errors import CollocationError
import numdifftools.fornberg as fornberg

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

        M = self.coll.num_nodes
        
        # this works for M==1
        # lmbda = P.lmbda_eval(L.u[0],L.time)
        # phi_one = P.phi_eval(L.u[0],L.dt,L.time,1)        
        # me.append( L.dt*( L.f[1].impl + L.f[1].expl +  phi_one.__rmul__(L.f[1].exp+lmbda.__rmul__(L.u[0]-L.u[1]))) )        
        # return me
    
        # compute phi[k][i] = phi_{k}(dt*c_i*lmbda), i=0,...,M-1, k=0,...,M
        phi = []
        c = self.coll.nodes
        for k in range(M+1):
            phi.append([])
            for i in range(M):
                phi[k].append(P.phi_eval(L.u[0],L.dt*c[i],L.time,k))

        # compute polynomial. remember that L.u[k+1] corresponds to c[k]
        lmbda = P.lmbda_eval(L.u[0],L.time)
        Q = []
        for k in range(M):
            Q.append(L.f[k+1].exp+lmbda*(L.u[0]-L.u[k+1]))

        # compute weights w such that PiQ^(k)(0) = sum_{j=0}^{M-1} w[k,j]*Q[j], k=0,...,M-1
        w = fornberg.fd_weights_all(c,0.,M-1)

        # compute weight for the integration of \int_0^ci exp(dt*(ci-r)lmbda)*PiQ(r)dr = \sum_{j=0}^{M-1} Qmat_exp[i,j]*Q[j]
        Qmat_exp = [[P.dtype_u(P.init,val=0.) for j in range(M)] for i in range(M)]
        for i in range(M):            
            for j in range(M):
                for k in range(M):
                    Qmat_exp[i][j] += (w[k,j]*c[i]**(k+1))*phi[k+1][i]

        # integrate RHS over all collocation nodes
        for m in range(1, self.coll.num_nodes + 1):
            me.append(L.dt * (self.coll.Qmat[m, 1] * (L.f[1].impl + L.f[1].expl) + Qmat_exp[m-1][0]*(Q[0])))
            # new instance of dtype_u, initialize values with 0
            for j in range(2, self.coll.num_nodes + 1):
                me[m - 1] += L.dt * (self.coll.Qmat[m, j] * (L.f[j].impl + L.f[j].expl) + Qmat_exp[m-1][j-1]*(Q[j-1]))

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

        integral = self.integrate()
        for i in range(1,M):
            integral[M-i] -= integral[M-i-1]

        if L.tau[0] is not None:
            integral[0] += L.tau[0]
        for m in range(1,M):
            if L.tau[m] is not None:
                integral[m] += L.tau[m]-L.tau[m-1]

        lmbda = P.lmbda_eval(L.u[0],L.time)
        phi_one = []
        for m in range(M):
            phi_one.append(P.phi_eval(L.u[0],L.dt*self.coll.delta_m[m],L.time,1))

        for m in range(M):
            integral[m] -= L.dt*self.coll.delta_m[m]*(L.f[m].expl+L.f[m+1].impl+phi_one[m]*(L.f[m].exp+lmbda*(L.u[0]-L.u[m])) )

        # do the sweep
        for m in range(M):

            rhs = L.u[m] + integral[m] + L.dt*self.coll.delta_m[m]*(L.f[m].expl+phi_one[m]*(L.f[m].exp+lmbda*(L.u[0]-L.u[m])))   

            # implicit solve with prefactor stemming from QI
            L.u[m + 1] = P.solve_system(
                rhs, L.dt * self.QI[m + 1, m + 1], L.u[m + 1], L.time + L.dt * self.coll.nodes[m]
            )

            # update function values
            L.f[m + 1] = P.eval_f(L.u[m + 1], L.time + L.dt * self.coll.nodes[m])

        # # indicate presence of new values at this level
        L.status.updated = True

        return None

    # def update_nodes(self):
    #     """
    #     Update the u- and f-values at the collocation nodes -> corresponds to a single sweep over all nodes

    #     Returns:
    #         None
    #     """

    #     # get current level and problem description
    #     L = self.level
    #     P = L.prob

    #     # only if the level has been touched before
    #     assert L.status.unlocked

    #     # get number of collocation nodes for easier access
    #     M = self.coll.num_nodes

    #     # gather all terms which are known already (e.g. from the previous iteration)
    #     # this corresponds to u0 + QF(u^k) - QIFI(u^k) - QEFE(u^k) + tau

    #     lmbda = P.lmbda_eval(L.u[0],L.time)
    #     phi_one = []
    #     for m in range(M):
    #         phi_one.append(P.phi_eval(L.u[0],L.dt*self.coll.delta_m[m],L.time,1))
    #     phi_one.append(P.dtype_u(P.init,0.))

    #     # get QF(u^k)
    #     integral = self.integrate()
    #     for m in range(M):
    #         # subtract QIFI(u^k)_m + QEFE(u^k)_m
    #         for j in range(1, M + 1):
    #             integral[m] -= L.dt * (self.QI[m + 1, j] * L.f[j].impl + self.QE[m + 1, j] * L.f[j].expl + self.QE[m + 1, j]*phi_one[j].__rmul__(L.f[j].exp+lmbda.__rmul__(L.u[0]-L.u[j])))
    #         # add initial value
    #         integral[m] += L.u[0]
    #         # add tau if associated
    #         if L.tau[m] is not None:
    #             integral[m] += L.tau[m]

    #     # do the sweep
    #     for m in range(0, M):
    #         # build rhs, consisting of the known values from above and new values from previous nodes (at k+1)
    #         rhs = P.dtype_u(integral[m])
    #         for j in range(1, m + 1):
    #             rhs += L.dt * (self.QI[m + 1, j] * L.f[j].impl + self.QE[m + 1, j] * L.f[j].expl+ self.QE[m + 1, j]*phi_one[j].__rmul__(L.f[j].exp+lmbda.__rmul__(L.u[0]-L.u[j])))

    #         # implicit solve with prefactor stemming from QI
    #         L.u[m + 1] = P.solve_system(
    #             rhs, L.dt * self.QI[m + 1, m + 1], L.u[m + 1], L.time + L.dt * self.coll.nodes[m]
    #         )

    #         # update function values
    #         L.f[m + 1] = P.eval_f(L.u[m + 1], L.time + L.dt * self.coll.nodes[m])

    #     # indicate presence of new values at this level
    #     L.status.updated = True

    #     return None

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
