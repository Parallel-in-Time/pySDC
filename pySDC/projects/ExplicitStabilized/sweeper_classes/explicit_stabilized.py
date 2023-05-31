import numpy as np

from pySDC.core.Sweeper import sweeper
from pySDC.core.Errors import CollocationError
from pySDC.projects.ExplicitStabilized.explicit_stabilized_classes.rho_estimator import rho_estimator
from pySDC.core.Lagrange import LagrangeApproximation

class explicit_stabilized(sweeper):
    """
    Custom sweeper class, implements Sweeper.py

    First-order stabilized exponential sweeper using explicit stabilized methods and exponential Euler as base integrators

    """

    def __init__(self, params):
        """
        Initialization routine for the custom sweeper

        Args:
            params: parameters for the sweeper
        """

        # call parent's initialization routine
        super(explicit_stabilized, self).__init__(params)

        self.damping = params['damping'] if 'damping' in params else 0.05
        self.safe_add = params['safe_add'] if 'safe_add' in params else 1

        self.es = self.init_es(params)
        self.s_prev = [0]*self.coll.num_nodes
        
        self.sub_nodes = [None]*self.coll.num_nodes
        self.interp_mat = [None]*self.coll.num_nodes
        self.interp_mat0 = [None]*self.coll.num_nodes
        
        self.nodes = self.coll.nodes
        self.nodes0 = np.array([0.])
        self.nodes0 = np.append(self.nodes0,self.nodes)
        self.lagrange = LagrangeApproximation(self.nodes)                         
        self.lagrange0 = LagrangeApproximation(self.nodes0)

        # updates spectral radius every rho_freq steps
        self.rho_freq = params['rho_freq'] if 'rho_freq' in params else 5
        self.rho_count = 0

        # self.with_node_zero = True

    def init_es(self,params):
        return [params['es_class'](self.damping,self.safe_add) for _ in range(self.coll.num_nodes)]
    
    def update_stages_coefficients(self):
        
        L = self.level
        P = L.prob
        M = self.coll.num_nodes

        if not hasattr(self,'rho'):
            if hasattr(P.exact,'rho') and callable(P.exact.rho):
                self.rho = P.exact.rho
            else:
                self.rho_estimator = rho_estimator(P)
                self.rho = self.rho_estimator.rho
        
        if self.rho_count % self.params.rho_freq == 0:
            self.estimated_rho = self.rho(y=L.u[0],t=L.time,fy=L.f[0])            
            self.rho_count = 0
            self.s = [0]*M
            for m in range(M):
                rho_m = self.estimated_rho
                self.s[m] = self.es[m].get_s(L.dt*self.coll.delta_m[m]*rho_m)
                if self.s[m]!=self.s_prev[m]:
                    self.es[m].update_coefficients(self.s[m])
                    self.sub_nodes[m] = self.coll.nodes[m]+(self.es[m].c-1.)*self.coll.delta_m[m]
                    self.interp_mat[m] = self.lagrange.getInterpolationMatrix(self.sub_nodes[m])
                    self.interp_mat0[m] = self.lagrange0.getInterpolationMatrix(self.sub_nodes[m])
            self.s_prev = self.s[:]

        self.rho_count += 1

    def integrate(self):
        """
        Integrates the right-hand side (here impl + expl + exp)

        Returns:
            list of dtype_u: containing the integral as values
        """

        # get current level and problem description
        L = self.level

        me = []

        # integrate RHS over all collocation nodes
        for m in range(1, self.coll.num_nodes + 1):
            me.append(L.dt * self.coll.Qmat[m, 1] * L.f[1])
            # new instance of dtype_u, initialize values with 0
            for j in range(2, self.coll.num_nodes + 1):
                me[m - 1] += L.dt * self.coll.Qmat[m, j] * L.f[j]

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

        self.update_stages_coefficients()

        integral = self.integrate()
        for m in range(M):
            integral[m] += L.u[0]
            if L.tau[m] is not None:
                integral[m] += L.tau[m]

        u_old = [P.dtype_u(u) for u in L.u]

        integral.insert(0,L.u[0])

        print(f's = {[es.s for es in self.es]}')

        self.dj = P.dtype_u(P.init,0.)
        self.djm1 = P.dtype_u(P.init,0.)
        self.djm2 = P.dtype_u(P.init,0.)

        def print_list_norms(lst,name):
            for i,l in enumerate(lst):
                print(f'|{name}[{i}]|={abs(l)}')

        # print_list_norms(integral,'integral')
        
        # do the sweep
        for m in range(M):

            dt = L.dt*self.coll.delta_m[m]
            es = self.es[m]
            sub_nodes = self.sub_nodes[m]     

            self.djm1.copy(self.dj)
            f1 = P.eval_f(integral[m]+self.djm1, L.time + L.dt * (self.coll.nodes[m]-self.coll.delta_m[m]))
            f2 = P.eval_f(u_old[m], L.time + L.dt * (self.coll.nodes[m]-self.coll.delta_m[m]))
            self.dj = self.djm1 + dt*es.mu[0]*(f1-f2)
            for j in range(2,es.s+1):
                self.djm2, self.djm1, self.dj = self.djm1, self.dj, self.djm2            

                I_int = self.interpolate(integral, m, j-2)
                I_u_old = self.interpolate(u_old, m, j-2)
                
                f1 = P.eval_f(I_int+self.djm1, L.time + L.dt * sub_nodes[j-2])
                f2 = P.eval_f(I_u_old, L.time + L.dt * sub_nodes[j-2])
                self.dj = f1-f2
                self.dj *= dt*es.mu[j-1]
                self.dj += es.nu[j-1]*self.djm1+es.kappa[j-1]*self.djm2

            L.u[m+1] = self.dj+integral[m+1]
            L.f[m + 1] = P.eval_f(L.u[m + 1], L.time + L.dt * self.coll.nodes[m])

        # # indicate presence of new values at this level
        L.status.updated = True

        return None
    
    def interpolate(self,val,m,j):
        interp_mat = self.interp_mat0[m]
        I_val = interp_mat[j,0]*val[0]
        for i in range(1,len(val)):
            I_val += interp_mat[j,i]*val[i]

        return I_val

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
