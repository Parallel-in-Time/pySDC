import numpy as np
from pySDC.Sweeper import sweeper
import math

import pySDC.Plugins.fault_tolerance as ft

class imex_1st_order(sweeper):
    """
    Custom sweeper class, implements Sweeper.py

    First-order IMEX sweeper using implicit/explicit Euler as base integrator

    Attributes:
        QI: implicit Euler integration matrix
        QE: explicit Euler integration matrix
    """

    def __init__(self,coll):
        """
        Initialization routine for the custom sweeper

        Args:
            coll: collocation object
        """

        # call parent's initialization routine
        super(imex_1st_order,self).__init__(coll)

        # IMEX integration matrices
        [self.QI, self.QE] = self.__get_Qd


    @property
    def __get_Qd(self):
        """
        Sets the integration matrices QI and QE for the IMEX sweeper

        Returns:
            QI: implicit Euler matrix, will also act on u0
            QE: explicit Euler matrix, will also act on u0
        """
        QI = np.zeros(np.shape(self.coll.Qmat))
        QE = np.zeros(np.shape(self.coll.Qmat))
        for m in range(self.coll.num_nodes + 1):
            QI[m, 1:m+1] = self.coll.delta_m[0:m]
            QE[m, 0:m] = self.coll.delta_m[0:m]

        return QI, QE


    def integrate(self):
        """
        Integrates the right-hand side (here impl + expl)

        Returns:
            list of dtype_u: containing the integral as values
        """

        # get current level and problem description
        L = self.level
        P = L.prob

        me = []

        # integrate RHS over all collocation nodes
        for m in range(1,self.coll.num_nodes+1):
            # new instance of dtype_u, initialize values with 0
            me.append(P.dtype_u(P.init,val=0))
            for j in range(1,self.coll.num_nodes+1):
                me[-1] += L.dt*self.coll.Qmat[m,j]*(L.f[j].impl + L.f[j].expl)

        return me

    def update_nodes(self,doitnow):
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

        # gather all terms which are known already (e.g. from the previous iteration)
        # this corresponds to u0 + QF(u^k) - QIFI(u^k) - QEFE(u^k) + tau

         # get QF(u^k)
        integral = self.integrate()
        for m in range(M):
            # subtract QIFI(u^k)_m - QEFE(u^k)_m
            for j in range(M+1):
                integral[m] -= L.dt*(self.QI[m+1,j]*L.f[j].impl + self.QE[m+1,j]*L.f[j].expl)
            # add initial value
            integral[m] += L.u[0]
            # add tau if associated
            if L.tau is not None:
                integral[m] += L.tau[m]

        oldres = np.zeros(M)
        res = self.integrate()
        for m in range(M):
             # add u0 and subtract u at current node
            res[m] += L.u[0] - L.u[m+1]
            # add tau if associated
            if L.tau is not None:
                res[m] += L.tau[m]
            # use abs function from data type here
            oldres[m] = abs(res[m])

        # do the sweep
        for m in range(0,M):

            # check if we will do a bitflip
            loop,index,pos,uf = ft.soft_fault_injection(P.nvars)
            flip = loop == 0 and index is not None and m > 0# and doitnow and m+1==4

            # repeat the evaluation at this node, until either we fixed a bitflip or we have done this twice already
            while loop < 2:

                # build rhs, consisting of the known values from above and new values from previous nodes (at k+1)
                rhs = P.dtype_u(integral[m])
                for j in range(m+1):
                    rhs += L.dt*(self.QI[m+1,j]*L.f[j].impl + self.QE[m+1,j]*L.f[j].expl)

                # implicit solve with prefactor stemming from QI
                L.u[m+1] = P.solve_system(rhs,L.dt*self.QI[m+1,m+1],L.u[m+1],L.time+L.dt*self.coll.nodes[m])
                # update function values
                L.f[m+1] = P.eval_f(L.u[m+1],L.time+L.dt*self.coll.nodes[m])

                if flip:

                    print('Flipping bit at node %i, index %i, position %i, uf %i...' %(m+1,index,pos,uf))

                    if uf == 0:
                        # this is for flips in u
                        print('pre:',L.u[m+1].values[index])
                        L.u[m+1].values[index] = ft.do_bitflip(L.u[m+1].values[index],pos)
                        print('post:',L.u[m+1].values[index])
                    elif uf == 1:
                        # this is for flips in f.impl 
                        print('pre:',L.f[m+1].impl.values[index])
                        L.f[m+1].impl.values[index] = ft.do_bitflip(L.f[m+1].impl.values[index],pos)
                        print('post:',L.f[m+1].impl.values[index])
                    else:
                        print('oooh, too bad..',uf)
                        exit()
                    flip = False


                # compute the residual with the new values just computed
                res = P.dtype_u(L.u[0])
                for j in range(1,self.coll.num_nodes+1):
                    res += L.dt*self.coll.Qmat[m+1,j]*(L.f[j].impl + L.f[j].expl)
                res -= L.u[m+1]
                newres = np.linalg.norm(res.values,np.inf)
                # print(newres,oldres[m])

                # first take on this node and the residual is too high
                if loop == 0 and (newres > 10*oldres[m] or math.isnan(newres)):
                    print('bad things happened, will repeat this step...',m,newres,oldres[m])
                    loop = 1
                # second take on this node and the residual is still too high
                elif loop == 1 and newres > 10*oldres[m]:
                    print('this was a false positive!',newres,oldres[m])
                    loop = 2
                # all is good
                else:
                    loop = 2


        # indicate presence of new values at this level
        L.status.updated = True

        return None


    def compute_end_point(self):
        """
        Compute u at the right point of the interval

        The value uend computed here might be a simple copy from u[M] (if right point is a collocation node) or
        a full evaluation of the Picard formulation (if right point is not a collocation node)
        """

        # get current level and problem description
        L = self.level
        P = L.prob

        # check if Mth node is equal to right point (flag is set in collocation class)
        if self.coll.right_is_node:
            # a copy is sufficient
            L.uend = P.dtype_u(L.u[-1])
        else:
            # start with u0 and add integral over the full interval (using coll.weights)
            L.uend = P.dtype_u(L.u[0])
            for m in range(self.coll.num_nodes):
                L.uend += L.dt*self.coll.weights[m]*(L.f[m+1].impl + L.f[m+1].expl)
            # add up tau correction of the full interval (last entry)
            if L.tau is not None:
                L.uend += L.tau[-1]

        return None
