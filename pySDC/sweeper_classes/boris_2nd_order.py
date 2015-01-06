import numpy as np

from pySDC.Sweeper import sweeper


class boris_2nd_order(sweeper):
    """
    Custom sweeper class, implements Sweeper.py

    Second-order sweeper using velocity-Verlet with Boris scheme as base integrator

    Attributes:
        S: node-to-node collocation matrix (first order)
        SQ: node-to-node collocation matrix (second order)
        ST: node-to-node trapezoidal matrix
        Sx: node-to-node Euler half-step for position update
    """

    def __init__(self,coll):
        """
        Initialization routine for the custom sweeper

        Args:
            coll: collocation object
        """

         # call parent's initialization routine
        super(boris_2nd_order,self).__init__(coll)

        # S- and SQ-matrices (derived from Q) and Sx- and ST-matrices for the integrator
        [self.S, self.ST, self.SQ, self.Sx] = self.__get_Qd(coll)

    def __get_Qd(self,coll):
        """
        Compute LU decomposition of Q^T

        Args:
            coll: collocation object
        Returns:
            S: node-to-node collocation matrix (first order)
            SQ: node-to-node collocation matrix (second order)
            ST: node-to-node trapezoidal matrix
            Sx: node-to-node Euler half-step for position update
        """

        # set implicit and explicit Euler matrices
        QI = np.zeros(np.shape(coll.Qmat))
        QE = np.zeros(np.shape(coll.Qmat))
        for m in np.arange(coll.num_nodes+1):
            QI[m, 1:m+1] = coll.delta_m[0:m]
            QE[m, 0:m] = coll.delta_m[0:m]
        # trapezoidal rule
        QT = 1/2*(QI+QE)
        # Qx as in the paper
        Qx = np.dot(QE,QT) + 1/2*QE*QE

        Sx = np.zeros(np.shape(coll.Qmat))
        ST = np.zeros(np.shape(coll.Qmat))
        S = np.zeros(np.shape(coll.Qmat))
        # fill-in node-to-node matrices
        Sx[0,:] = Qx[0,:]
        ST[0,:] = QT[0,:]
        S[0,:] = coll.Qmat[0,:]
        for m in range(coll.num_nodes):
            Sx[m+1,:] = Qx[m+1,:] - Qx[m,:]
            ST[m+1,:] = QT[m+1,:] - QT[m,:]
            S[m+1,:] = coll.Qmat[m+1,:] - coll.Qmat[m,:]
        # SQ via dot-product, could also be done via QQ
        SQ = np.dot(S,coll.Qmat)
        return [S,ST,SQ,Sx]


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

        # initialize integral terms with zeros, will add stuff later
        integral = [P.dtype_u(P.init,vals=(0,0,0,0)) for l in range(M)]

        # gather all terms which are known already (e.g. from the previous iteration)
        # this corresponds to SF(u^k) - SdF(u^k) + tau (note: have integrals in pos and vel!)
        for m in range(M):
            for j in range(M+1):
                # build RHS from f-terms (containing the E field) and the B field
                f = P.build_f(L.f[j],L.u[j],L.time+L.dt*self.coll.nodes[j-1])
                # add SQF(u^k) - SxF(u^k) for the position
                integral[m].pos += L.dt*(L.dt*(self.SQ[m+1,j]-self.Sx[m+1,j])*f)
                # add SF(u^k) - STF(u^k) for the velocity
                integral[m].vel += L.dt*(self.S[m+1,j]-self.ST[m+1,j])*f
            # add tau if associated
            if L.tau is not None:
                integral[m] += L.tau[m]

        # do the sweep
        for m in range(0,M):
            # build rhs, consisting of the known values from above and new values from previous nodes (at k+1)
            tmp = P.dtype_u(integral[m])
            for j in range(m+1):
                # build RHS from f-terms (containing the E field) and the B field
                f = P.build_f(L.f[j],L.u[j],L.time+L.dt*self.coll.nodes[j-1])
                # add SxF(u^{k+1})
                tmp.pos += L.dt*(L.dt*self.Sx[m+1,j]*f)
            # add pos at previous node + dt*v0
            tmp.pos += L.u[m].pos + L.dt*self.coll.delta_m[m]*L.u[0].vel
            # set new position, is explicit
            L.u[m+1].pos = tmp.pos

            # get E field with new positions and compute mean
            L.f[m+1] = P.eval_f(L.u[m+1],L.time+L.dt*self.coll.nodes[m])

            ck = tmp.vel

            # do the boris scheme
            L.u[m+1].vel = P.boris_solver(ck,L.dt*self.coll.delta_m[m],L.f[m],L.f[m+1],L.u[m].vel)


        # indicate presence of new values at this level
        L.status.updated = True

        return None


    def integrate(self,weights):
        """
        Integrates the right-hand side

        Args:
            weights: integration weights, length num_nodes
        Returns:
            dtype_u: containing the integral as values
        """

        # get current level and problem description
        L = self.level
        P = L.prob

        # create new instance of dtype_u, initialize values with 0
        p = P.dtype_u(P.init,vals=(0,0,0,0))

         # integrate RHS over all collocation nodes, RHS is here v and f(x,v)
        for j in range(self.coll.num_nodes):
            f = P.build_f(L.f[j+1],L.u[j+1],L.time+L.dt*self.coll.nodes[j])
            p.pos += L.dt*weights[j]*L.u[j+1].vel
            p.vel += L.dt*weights[j]*f

        return p
