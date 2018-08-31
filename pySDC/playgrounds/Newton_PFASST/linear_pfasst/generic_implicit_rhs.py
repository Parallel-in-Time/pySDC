from pySDC.core.Sweeper import sweeper

from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit


class generic_implicit_rhs(generic_implicit):
    """
    Generic implicit sweeper, expecting lower triangular matrix type as input, has RHS to consider

    """

    def integrate(self):
        """
        Integrates the right-hand side

        Returns:
            list of dtype_u: containing the integral as values
        """

        # get current level and problem description
        L = self.level
        P = L.prob

        me = []

        # integrate f over all collocation nodes
        for m in range(1, self.coll.num_nodes + 1):
            # new instance of dtype_u, initialize values with 0
            me.append(P.dtype_u(P.init, val=0))
            for j in range(1, self.coll.num_nodes + 1):
                me[-1] += L.dt * self.coll.Qmat[m, j] * L.f[j]
            # This is a dirty trick, since the rhs does not belong to the integral. However, this saves a lot of
            # work/code, since whenever rhs needs to be added to something, the integral has to be/was added, too.
            me[-1] += L.rhs[m - 1]

        return me