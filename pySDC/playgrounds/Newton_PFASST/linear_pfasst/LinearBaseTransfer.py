from pySDC.core.Errors import UnlockError
from pySDC.core.BaseTransfer import base_transfer


class linear_base_transfer(base_transfer):
    """
    Standard base_transfer class

    Attributes:
        logger: custom logger for sweeper-related logging
        params(__Pars): parameter object containing the custom parameters passed by the user
        fine (pySDC.Level.level): reference to the fine level
        coarse (pySDC.Level.level): reference to the coarse level
    """

    def __init__(self, fine_level, coarse_level, base_transfer_params, space_transfer_class, space_transfer_params):
        """
        Initialization routine

        """

        super(linear_base_transfer, self).__init__(fine_level, coarse_level, base_transfer_params, space_transfer_class,
                                                   space_transfer_params)

    def restrict(self):
        """
        Space-time restriction routine

        """

        # get data for easier access
        F = self.fine
        G = self.coarse

        PG = G.prob

        SF = F.sweep
        SG = G.sweep

        # only if the level is unlocked at least by prediction
        if not F.status.unlocked:
            raise UnlockError('fine level is still locked, cannot use data from there')

        # build fine level integral
        intF = F.sweep.integrate()

        # restrict fine values in space
        tmp_res = []
        for m in range(SF.coll.num_nodes):
            # TODO: how to make sure communication did happen before the following statement?
            res = F.rhs[m] - (F.u[m + 1] - intF[m] - F.u[0])
            tmp_res.append(self.space_transfer.restrict(res))

        # restrict collocation values
        for n in range(SG.coll.num_nodes):
            G.rhs[n] = PG.dtype_u(PG.init, val=0.0)
            for m in range(SF.coll.num_nodes):
                G.rhs[n] += self.Rcoll[n, m] * tmp_res[m]
            G.u[n + 1] = PG.dtype_u(PG.init, val=0.0)
            G.f[n + 1] = PG.dtype_f(PG.init, val=0.0)  # This is zeros because we are in the linear case!

        # TODO: do we need G.u[0]?
        G.u[0] = self.space_transfer.restrict(F.u[0])
        G.f[0] = PG.eval_f(G.u[0], F.time)

        # works as a predictor
        G.status.unlocked = True

        return None

    def prolong(self):
        """
        Space-time prolongation routine

        """

        # get data for easier access
        F = self.fine
        G = self.coarse

        PF = F.prob

        SF = F.sweep
        SG = G.sweep

        # only of the level is unlocked at least by prediction or restriction
        if not G.status.unlocked:
            raise UnlockError('coarse level is still locked, cannot use data from there')

        # build coarse correction

        # interpolate values in space first
        tmp_u = []
        for m in range(1, SG.coll.num_nodes + 1):
            tmp_u.append(self.space_transfer.prolong(G.u[m]))

        # interpolate values in collocation
        for n in range(SF.coll.num_nodes):
            for m in range(SG.coll.num_nodes):
                F.u[n + 1] += self.Pcoll[n, m] * tmp_u[m]

        # re-evaluate f on fine level
        # F.f[0] = PF.eval_f(F.u[0], F.time)
        for m in range(1, SF.coll.num_nodes + 1):
            F.f[m] = PF.eval_f(F.u[m], F.time + F.dt * SF.coll.nodes[m - 1])

        return None

    def prolong_f(self):
        """
        Space-time prolongation routine w.r.t. the rhs f

        """

        # get data for easier access
        F = self.fine
        G = self.coarse

        PG = G.prob

        SF = F.sweep
        SG = G.sweep

        # only of the level is unlocked at least by prediction or restriction
        if not G.status.unlocked:
            raise UnlockError('coarse level is still locked, cannot use data from there')

        raise NotImplementedError('f-prolongation has not been implemented yet')

