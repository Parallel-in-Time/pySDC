from pySDC.core.Hooks import hooks


class error_output(hooks):
    """
    Hook class to add output of error
    """

    def pre_iteration(self, step, level_number):

        super(error_output, self).pre_iteration(step, level_number)

        # some abbreviations
        L = step.levels[level_number]
        P = L.prob

        for m in range(1, L.sweep.coll.num_nodes + 1):
            L.uold[m] = P.dtype_u(L.u[m])

    def post_iteration(self, step, level_number):
        """
        Default routine called after each iteration
        Args:
            step: the current step
            level_number: the current level number
        """

        super(error_output, self).post_iteration(step, level_number)

        # some abbreviations
        L = step.levels[level_number]
        P = L.prob

        L.sweep.compute_end_point()

        uex = P.u_exact(step.time + step.dt)
        err = abs(uex - L.uend)

        est = []

        for m in range(1, L.sweep.coll.num_nodes + 1):
            est.append(abs(L.uold[m] - L.u[m]))

        est_all = max(est)

        print(step.status.iter, err, est_all, L.status.residual)
