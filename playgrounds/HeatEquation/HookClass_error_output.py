from __future__ import division
from pySDC.core.Hooks import hooks


class error_output(hooks):
    """
    Hook class to add output of error
    """

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

        uex = P.u_exact(step.time + step.dt)
        err = abs(uex - L.uend)
        print(step.status.iter, err, L.status.residual)
