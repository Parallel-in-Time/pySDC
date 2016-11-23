from __future__ import division
from pySDC.core.Hooks import hooks


class gmres_tolerance(hooks):

    def pre_iteration(self, step, level_number):
        """
        Routine called before iteration starts, set new GMRES tolerance depending on the initial SDC residual

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        super(gmres_tolerance, self).pre_iteration(step, level_number)

        L = step.levels[level_number]
        L.sweep.compute_residual()
        L.prob.gmres_tol_limit = max(L.status.residual * L.prob.gmres_tol_factor, L.prob.gmres_tol_limit)

    def post_sweep(self, step, level_number):
        """
        Routine called after each sweep, set new GMRES tolerance depending on the previous SDC residual

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """

        super(gmres_tolerance, self).post_sweep(step, level_number)

        L = step.levels[level_number]

        L.prob.gmres_tol_limit = max(L.status.residual * L.prob.gmres_tol_factor, L.prob.gmres_tol_limit)
