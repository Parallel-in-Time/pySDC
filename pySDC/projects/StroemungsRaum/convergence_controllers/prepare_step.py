from pySDC.core.convergence_controller import ConvergenceController


class PrepareStep(ConvergenceController):
    """
    Tell the problems of every level about the step before its first solve, by calling their
    ``prepare_step(t0, dt, coll)`` with the start of the step, its size and the sweeper's collocation rule.

    The sweepers tell a problem only the time of each solve. A problem that needs the whole step can get it here instead
    of from a sweeper of its own: the projection method of
    :class:`~pySDC.projects.StroemungsRaum.problem_classes.NavierStokes_2D_projection_Firedrake.NavierStokes2DProjectionFiredrake`
    needs the time from the start of the step to the node, and the differentiated boundary conditions of
    :class:`~pySDC.projects.StroemungsRaum.problem_classes.NavierStokes_2D_TaylorGreen_monolithic_Firedrake.NavierStokes2DTaylorGreenFiredrake`
    need the collocation rule. Add it to the description with ``'convergence_controllers': {PrepareStep: {}}``.

    It acts after the spreading of the initial value, which evaluates right-hand sides but solves nothing, so the
    problems of all levels know the step before the first sweep, and again after a restart.
    """

    def post_spread_processing(self, controller, S, **kwargs):
        """
        Hand the step to the problem of every level.

        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.Step): The current step

        Returns:
            None
        """
        for L in S.levels:
            L.prob.prepare_step(L.time, L.dt, L.sweep.coll)
