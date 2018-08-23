from pySDC.core.Hooks import hooks


class update_problem_params(hooks):

    def pre_step(self, step, level_number):
        """
        Hook called before each step

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        super(update_problem_params, self).pre_step(step, level_number)

        L = step.levels[level_number]

        L.prob.tn = L.time
        L.prob.tnp = L.time + L.dt

