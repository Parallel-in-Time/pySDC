import matplotlib.pyplot as plt
# import progressbar

from pySDC.core.Hooks import hooks


class trajectories(hooks):

    def __init__(self):
        """
        Initialization of particles output
        """
        super(trajectories, self).__init__()

        fig = plt.figure()
        self.ax = fig.add_subplot(111)
        self.sframe = None
        # self.bar_run = None

    def pre_run(self, step, level_number):
        """
        Overwrite default routine called before time-loop starts
        Args:
            step: the current step
            level_number: the current level number
        """
        super(trajectories, self).pre_run(step, level_number)

        # some abbreviations
        L = step.levels[level_number]

        # if hasattr(L.prob.params, 'Tend'):
        #     self.bar_run = progressbar.ProgressBar(max_value=L.prob.params.Tend)
        # else:
        #     self.bar_run = progressbar.ProgressBar(max_value=progressbar.UnknownLength)

    def post_step(self, step, level_number):
        """
        Default routine called after each iteration
        Args:
            step: the current step
            level_number: the current level number
        """

        super(trajectories, self).post_step(step, level_number)

        # some abbreviations
        L = step.levels[level_number]

        # self.bar_run.update(L.time)

        L.sweep.compute_end_point()

        # oldcol = self.sframe

        self.sframe = self.ax.scatter(L.uend[0], L.uend[1])
        # Remove old line collection before drawing
        # if oldcol is not None:
        #     self.ax.collections.remove(oldcol)
        plt.pause(0.001)

        return None
