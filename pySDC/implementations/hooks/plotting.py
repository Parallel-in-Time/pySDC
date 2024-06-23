from pySDC.core.hooks import Hooks
import matplotlib.pyplot as plt


class PlottingHook(Hooks):  # pragma: no cover
    save_plot = None  # Supply a string to the path where you want to save
    live_plot = 1e-9  # Supply `None` if you don't want live plotting

    def __init__(self):
        super().__init__()
        self.plot_counter = 0

    def pre_run(self, step, level_number):
        prob = step.levels[level_number].prob
        self.fig = prob.get_fig()

    def plot(self, step, level_number, plot_ic=False):
        level = step.levels[level_number]
        prob = level.prob

        if plot_ic:
            u = level.u[0]
        else:
            level.sweep.compute_end_point()
            u = level.uend

        prob.plot(u=u, t=step.time, fig=self.fig)

        if self.save_plot is not None:
            path = f'{self.save_plot}_{self.plot_counter:04d}.png'
            self.fig.savefig(path, dpi=100)
            self.logger.log(25, f'Saved figure {path!r}.')

        if self.live_plot is not None:
            plt.pause(self.live_plot)

        self.plot_counter += 1


class PlotPostStep(PlottingHook):  # pragma: no cover
    """
    Call a plotting function of the problem after every step
    """

    plot_every = 1

    def __init__(self):
        super().__init__()
        self.skip_counter = 0

    def pre_run(self, step, level_number):
        if level_number > 0:
            return
        super().pre_run(step, level_number)
        self.plot(step, level_number, plot_ic=True)

    def post_step(self, step, level_number):
        """
        Call the plotting function after the step

        Args:
            step (pySDC.Step.step): The current step
            level_number (int): Number of current level

        Returns:
            None
        """
        if level_number > 0:
            return

        self.skip_counter += 1

        if self.skip_counter % self.plot_every >= 1:
            return

        self.plot(step, level_number)
