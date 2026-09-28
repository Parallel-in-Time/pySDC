from pySDC.core.hooks import Hooks
import matplotlib.pyplot as plt


class PlottingHook(Hooks):  # pragma: no cover
    """
    Base class for hooks that plot the solution with the problem's `plot` method, optionally saving each figure.
    """

    save_plot = None  # Supply a string to the path where you want to save
    live_plot = 1e-9  # Supply `None` if you don't want live plotting

    def __init__(self):
        """Start counting the plots from zero."""
        super().__init__()
        self.plot_counter = 0

    def pre_run(self, step, level_number):
        """
        Get the figure to plot into from the problem's ``get_fig`` method.

        Args:
            step (pySDC.Step.step): The current step
            level_number (int): Number of current level

        Returns:
            None
        """
        super().pre_run(step, level_number)
        prob = step.levels[level_number].prob
        self.fig = prob.get_fig()

    def plot(self, step, level_number, plot_ic=False):
        """
        Plot the initial conditions or the solution at the end of the step with the problem's ``plot`` method. The
        figure is saved to ``<save_plot>_<counter>.png`` if ``save_plot`` is set, and shown for ``live_plot`` seconds if
        that is not ``None``.

        Args:
            step (pySDC.Step.step): The current step
            level_number (int): Number of current level
            plot_ic (bool): plot the initial conditions ``u[0]`` instead of the end point

        Returns:
            None
        """
        level = step.levels[level_number]
        prob = level.prob

        if plot_ic:
            u = level.u[0]
            t = level.time
        else:
            level.sweep.compute_end_point()
            u = level.uend
            t = level.time + level.dt

        prob.plot(u=u, t=t, fig=self.fig)

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
        """Start counting the steps since the last plot from zero."""
        super().__init__()
        self.skip_counter = 0

    def pre_run(self, step, level_number):
        """
        Set up the figure and plot the initial conditions, on the finest level only.

        Args:
            step (pySDC.Step.step): The current step
            level_number (int): Number of current level

        Returns:
            None
        """
        if level_number > 0:
            # no figure on coarse levels, but the rest of the chain
            return super(PlottingHook, self).pre_run(step, level_number)
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
        super().post_step(step, level_number)
        if level_number > 0:
            return

        self.skip_counter += 1

        if self.skip_counter % self.plot_every >= 1:
            return

        self.plot(step, level_number)
