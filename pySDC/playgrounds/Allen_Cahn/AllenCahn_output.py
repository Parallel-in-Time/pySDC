import pySDC.helpers.plot_helper as plt_helper
from pySDC.core.hooks import Hooks


class output(Hooks):
    def __init__(self):
        """
        Initialization of Allen-Cahn monitoring
        """
        super(output, self).__init__()

        plt_helper.setup_mpl()

        self.counter = 0
        self.fig = None
        self.ax = None
        self.output_ratio = 1

    def pre_run(self, step, level_number):
        super(output, self).pre_step(step, level_number)
        # some abbreviations
        L = step.levels[0]

        self.counter = 0
        if self.counter % self.output_ratio == 0:
            self.fig, self.ax = plt_helper.newfig(textwidth=238, scale=1.0, ratio=1.0)

            im1 = self.ax.imshow(L.u[0], vmin=L.prob.params.rest, vmax=L.prob.params.depol)

            self.fig.colorbar(im1, ax=self.ax)
            fname = 'data/AC_' + L.prob.params.init_type + '_output_' + str(self.counter).zfill(8)
            plt_helper.savefig(fname, save_pgf=False, save_pdf=False, save_png=True)

    def post_step(self, step, level_number):
        """
        Overwrite standard post step hook

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        super(output, self).post_step(step, level_number)

        # some abbreviations
        L = step.levels[0]

        self.counter += 1
        if self.counter % self.output_ratio == 0:
            self.fig, self.ax = plt_helper.newfig(textwidth=238, scale=1.0, ratio=1.0)

            im1 = self.ax.imshow(L.uend, vmin=L.prob.params.rest, vmax=L.prob.params.depol)

            self.fig.colorbar(im1, ax=self.ax)
            fname = 'data/AC_' + L.prob.params.init_type + '_output_' + str(self.counter).zfill(8)
            plt_helper.savefig(fname, save_pgf=False, save_pdf=False, save_png=True)
