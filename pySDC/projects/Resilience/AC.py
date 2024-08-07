# script to run an Allen-Cahn problem
from pySDC.implementations.problem_classes.AllenCahn_2D_FFT import allencahn2d_imex
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.core.hooks import Hooks
from pySDC.projects.Resilience.hook import hook_collection, LogData
from pySDC.projects.Resilience.strategies import merge_descriptions
import matplotlib.pyplot as plt
import numpy as np

from pySDC.core.errors import ConvergenceError


class allencahn_imex_timeforcing_adaptivity(allencahn2d_imex):
    r"""
    Add more source terms to `allencahn_imex_timeforcing` such that the time-scale changes and we can benefit from adaptivity.
    """

    def __init__(self, time_freq=2.0, time_dep_strength=1e-2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._makeAttributeAndRegister('time_freq', 'time_dep_strength', localVars=locals(), readOnly=True)

    def eval_f(self, u, t):
        f = super().eval_f(u, t)
        time_mod = self.get_time_dep_fac(self.time_freq, self.time_dep_strength, t)

        if self.eps > 0:
            f.expl = -2.0 / self.eps**2 * u * (1.0 - u) * (1.0 - 2.0 * u)

        # build sum over RHS without driving force
        Rt = float(np.sum(f.impl + f.expl))

        # build sum over driving force term
        Ht = float(np.sum(6.0 * u * (1.0 - u)))

        # add/subtract time-dependent driving force
        if Ht != 0.0:
            dw = Rt / Ht * time_mod
        else:
            dw = 0.0

        f.expl -= 6.0 * dw * u * (1.0 - u)

        return f

    @staticmethod
    def get_time_dep_fac(time_freq, time_dep_strength, t):
        return 1 - time_dep_strength * np.sin(time_freq * 2 * np.pi / 0.032 * t)


class monitor(Hooks):
    phase_thresh = 0.0  # count everything above this threshold to the high phase.

    def __init__(self):
        """
        Initialization of Allen-Cahn monitoring
        """
        super().__init__()

        self.init_radius = None

    def get_exact_radius(self, t):
        return np.sqrt(max(self.init_radius**2 - 2.0 * t, 0))

    @classmethod
    def get_radius(cls, u, dx):
        c = np.count_nonzero(u > cls.phase_thresh)
        return np.sqrt(c / np.pi) * dx

    @staticmethod
    def get_interface_width(u, L):
        # TODO: How does this generalize to different phase transitions?
        rows1 = np.where(u[L.prob.init[0][0] // 2, : L.prob.init[0][0] // 2] > -0.99)
        rows2 = np.where(u[L.prob.init[0][0] // 2, : L.prob.init[0][0] // 2] < 0.99)

        return (rows2[0][-1] - rows1[0][0]) * L.prob.dx / L.prob.eps

    def pre_run(self, step, level_number):
        """
        Record radius of the blob, exact radius and interface width.

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        super().pre_run(step, level_number)
        L = step.levels[0]

        radius = self.get_radius(L.u[0], L.prob.dx)
        self.init_radius = L.prob.radius

        if L.time == 0.0:
            self.add_to_stats(
                process=step.status.slot,
                time=L.time,
                level=-1,
                iter=step.status.iter,
                sweep=L.status.sweep,
                type='computed_radius',
                value=radius,
            )
            self.add_to_stats(
                process=step.status.slot,
                time=L.time,
                level=-1,
                iter=step.status.iter,
                sweep=L.status.sweep,
                type='exact_radius',
                value=self.init_radius,
            )

    def post_step(self, step, level_number):
        """
        Record radius of the blob, exact radius and interface width.

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        super().post_step(step, level_number)

        # some abbreviations
        L = step.levels[0]

        radius = self.get_radius(L.uend, L.prob.dx)

        exact_radius = self.get_exact_radius(L.time + L.dt)

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=-1,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='computed_radius',
            value=radius,
        )
        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=-1,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='exact_radius',
            value=exact_radius,
        )


def run_AC(
    custom_description=None,
    num_procs=1,
    Tend=1e-2,
    hook_class=LogData,
    fault_stuff=None,
    custom_controller_params=None,
    imex=False,
    u0=None,
    t0=None,
    use_MPI=False,
    live_plot=False,
    FFT=True,
    time_forcing=True,
    **kwargs,
):
    """
    Args:
        custom_description (dict): Overwrite presets
        num_procs (int): Number of steps for MSSDC
        Tend (float): Time to integrate to
        hook_class (pySDC.Hook): A hook to store data
        fault_stuff (dict): A dictionary with information on how to add faults
        custom_controller_params (dict): Overwrite presets
        imex (bool): Solve the problem IMEX or fully implicit
        u0 (dtype_u): Initial value
        t0 (float): Starting time
        use_MPI (bool): Whether or not to use MPI

    Returns:
        dict: The stats object
        controller: The controller
        bool: If the code crashed
    """
    if custom_description is not None:
        problem_params = custom_description.get('problem_params', {})
        if 'imex' in problem_params.keys():
            imex = problem_params['imex']
            problem_params.pop('imex', None)
        if 'FFT' in problem_params.keys():
            FFT = problem_params['FFT']
            problem_params.pop('FFT', None)

    # import problem and sweeper class
    if time_forcing:
        problem_class = allencahn_imex_timeforcing_adaptivity
        from pySDC.projects.Resilience.sweepers import imex_1st_order_efficient as sweeper_class
    elif FFT:
        from pySDC.implementations.problem_classes.AllenCahn_2D_FFT import allencahn2d_imex as problem_class
        from pySDC.projects.Resilience.sweepers import imex_1st_order_efficient as sweeper_class
    elif imex:
        from pySDC.implementations.problem_classes.AllenCahn_2D_FD import allencahn_semiimplicit as problem_class
        from pySDC.projects.Resilience.sweepers import imex_1st_order_efficient as sweeper_class
    else:
        from pySDC.implementations.problem_classes.AllenCahn_2D_FD import allencahn_fullyimplicit as problem_class
        from pySDC.projects.Resilience.sweepers import generic_implicit_efficient as sweeper_class

    level_params = {}
    level_params['dt'] = 1e-4
    level_params['restol'] = 1e-8

    sweeper_params = {}
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 3
    sweeper_params['QI'] = 'LU'
    sweeper_params['QE'] = 'PIC'

    # problem params
    fd_params = {
        'newton_tol': 1e-9,
        'order': 2,
    }
    problem_params = {
        'nvars': (128, 128),
        'init_type': 'circle',
    }
    if not FFT:
        problem_params = {**problem_params, **fd_params}

    step_params = {}
    step_params['maxiter'] = 5

    controller_params = {}
    controller_params['logger_level'] = 30
    controller_params['hook_class'] = (
        hook_collection + (hook_class if type(hook_class) == list else [hook_class]) + ([LivePlot] if live_plot else [])
    )
    controller_params['mssdc_jac'] = False

    if custom_controller_params is not None:
        controller_params = {**controller_params, **custom_controller_params}

    description = {}
    description['problem_class'] = problem_class
    description['problem_params'] = problem_params
    description['sweeper_class'] = sweeper_class
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    if custom_description is not None:
        description = merge_descriptions(description, custom_description)

    t0 = 0.0 if t0 is None else t0

    controller_args = {
        'controller_params': controller_params,
        'description': description,
    }
    if use_MPI:
        from mpi4py import MPI
        from pySDC.implementations.controller_classes.controller_MPI import controller_MPI

        comm = kwargs.get('comm', MPI.COMM_WORLD)
        controller = controller_MPI(**controller_args, comm=comm)
        P = controller.S.levels[0].prob
    else:
        controller = controller_nonMPI(**controller_args, num_procs=num_procs)
        P = controller.MS[0].levels[0].prob

    uinit = P.u_exact(t0) if u0 is None else u0

    if fault_stuff is not None:
        from pySDC.projects.Resilience.fault_injection import prepare_controller_for_faults

        prepare_controller_for_faults(controller, fault_stuff)

    crash = False
    try:
        uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)
    except ConvergenceError as e:
        print(f'Warning: Premature termination!: {e}')
        stats = controller.return_stats()
        crash = True
    return stats, controller, crash


def plot_solution(stats):  # pragma: no cover
    import matplotlib.pyplot as plt
    from pySDC.helpers.stats_helper import get_sorted

    fig, ax = plt.subplots(1, 1)

    u = get_sorted(stats, type='u', recomputed=False)
    for me in u:  # pun intended
        ax.imshow(me[1], vmin=-1, vmax=1)
        ax.set_title(f't={me[0]:.2e}')
        plt.pause(1e-1)

    plt.show()


class LivePlot(Hooks):  # pragma: no cover
    def __init__(self):
        super().__init__()
        self.fig, self.axs = plt.subplots(1, 3, figsize=(12, 4))
        self.radius = []
        self.exact_radius = []
        self.t = []
        self.dt = []

    def post_step(self, step, level_number):
        super().post_step(step, level_number)
        L = step.levels[level_number]
        self.t += [step.time + step.dt]

        # plot solution
        self.axs[0].cla()
        if len(L.uend.shape) > 1:
            self.axs[0].imshow(L.uend, vmin=0.0, vmax=1.0)

            # plot radius
            self.axs[1].cla()
            radius, _ = LogRadius.compute_radius(step.levels[level_number])
            exact_radius = LogRadius.exact_radius(step.levels[level_number])

            self.radius += [radius]
            self.exact_radius += [exact_radius]
            self.axs[1].plot(self.t, self.exact_radius, label='exact')
            self.axs[1].plot(self.t, self.radius, label='numerical')
            self.axs[1].set_ylim([0, 0.26])
            self.axs[1].set_xlim([0, 0.03])
            self.axs[1].legend(frameon=False)
            self.axs[1].set_title(r'Radius')
        else:
            self.axs[0].plot(L.prob.xvalues, L.prob.u_exact(t=L.time + L.dt), label='exact')
            self.axs[0].plot(L.prob.xvalues, L.uend, label='numerical')
        self.axs[0].set_title(f't = {step.time + step.dt:.2e}')

        # plot step size
        self.axs[2].cla()
        self.dt += [step.dt]
        self.axs[2].plot(self.t, self.dt)
        self.axs[2].set_yscale('log')
        self.axs[2].axhline(step.levels[level_number].prob.eps ** 2, label=r'$\epsilon^2$', color='black', ls='--')
        self.axs[2].legend(frameon=False)
        self.axs[2].set_xlim([0, 0.03])
        self.axs[2].set_title(r'$\Delta t$')

        if step.status.restart:
            for me in [self.radius, self.exact_radius, self.t, self.dt]:
                try:
                    me.pop(-1)
                except (TypeError, IndexError):
                    pass

        plt.pause(1e-9)


class LogRadius(Hooks):
    @staticmethod
    def compute_radius(L):
        c = np.count_nonzero(L.u[0] > 0.0)
        radius = np.sqrt(c / np.pi) * L.prob.dx

        rows, cols = np.where(L.u[0] > 0.0)

        rows1 = np.where(L.u[0][int((L.prob.init[0][0]) / 2), : int((L.prob.init[0][0]) / 2)] > -0.99)
        rows2 = np.where(L.u[0][int((L.prob.init[0][0]) / 2), : int((L.prob.init[0][0]) / 2)] < 0.99)
        interface_width = (rows2[0][-1] - rows1[0][0]) * L.prob.dx / L.prob.eps

        return radius, interface_width

    @staticmethod
    def exact_radius(L):
        init_radius = L.prob.radius
        return np.sqrt(max(init_radius**2 - 2.0 * (L.time + L.dt), 0))

    def pre_run(self, step, level_number):
        """
        Overwrite standard pre run hook

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        super().pre_run(step, level_number)
        L = step.levels[0]

        radius, interface_width = self.compute_radius(L)
        exact_radius = self.exact_radius(L)

        if L.time == 0.0:
            self.add_to_stats(
                process=step.status.slot,
                time=L.time,
                level=-1,
                iter=step.status.iter,
                sweep=L.status.sweep,
                type='computed_radius',
                value=radius,
            )
            self.add_to_stats(
                process=step.status.slot,
                time=L.time,
                level=-1,
                iter=step.status.iter,
                sweep=L.status.sweep,
                type='exact_radius',
                value=exact_radius,
            )
            self.add_to_stats(
                process=step.status.slot,
                time=L.time,
                level=-1,
                iter=step.status.iter,
                sweep=L.status.sweep,
                type='interface_width',
                value=interface_width,
            )

    def post_run(self, step, level_number):
        """
        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        super().post_run(step, level_number)

        L = step.levels[0]

        exact_radius = self.exact_radius(L)
        radius, interface_width = self.compute_radius(L)

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=-1,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='computed_radius',
            value=radius,
        )
        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=-1,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='exact_radius',
            value=exact_radius,
        )
        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=-1,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='interface_width',
            value=interface_width,
        )
        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=level_number,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='e_global_post_run',
            value=abs(radius - exact_radius),
        )
        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=level_number,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='e_global_rel_post_run',
            value=abs(radius - exact_radius) / abs(exact_radius),
        )


if __name__ == '__main__':

    stats, _, _ = run_AC()
    plot_solution(stats)
