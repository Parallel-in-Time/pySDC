from mpi4py import MPI
import numpy as np

from pySDC.helpers.stats_helper import get_sorted

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.problem_classes.NonlinearSchroedinger_MPIFFT import (
    nonlinearschroedinger_imex,
    nonlinearschroedinger_fully_implicit,
)
from pySDC.implementations.transfer_classes.TransferMesh_MPIFFT import fft_to_fft
from pySDC.projects.Resilience.hook import LogData, hook_collection
from pySDC.projects.Resilience.strategies import merge_descriptions

from pySDC.core.Hooks import hooks

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


class live_plotting_with_error(hooks):  # pragma: no cover
    def __init__(self):
        super().__init__()
        self.fig, self.axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(12, 7))

        divider = make_axes_locatable(self.axs[1])
        self.cax_right = divider.append_axes('right', size='5%', pad=0.05)
        divider = make_axes_locatable(self.axs[0])
        self.cax_left = divider.append_axes('right', size='5%', pad=0.05)

    def post_step(self, step, level_number):
        lvl = step.levels[level_number]
        lvl.sweep.compute_end_point()

        self.axs[0].cla()
        im1 = self.axs[0].imshow(np.abs(lvl.uend), vmin=0, vmax=2.0)
        self.fig.colorbar(im1, cax=self.cax_left)

        self.axs[1].cla()
        im = self.axs[1].imshow(np.abs(lvl.prob.u_exact(lvl.time + lvl.dt) - lvl.uend))
        self.fig.colorbar(im, cax=self.cax_right)

        self.fig.suptitle(f't={lvl.time:.2f}')
        self.axs[0].set_title('solution')
        self.axs[1].set_title('error')
        plt.pause(1e-9)


class live_plotting(hooks):  # pragma: no cover
    def __init__(self):
        super().__init__()
        self.fig, self.ax = plt.subplots()
        divider = make_axes_locatable(self.ax)
        self.cax = divider.append_axes('right', size='5%', pad=0.05)

    def post_step(self, step, level_number):
        lvl = step.levels[level_number]
        lvl.sweep.compute_end_point()

        self.ax.cla()
        im = self.ax.imshow(np.abs(lvl.uend), vmin=0.2, vmax=1.8)
        self.ax.set_title(f't={lvl.time + lvl.dt:.2f}')
        self.fig.colorbar(im, cax=self.cax)
        plt.pause(1e-9)


def run_Schroedinger(
    custom_description=None,
    num_procs=1,
    Tend=1.0,
    hook_class=LogData,
    fault_stuff=None,
    custom_controller_params=None,
    use_MPI=False,
    space_comm=None,
    imex=True,
    **kwargs,
):
    """
    Run a Schroedinger problem with default parameters.

    Args:
        custom_description (dict): Overwrite presets
        num_procs (int): Number of steps for MSSDC
        Tend (float): Time to integrate to
        hook_class (pySDC.Hook): A hook to store data
        fault_stuff (dict): A dictionary with information on how to add faults
        custom_controller_params (dict): Overwrite presets
        use_MPI (bool): Whether or not to use MPI
        space_comm (mpi4py.Intracomm): Space communicator
        imex (bool): Whether to use IMEX implementation or the fully implicit one

    Returns:
        dict: The stats object
        controller: The controller
        Tend: The time that was supposed to be integrated to
    """
    if custom_description is not None:
        problem_params = custom_description.get('problem_params', {})
        if 'imex' in problem_params.keys():
            imex = problem_params['imex']
            problem_params.pop('imex', None)

    from mpi4py import MPI

    space_comm = MPI.COMM_SELF if space_comm is None else space_comm
    rank = space_comm.Get_rank()

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-8
    level_params['dt'] = 1e-01 / 2
    level_params['nsweeps'] = 1

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 3
    sweeper_params['QI'] = 'IE'
    sweeper_params['initial_guess'] = 'spread'

    # initialize problem parameters
    problem_params = dict()
    problem_params['nvars'] = (128, 128)
    problem_params['spectral'] = False
    problem_params['c'] = 1.0
    problem_params['comm'] = space_comm
    if not imex:
        problem_params['liniter'] = 99
        problem_params['lintol'] = 1e-8

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 50

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 15 if rank == 0 else 99
    controller_params['hook_class'] = hook_collection + (hook_class if type(hook_class) == list else [hook_class])
    controller_params['mssdc_jac'] = False

    # fill description dictionary for easy step instantiation
    if custom_controller_params is not None:
        controller_params = {**controller_params, **custom_controller_params}

    description = dict()
    description['problem_params'] = problem_params
    description['problem_class'] = nonlinearschroedinger_imex if imex else nonlinearschroedinger_fully_implicit
    description['sweeper_class'] = imex_1st_order if imex else generic_implicit
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    if custom_description is not None:
        description = merge_descriptions(description, custom_description)

    # set time parameters
    t0 = 0.0

    # instantiate controller
    controller_args = {
        'controller_params': controller_params,
        'description': description,
    }
    if use_MPI:
        from pySDC.implementations.controller_classes.controller_MPI import controller_MPI

        comm = kwargs.get('comm', MPI.COMM_WORLD)
        controller = controller_MPI(**controller_args, comm=comm)
        P = controller.S.levels[0].prob
    else:
        controller = controller_nonMPI(**controller_args, num_procs=num_procs)
        P = controller.MS[0].levels[0].prob

    uinit = P.u_exact(t0)

    # insert faults
    if fault_stuff is not None:
        from pySDC.projects.Resilience.fault_injection import prepare_controller_for_faults

        nvars = [me / 2 for me in problem_params['nvars']]
        nvars[0] += 1

        rnd_args = {'problem_pos': nvars}
        prepare_controller_for_faults(controller, fault_stuff, rnd_args)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    return stats, controller, Tend


def main():
    stats, _, _ = run_Schroedinger(space_comm=MPI.COMM_WORLD, hook_class=live_plotting, imex=False)
    plt.show()


if __name__ == "__main__":
    main()
