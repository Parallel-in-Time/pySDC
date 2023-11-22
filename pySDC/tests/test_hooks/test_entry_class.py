import pytest

from pySDC.core.Hooks import hooks, meta_data, namedtuple

# In case the starship needs manual override of the reentry sequence, we set a code for unlocking manual controls.
# Because humans may go crazy, or worse, deflect to the enemy when they enter space, we can't tell them the code, or how
# the flight is progressing for that matter. Hence, the weather data on convection in the atmosphere is locked with the
# same code.
convection_meta_data = {
    **meta_data,
    'unlock_manual_controls': None,
}
Entry = namedtuple('Entry', convection_meta_data.keys())


class convection_hook(hooks):
    meta_data = convection_meta_data
    entry = Entry
    starship = 'vostok'

    def post_step(self, step, level_number):
        """
        Log the amount of convection, but lock it with a special code that we will definitely not tell Yuri...

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number

        Returns:
            None
        """
        L = step.levels[level_number]
        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='convection',
            value=L.uend[0],
            unlock_manual_controls=125 if self.starship == 'vostok' else None,
        )


def win_space_race(useMPI):
    from pySDC.implementations.hooks.log_work import LogWork
    from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
    from pySDC.implementations.problem_classes.Lorenz import LorenzAttractor
    from pySDC.helpers.stats_helper import get_sorted

    num_steps = 4

    # initialize level parameters
    level_params = {}
    level_params['dt'] = 1e-2
    level_params['restol'] = -1

    # initialize sweeper parameters
    sweeper_params = {}
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 1
    sweeper_params['QI'] = 'IE'

    problem_params = {}

    # initialize step parameters
    step_params = {}
    step_params['maxiter'] = 1

    # initialize controller parameters
    controller_params = {}
    controller_params['logger_level'] = 30
    controller_params['hook_class'] = convection_hook
    controller_params['mssdc_jac'] = False

    # fill description dictionary for easy step instantiation
    description = {}
    description['problem_class'] = LorenzAttractor
    description['problem_params'] = problem_params
    description['sweeper_class'] = generic_implicit
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    # set time parameters
    t0 = 0.0

    # instantiate controller
    if useMPI:
        from mpi4py import MPI
        from pySDC.implementations.controller_classes.controller_MPI import controller_MPI

        comm = MPI.COMM_WORLD

        controller = controller_MPI(controller_params=controller_params, description=description, comm=comm)
        P = controller.S.levels[0].prob
    else:
        from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

        comm = None
        controller = controller_nonMPI(
            num_procs=num_steps, controller_params=controller_params, description=description
        )
        P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    uend, stats = controller.run(u0=uinit, t0=t0, Tend=num_steps * level_params['dt'])

    from pySDC.helpers.stats_helper import get_list_of_types

    expected = -1
    for code in [None, 0, 125]:
        for type in ['residual_post_step', 'convection']:
            res = get_sorted(stats, type=type, unlock_manual_controls=code, comm=comm)
            if type == 'residual_post_step':
                expected = num_steps if code is None else 0
            if type == 'convection':
                expected = num_steps if code in [None, 125] else 0  # hmmm... security doesn't seem too good...

            assert (
                len(res) == expected
            ), f'Unexpected number of entries in stats for type {type} and code {code}! Got {len(res)}, but expected {expected}.'

    return None


@pytest.mark.base
def test_entry_class():
    win_space_race(False)


@pytest.mark.mpi4py
def test_entry_class_MPI():
    import os
    import subprocess

    num_procs = 4

    # Set python path once
    my_env = os.environ.copy()
    my_env['PYTHONPATH'] = '../../..:.'
    my_env['COVERAGE_PROCESS_START'] = 'pyproject.toml'

    cmd = f"mpirun -np {num_procs} python {__file__}".split()

    p = subprocess.Popen(cmd, env=my_env, cwd=".")

    p.wait()
    assert p.returncode == 0, 'ERROR: did not get return code 0, got %s with %2i processes' % (
        p.returncode,
        num_procs,
    )


if __name__ == "__main__":
    import sys

    win_space_race(True)
