import numpy as np
import dill

from pySDC.helpers.stats_helper import get_sorted
from pySDC.core import CollBase as Collocation
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.StateSpaceInverter import state_space_inverter
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order

# from pySDC.implementations.sweeper_classes.generic_LU import generic_LU
from pySDC.playgrounds.EnergyGrids.log_data import log_data
import pySDC.helpers.plot_helper as plt_helper
from pySDC.core.Hooks import hooks

class log_data(hooks):
    def post_step(self, step, level_number):

        super(log_data, self).post_step(step, level_number)

        # some abbreviations
        L = step.levels[level_number]

        L.sweep.compute_end_point()

        self.add_to_stats(
            process=step.status.slot,
            time=L.time+L.dt,
            level=L.level_index,
            iter=0,
            sweep=L.status.sweep,
            type='voltage CDC1',
            value=L.uend[0],
        )
        self.add_to_stats(
            process=step.status.slot,
            time=L.time+L.dt,
            level=L.level_index,
            iter=0,
            sweep=L.status.sweep,
            type='voltage CDC2',
            value=L.uend[1],
        )
        self.add_to_stats(
            process=step.status.slot,
            time=L.time+L.dt,
            level=L.level_index,
            iter=0,
            sweep=L.status.sweep,
            type='current L1',
            value=L.uend[2],
        )
        self.add_to_stats(
            process=step.status.slot,
            time=L.time+L.dt,
            level=L.level_index,
            iter=0,
            sweep=L.status.sweep,
            type='current L2',
            value=L.uend[3],
        )
        self.add_to_stats(
            process=step.status.slot,
            time=L.time+L.dt,
            level=L.level_index,
            iter=0,
            sweep=L.status.sweep,
            type='current L3',
            value=L.uend[4],
        )
        self.add_to_stats(
            process=step.status.slot,
            time=L.time+L.dt,
            level=L.level_index,
            iter=0,
            sweep=L.status.sweep,
            type='voltage C1',
            value=L.uend[5],
        )
        self.add_to_stats(
            process=step.status.slot,
            time=L.time+L.dt,
            level=L.level_index,
            iter=0,
            sweep=L.status.sweep,
            type='voltage C2',
            value=L.uend[6],
        )
        self.add_to_stats(
            process=step.status.slot,
            time=L.time+L.dt,
            level=L.level_index,
            iter=0,
            sweep=L.status.sweep,
            type='voltage C3', 
            value=L.uend[7],
        ) 
                          
def main():
    """
    A playground to do PFASST runs for the state space inverter model
    """ 

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-12
    level_params['dt'] = 1e-6  

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'LOBATTO'
    sweeper_params['num_nodes'] = 5
    sweeper_params['QI'] = 'LU'  # For the IMEX sweeper, the LU-trick can be activated for the implicit part

    # initialize problem parameters
    problem_params = dict()
    problem_params['fsw'] = 1e3
    problem_params['V1'] = 100.0
    problem_params['V2'] = 100.0
    problem_params['Rs1'] = 0.01
    problem_params['Rs2'] = 0.01
    problem_params['CDC1'] = 0.001
    problem_params['CDC2'] = 0.001
    problem_params['C1'] = 0.00002
    problem_params['C2'] = 0.00002
    problem_params['C3'] = 0.00002
    problem_params['L1'] = 0.01
    problem_params['L2'] = 0.01
    problem_params['L3'] = 0.01
    problem_params['Rl1'] = 0.01
    problem_params['Rl2'] = 0.01
    problem_params['Rl3'] = 0.01

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 20

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 20
    controller_params['hook_class'] = log_data

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = state_space_inverter  # pass problem class
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = imex_1st_order  # pass sweeper
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params

    # set time parameters
    t0 = 0.0
    Tend = 1e-1

    # instantiate controller
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    fname = 'data/state_space.dat'
    f = open(fname, 'wb')
    dill.dump(stats, f)
    f.close()

def plot_voltages(cwd='./'):
    f = open(cwd + 'state_space.dat', 'rb')
    stats = dill.load(f)
    f.close()

    # convert filtered statistics to list of iterations count, sorted by process
    vCDC1 = get_sorted(stats, type='voltage CDC1', sortby='time')
    vCDC2 = get_sorted(stats, type='voltage CDC2', sortby='time')
    iL1 = get_sorted(stats, type='current L1', sortby='time')
    iL2 = get_sorted(stats, type='current L2', sortby='time')
    iL3 = get_sorted(stats, type='current L3', sortby='time')
    vC1 = get_sorted(stats, type='voltage C1', sortby='time')
    vC2 = get_sorted(stats, type='voltage C2', sortby='time')
    vC3 = get_sorted(stats, type='voltage C3', sortby='time')

    times = [v[0] for v in vCDC1]

    # plt_helper.setup_mpl()
    plt_helper.plt.plot(times, [v[1] for v in vC1], label='voltage C1')
    plt_helper.plt.plot(times, [v[1] for v in vC2], label='voltage C2')
    plt_helper.plt.plot(times, [v[1] for v in vC3], label='voltage C3')
    plt_helper.plt.legend()

    plt_helper.plt.show()


if __name__ == "__main__":
    main()
    plot_voltages()
