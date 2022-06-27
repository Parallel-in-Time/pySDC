import numpy as np
import dill

from pySDC.helpers.stats_helper import get_sorted
from pySDC.implementations.collocations import Collocation
from pySDC.projects.PinTSimE.switch_controller_nonMPI import switch_controller_nonMPI
from pySDC.implementations.problem_classes.BuckConverter_OneSwitch import buck_converter_oneswitch
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
# from pySDC.implementations.sweeper_classes.generic_LU import generic_LU
from pySDC.projects.PinTSimE.piline_model import log_data, setup_mpl
import pySDC.helpers.plot_helper as plt_helper


def run_cases(t0, dt_first, dt_rest, Tend):
    """
        Function computes a solution for buck converter with one switch at time t_switch
    """

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1E-13
    level_params['dt'] = dt_first

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = Collocation
    sweeper_params['node_type'] = 'LEGENDRE'
    sweeper_params['quad_type'] = 'LOBATTO'
    sweeper_params['num_nodes'] = 5
    sweeper_params['QI'] = 'LU'  # For the IMEX sweeper, the LU-trick can be activated for the implicit part

    # initialize problem parameters
    problem_params = dict()
    problem_params['t_switch'] = 0.01
    problem_params['dt_rest'] = dt_rest
    problem_params['Vs'] = 10.0
    problem_params['Rs'] = 0.5
    problem_params['C1'] = 1e-3
    problem_params['Rp'] = 0.01
    problem_params['L1'] = 1e-3
    problem_params['C2'] = 1e-3
    problem_params['Rl'] = 10

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 20

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 20
    controller_params['hook_class'] = log_data

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = buck_converter_oneswitch   # pass problem class
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = imex_1st_order   # pass sweeper
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params      # pass level parameters
    description['step_params'] = step_params
    
    assert 't_switch' in description['problem_params'].keys(), 'Please supply "t_switch" in the problem parameters'
    
    # set time parameters
    t0 = t0
    Tend = Tend

    assert t0 < problem_params['t_switch'] < Tend, 'Please set "t_switch" greater than t0 and less than Tend'

    # instantiate controller
    controller = switch_controller_nonMPI(num_procs=1, controller_params=controller_params,
                                          description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    return stats
    
def plots(dat_name, cwd='./'):
    f = open(cwd + '{}.dat'.format(dat_name), 'rb')
    stats = dill.load(f)
    f.close()
    
    # convert filtered statistics to list of iterations count, sorted by process
    v1 = get_sorted(stats, type='v1', sortby='time')
    v2 = get_sorted(stats, type='v2', sortby='time')
    p3 = get_sorted(stats, type='p3', sortby='time')

    times = [v[0] for v in v1]

    setup_mpl()
    fig, ax = plt_helper.plt.subplots(1, 1, figsize=(4.5, 3))
    ax.plot(times, [v[1] for v in v1], linewidth=1, label='$v_{C_1}$')
    ax.plot(times, [v[1] for v in v2], linewidth=1, label='$v_{C_2}$')
    ax.plot(times, [v[1] for v in p3], linewidth=1, label='$i_{L_\pi}$')
    ax.legend(frameon=False, fontsize=12, loc='upper right')
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Energy')
    
    fig.savefig('{}.png'.format(dat_name), dpi=300, bbox_inches='tight')

def plot_error(dat_name_ref, all_cases, cwd='./'):

    f_ref = open(cwd + '{}.dat'.format(dat_name_ref), 'rb')
    stats_ref = dill.load(f_ref)
    f_ref.close()

    # convert filtered statistics to list of iterations count, sorted by process
    v1_ref = get_sorted(stats_ref, type='v1', sortby='time')
    v2_ref = get_sorted(stats_ref, type='v2', sortby='time')
    p3_ref = get_sorted(stats_ref, type='p3', sortby='time')
    
    t_ref = [v[0] for v in v1_ref]
    
    # define a dictionary for reference solution for better compare of values
    ref_dict = reference_as_dictionary(t_ref, v1_ref, v2_ref, p3_ref)
    ref_keys = ref_dict.keys()
    
    for case in all_cases:
        f_case = open(cwd + '{}.dat'.format(case), 'rb')
        stats_case = dill.load(f_case)
        f_case.close()
    
        v1_case = get_sorted(stats_case, type='v1', sortby='time')
        v2_case = get_sorted(stats_case, type='v2', sortby='time')
        p3_case = get_sorted(stats_case, type='p3', sortby='time')
    
        t_case = [v[0] for v in v1_case]
        v1 = [v[1] for v in v1_case]
        v2 = [v[1] for v in v2_case]
        p3 = [v[1] for v in p3_case]

        restored_keys = []
        for item in t_case:
            for ref_item in ref_keys:
                if round(item, 15) == round(ref_item, 15):
                    restored_keys.append(ref_item)
                    
        refasarray = np.zeros((3, len(keys1)))
        for i in range(len(restored_keys)):
            refasarray[:, i] = np.asarray(ref_dict[keys1[i]])
            
        diff_v1 = np.zeros(len(restored_keys))
        diff_v2 = np.zeros(len(restored_keys))
        diff_p1 = np.zeros(len(restored_keys))

        for i in range(len(restored_keys)):
            diff_v1 = v1[i] - refasarray[0, i]
            diff_v2 = v2[i] - refasarray[1, i]
            diff_p3 = p3[i] - refasarray[2, i]
            
        setup_mpl()
        fig, ax = plt_helper.plt.subplots(1, 1, figsize=(4.5, 3))
        ax.set_title('Error for case {}'.format(case))
        ax.plot(t_case, diff_v1, linewidth=1, label='$v_{C_1}$')
        ax.plot(t_case, diff_v2, linewidth=1, label='$v_{C_2}$')
        ax.plot(t_case, diff_p3, linewidth=1, label='$i_{L_\pi}$')
        ax.axvline(x=problem_params['t_switch'], label='Switch')
        ax.legend(frameon=False, fontsize=12, loc='upper right')
    
        ax.set_xlabel('Time')
        ax.set_ylabel('Energy')
    
        fig.savefig('Error_{}.png'.format(case), dpi=300, bbox_inches='tight')

def reference_as_dictionary(t_ref, v1_ref, v2_ref, p3_ref):
    solution = dict()
    i = 0
    for times in t_ref:
        solution[times] = [v1_ref[i], v2_ref[i], p3_ref[i]]
        i += 1

    return solution

def main():
    """
        Program to compute the residuals for two different switching cases:

            1. Switch occurs on a time step
            2. Switch occurs on a node
            3. Switch occurs between two different collocation nodes
    """

    dt = [[2e-4, 2e-4], [1e-4, 2e-4], [9e-5, 2e-4], [1e-8, 1e-8]]
    all_cases = ['Buck_switch_on_timestep', 'Buck_switch_on_node', 'Buck_switch_btw_nodes', 'Buck_reference']
    t0 = 0.0
    Tend = 0.02
    
    for dt_list, case in zip(dt, all_cases):
        stats = run_cases(t0, dt_list[0], dt_list[1], Tend)

        # save stats
        fname = '{}.dat'.format(case)
        f = open(fname, 'wb')
        dill.dump(stats, f)
        f.close()
        
    plot_error(all_cases[-1], all_cases[0:-2], cwd='./')
    


if __name__ == "__main__":
    main()
