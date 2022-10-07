import numpy as np
import dill

from pySDC.helpers.stats_helper import get_sorted
from pySDC.implementations.collocations import Collocation
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
#from pySDC.projects.PinTSimE.switch_controller_nonMPI import switch_controller_nonMPI
from pySDC.implementations.problem_classes.Battery import battery
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.projects.PinTSimE.piline_model import setup_mpl
import pySDC.helpers.plot_helper as plt_helper

from pySDC.core.Hooks import hooks


class log_data(hooks):

    def post_iteration(self, step, level_number):

        super(log_data, self).post_iteration(step, level_number)

        # some abbreviations
        L = step.levels[level_number]

        L.sweep.compute_end_point()

        self.add_to_stats(process=step.status.slot, time=L.time, level=L.level_index, iter=0,
                          sweep=L.status.sweep, type='current L', value=L.uend[0])
        self.add_to_stats(process=step.status.slot, time=L.time, level=L.level_index, iter=0,
                          sweep=L.status.sweep, type='voltage C', value=L.uend[1])
        #self.add_to_stats(process=step.status.slot, time=L.time+L.dt, level=L.level_index,
        #                  iter=step.status.iter, sweep=L.status.sweep, type='residuals',
        #                  value=L.status.residual)
        self.add_to_stats(process=step.status.slot, time=L.time, level=L.level_index, iter=-1,
                          sweep=L.status.sweep, type='residual_post_iteration', value=L.status.residual)


def main():
    """
        Test program to present residuals for the battery drain model for each time and each iteration
    """

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1E-13
    level_params['dt'] = 1E-3

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = Collocation
    sweeper_params['node_type'] = 'LEGENDRE'
    sweeper_params['quad_type'] = 'LOBATTO'
    sweeper_params['num_nodes'] = 5
    sweeper_params['QI'] = 'LU'  # For the IMEX sweeper, the LU-trick can be activated for the implicit part
    sweeper_params['initial_guess'] = 'zero'

    # initialize problem parameters
    problem_params = dict()
    problem_params['Vs'] = 5.0
    problem_params['Rs'] = 0.5
    problem_params['C'] = 1
    problem_params['R'] = 1
    problem_params['L'] = 1
    problem_params['alpha'] = 10
    problem_params['V_ref'] = 1

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 20

    # initialize controller parameters
    controller_params = dict()
    controller_params['use_switch_estimator'] = False
    controller_params['logger_level'] = 20
    controller_params['hook_class'] = log_data

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = battery  # pass problem class
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = imex_1st_order  # pass sweeper
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params

    assert problem_params['alpha'] > problem_params['V_ref'], 'Please set "alpha" greater than "V_ref"'
    assert problem_params['V_ref'] > 0, 'Please set "V_ref" greater than 0'

    assert 'errtol' not in description['step_params'].keys(), 'No exact solution known to compute error'
    assert 'alpha' in description['problem_params'].keys(), 'Please supply "alpha" in the problem parameters'
    assert 'V_ref' in description['problem_params'].keys(), 'Please supply "V_ref" in the problem parameters'

    # set time parameters
    t0 = 0.0
    Tend = 4.0

    # instantiate controller
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    # fname = 'data/battery_residuals.dat'
    fname = 'battery_residuals.dat'
    f = open(fname, 'wb')
    dill.dump(stats, f)
    f.close()

    plot_residuals(problem_params)
    
def plot_residuals(problem_params, cwd='./'):
    """
        Routine to plot residuals for battery drain model in area of the switch
    """

    f = open(cwd + 'battery_residuals.dat', 'rb')
    stats = dill.load(f)
    f.close()
    
    # filter residuals by time
    res = get_sorted(stats, type='residual_post_iteration', sortby='time')

    # filter iteration statistics by time
    iter_counts = get_sorted(stats, type='niter', sortby='time')
    times = [item[0] for item in iter_counts]
    
    # dictionary with times as key and (niter, residual) as value
    res_dict = dict()
    for iter_item in iter_counts:
        for res_item in res:
            res_dict.setdefault(res_item[0], [])
            if res_item[0] == iter_item[0]:
                res_dict[res_item[0]].append(np.array([iter_item[1], res_item[1]]))
    
    vC = get_sorted(stats, type='voltage C', sortby='time')
    vC_val = [v[1] for v in vC]
            
    # looking for switch
    restored_keys = []
    i = 0
    for iter_item in iter_counts:
        print(i)
        if problem_params['V_ref'] < vC_val[i] < problem_params['V_ref'] + 0.5:
            print(len(vC_val), i)
            restored_keys.append(iter_item[0])
         
        elif np.isclose(vC_val[i], problem_params['V_ref'], atol=1e-6) == True:
            print("second if:", i)
            restored_keys.append(iter_item[0])
            i_switch = i
            key_switch = iter_item[0]
            break
         
        i += 1
    
    #k = 0 
    #resasarray = np.zeros((len(restored_keys), 2))    
    #for key_item in restored_keys:
    #    resasarray[k, :] = res_dict[key_item]
    #    k += 1
    
    for key_item in restored_keys:
        element = res_dict[key_item]
        print(element, type(element))
        
        #for item in element:
            
         
    setup_mpl()
    fig, ax = plt_helper.plt.subplots(1, 1, figsize=(4.5, 3))
    ax.plot(resasarray[:, 1], resasarray[:, 0])
    ax.set_yscale('log', base=10)
    fig.savefig('battery_residuals.png', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    main()
