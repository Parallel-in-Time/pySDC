from pathlib import Path
import time
import numpy as np

from pySDC.core.Errors import ParameterError

from pySDC.projects.ExplicitStabilized.problem_classes.ParabolicSystem_FEniCSx import parabolic_system, parabolic_system_multirate, parabolic_system_imex
import pySDC.projects.ExplicitStabilized.problem_classes.parabolic_system_helpers.problems as problems
from pySDC.projects.ExplicitStabilized.transfer_classes.TransferFenicsxMesh import mesh_to_mesh_fenicsx
from pySDC.projects.ExplicitStabilized.hooks.HookClass_pde import pde_hook

from pySDC.helpers.stats_helper import get_sorted
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.sweeper_classes.explicit import explicit

from pySDC.projects.ExplicitStabilized.explicit_stabilized_classes.es_methods import RKW1, RKC1, RKU1, HSRKU1, mRKC1

def main():

    # define integration methods
    integrators = ['IMEX']
    # integrators = ['MS_EE']
    # integrators = ['MS_ES']
    # integrators = ['MS_mES']
    # # integrators = ['ES']
    # integrators = ['mES']
    
    num_procs = 1

    ref = 0

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 5e-6
    level_params['dt'] = 0.1
    level_params['nsweeps'] = [1]
    level_params['residual_type'] = 'full_rel'

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['initial_guess'] = 'spread'
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = [3]
    # specific for explicit stabilized methods
    # sweeper_params['es_class'] = RKW1
    # sweeper_params['es_class_outer'] = RKW1
    # sweeper_params['es_class_inner'] = RKW1
    # sweeper_params['res_comp'] = 'f_eta'
    # sweeper_params['damping'] = 0.05
    # sweeper_params['safe_add'] = 0
    # sweeper_params['order'] = [3]
    # sweeper_params['nodes_choice'] = 'all' # closest_radau, last, all
    # sweeper_params['rho_freq'] = 10

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 1

    # initialize problem parameters
    problem_params = dict()
    problem_params['family'] = 'CG'
    problem_params['order'] = 1
    problem_params['mass_lumping'] = True # has effect for family=CG and order=1
    problem_params['n_elems'] = [40]
    problem_params['refinements'] = [0]
    problem_params['dim'] = 2
    problem_params['f_interp'] = True
    # problem_params['solver_ksp'] = 'cg' # comment these two lines to use default solver (cholesky for dim<=2 and cg+hypre for dim=3)
    # problem_params['solver_pc'] = 'hypre' # work in parallel with cg: hypre, hmg, gamg, jacobi,
    # problem_params['solver_ksp'] = 'preonly' # comment these two lines to use default solver (cholesky for dim<=2 and cg+hypre for dim=3)
    # problem_params['solver_pc'] = 'cholesky'
    problem_params['enable_output'] = False
    problem_params['output_folder'] = './data/results/'
    problem_params['output_file_name'] = 'bruss'
    problem_params['exact_solution_class'] = problems.brusselator


    # base transfer parameters
    base_transfer_params = dict()
    base_transfer_params['finter'] = False

    # initialize controller parameters
    controller_params = dict()
    controller_params['predict_type'] = 'pfasst_burnin'
    controller_params['log_to_file'] = False
    controller_params['fname'] = 'data/ExplicitStabilized_HeatEquation'
    controller_params['logger_level'] = 20
    controller_params['dump_setup'] = False
    controller_params['hook_class'] = pde_hook

    Path("data").mkdir(parents=True, exist_ok=True)

    for integrator in integrators:
        
        description = dict()          
        if integrator == 'IMEX':
            parabolic_system_type = parabolic_system_imex
            description['sweeper_class'] = imex_1st_order        
        # elif integrator == 'ES':                     
        #     ParabolicProblemType = parabolic_system
        #     description['sweeper_class'] = explicit_stabilized
        # elif integrator == 'mES':                     
        #     ParabolicProblemType = parabolic_system_multirate 
        #     description['sweeper_class'] = multirate_explicit_stabilized
        # elif integrator == 'MS_ES':
        #     ParabolicProblemType = parabolic_system
        #     description['sweeper_class'] = ms_explicit_stabilized
        # elif integrator == 'MS_mES':
        #     ParabolicProblemType = parabolic_system_multirate
        #     description['sweeper_class'] = ms_multirate_explicit_stabilized
        # elif integrator == 'EE':
        #     ParabolicProblemType = parabolic_system
        #     description['sweeper_class'] = explicit
        # elif integrator == 'MS_EE':
        #     ParabolicProblemType = parabolic_system
        #     description['sweeper_class'] = ms_explicit      
        else:
            raise ParameterError('Unknown integrator.')
        
        description['problem_class'] = parabolic_system_type
        description['problem_params'] = problem_params
        description['sweeper_params'] = sweeper_params
        description['level_params'] = level_params
        description['step_params'] = step_params
        description['base_transfer_params'] = base_transfer_params
        description['space_transfer_class'] = mesh_to_mesh_fenicsx

        # instantiate the controller
        controller = controller_nonMPI(num_procs=num_procs, controller_params=controller_params, description=description)

        # get initial values on finest level
        P = controller.MS[0].levels[0].prob
        # set time parameters
        t0 = P.t0
        Tend = P.Tend
        uinit = P.initial_value()

        prob_size = P.get_size()
        if P.domain.comm.rank == 0:
            print(f'Problem size: {prob_size}')

        data = (uinit.n_loc_dofs+uinit.n_ghost_dofs,uinit.n_loc_dofs,uinit.n_ghost_dofs,uinit.n_ghost_dofs/(uinit.n_loc_dofs+uinit.n_loc_dofs))
        data = P.domain.comm.gather(data, root=0)
        if P.domain.comm.rank == 0:
            for i,d in enumerate(data):
                print(f'Processor {i}: tot_dofs = {d[0]:.2e}, n_loc_dofs = {d[1]:.2e}, n_ghost_dofs = {d[2]:.2e}, %ghost = {100*d[3]:.2f}')        

        # call main function to get things done...
        uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

        # compute exact solution and compare
        if P.know_exact:            
            P.compute_errors(uend,Tend)            

        # filter statistics by type (number of iterations)
        iter_counts = get_sorted(stats, type='niter', sortby='time')

        niters = np.array([item[1] for item in iter_counts])
        out = 'Mean number of iterations: %4.2f' % np.mean(niters)
        print(out)
        out = 'Std and var for number of iterations: %4.2f -- %4.2f' % (float(np.std(niters)), float(np.var(niters)))
        print(out)

        timing = get_sorted(stats, type='timing_run', sortby='time')
        out = f'Time to solution: {timing[0][1]:6.4f} sec.'
        print(out)



if __name__ == "__main__":
    main()
