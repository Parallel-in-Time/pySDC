import numpy as np

from pySDC.helpers.stats_helper import filter_stats, sort_stats
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.Van_der_Pol_implicit import vanderpol
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.transfer_classes.TransferMesh_NoCoarse import mesh_to_mesh
from pySDC.projects.RDC.equidistant_RDC import Equidistant_RDC


def run_RDC(cwd=''):
    """
    Van der Pol's oscillator with RDC, MLRDC and PFASST

    Args:
        cwd (string): current working directory

    Returns:
        list: list of errors and mean number of iterations (for testing)
    """

    # set time parameters
    t0 = 0.0
    Tend = 10.0

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1E-10
    level_params['dt'] = 0.25

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = Equidistant_RDC
    sweeper_params['num_nodes'] = 20
    sweeper_params['QI'] = 'IE'

    # initialize problem parameters
    problem_params = dict()
    problem_params['newton_tol'] = 1E-12
    problem_params['newton_maxiter'] = 50
    problem_params['mu'] = 10
    problem_params['u0'] = (2.0, 0)

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 50

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 50

    base_transfer_params = dict()
    # base_transfer_params['finter'] = True
    # base_transfer_params['coll_iorder'] = 2
    # base_transfer_params['coll_rorder'] = 2

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = vanderpol
    description['problem_params'] = problem_params
    description['sweeper_class'] = generic_implicit
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params
    description['space_transfer_class'] = mesh_to_mesh
    description['base_transfer_params'] = base_transfer_params

    results = []
    ref_sol = np.load(cwd + 'data/vdp_ref.npy')

    # instantiate the controller
    controller_rdc = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller_rdc.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend_rdc, stats_rdc = controller_rdc.run(u0=uinit, t0=t0, Tend=Tend)

    # filter statistics by type (number of iterations)
    filtered_stats = filter_stats(stats_rdc, type='niter')
    # convert filtered statistics to list of iterations count, sorted by process
    iter_counts = sort_stats(filtered_stats, sortby='time')
    mean_niter = np.mean(np.array([item[1] for item in iter_counts]))

    err = np.linalg.norm(uend_rdc.values - ref_sol, np.inf) / np.linalg.norm(ref_sol, np.inf)
    print('RDC       : Mean number of iterations: %6.3f -- Error: %8.4e' % (mean_niter, err))
    results.append((err, mean_niter))

    sweeper_params['num_nodes'] = [sweeper_params['num_nodes'], 10]
    controller_mlrdc = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    uend_mlrdc, stats_mlrdc = controller_mlrdc.run(u0=uinit, t0=t0, Tend=Tend)

    # filter statistics by type (number of iterations)
    filtered_stats = filter_stats(stats_mlrdc, type='niter')
    # convert filtered statistics to list of iterations count, sorted by process
    iter_counts = sort_stats(filtered_stats, sortby='time')
    mean_niter = np.mean(np.array([item[1] for item in iter_counts]))

    err = np.linalg.norm(uend_mlrdc.values - ref_sol, np.inf) / np.linalg.norm(ref_sol, np.inf)
    print('MLRDC     : Mean number of iterations: %6.3f -- Error: %8.4e' % (mean_niter, err))
    results.append((err, mean_niter))

    controller_pfasst = controller_nonMPI(num_procs=10, controller_params=controller_params, description=description)

    uend_pfasst, stats_pfasst = controller_pfasst.run(u0=uinit, t0=t0, Tend=Tend)

    # filter statistics by type (number of iterations)
    filtered_stats = filter_stats(stats_pfasst, type='niter')
    # convert filtered statistics to list of iterations count, sorted by process
    iter_counts = sort_stats(filtered_stats, sortby='time')
    mean_niter = np.mean(np.array([item[1] for item in iter_counts]))

    err = np.linalg.norm(uend_pfasst.values - ref_sol, np.inf) / np.linalg.norm(ref_sol, np.inf)
    print('PFASST(10): Mean number of iterations: %6.3f -- Error: %8.4e' % (mean_niter, err))
    results.append((err, mean_niter))

    return results


if __name__ == "__main__":
    results = run_RDC()
