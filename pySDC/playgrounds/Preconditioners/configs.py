"""
Get configurations as well as functions for where to store and load for various runs here
"""
import numpy as np
import time
import pickle

from pySDC.core.Collocation import CollBase
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order

from pySDC.projects.Resilience.vdp import run_vdp
from pySDC.projects.Resilience.piline import run_piline
from pySDC.projects.Resilience.advection import run_advection
from pySDC.playgrounds.Preconditioners.diagonal_precon_sweeper import DiagPrecon, DiagPreconIMEX
from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity, AdaptivityResidual


args_default = {
    'quad_type': 'RADAU-RIGHT',
}


args_vdp = {
    **args_default,
    'prob': run_vdp,
    'sweeper': DiagPrecon,
    'serial_sweeper': generic_implicit,
    'e_tol': 2e-5,
    'problem_params': None,
    'r_tol': 1.,
    'k': 1,
    'e': 1,
    'e_em': 1.,
    'name': 'vdp',
}


args_piline = {
    **args_default,
    'prob': run_piline,
    'sweeper': DiagPreconIMEX,
    'serial_sweeper': imex_1st_order,
    'e_tol': 1e-7,
    'problem_params': None,
    'r_tol': np.inf,
    'k': 2461,
    'e': 4.14e-8,
    'e_em': 7.27e-8,
    'name': 'piline',
}


args_advection = {
    **args_default,
    'prob': run_advection,
    'sweeper': DiagPrecon,
    'serial_sweeper': generic_implicit,
    'e_tol': 1e-9,
    'problem_params': {'freq': -1, 'sigma': 6e-2},
    'r_tol': 2e-11,
    'k': 475,
    'e': 5.98e-8,
    'e_em': 5.91e-10,
    'name': 'advection',
}


problems = {
    'vdp': args_vdp,
    'piline': args_piline,
    'advection': args_advection,
}


def get_convergence_controllers(params, adaptivity=False, residual_adaptivity=False, **kwargs):
    """
    Prepare convergence controllers for a run with a specific configuration

    Args:
        adaptivity (bool): Add the adaptivity convergence controller
        residual_adaptivity (bool): Add the convergence controller for residual based adaptivity

    Returns:
        dict: Convergence controllers
    """
    convergence_controllers = {}

    if adaptivity:
        convergence_controllers[Adaptivity] = {'e_tol': params['e_tol']}

    if residual_adaptivity:
        convergence_controllers[AdaptivityResidual] = {'e_tol': params['r_tol'], 'max_restarts': 99}

    return convergence_controllers


def get_serial_preconditioner(params, LU=False, IE=False, **kwargs):
    """
    Replace sweeper with serial sweeper and add the corresponding preconditioner

    Args:
        LU (bool): LU preconditioner
        IE (bool): serial implicit Euler preconditioner

    Returns:
        dict: Updated params
    """
    if LU:
        params['sweeper'] = params['serial_sweeper']
        params['QI'] = 'LU'
    elif IE:
        params['sweeper'] = params['serial_sweeper']
        params['QI'] = 'IE'
    return params


def get_params(problem, **kwargs):
    """
    Get parameters for specific configuration

    Args:
        problem (str): The name of the problem that has been run to obtain the preconditioner

    Returns:
        dict: Parameters for running a problem
    """
    if problem not in problems.keys():
        raise NotImplementedError(f'{problem} has no predefined parameters in `configs.py`!')

    args = problems[problem]

    args['convergence_controllers'] = get_convergence_controllers(args, **kwargs)

    args = get_serial_preconditioner(args, **kwargs)
    return args


def get_name(problem, nodes, **kwargs):
    '''
    Get the name of a preconditioner for storing and loding

    Args:
        problem (str): The name of the problem that has been run to obtain the preconditioner
        nodes (int): Number of nodes that have been used

    Returns:
        str: Name of the preconditioner
    '''
    name = f'{problem}'
    for c in kwargs.keys():
        name = f'{name}-{c}'
    name += f'-{nodes}nodes'
    return name


def get_path(problem, nodes, **kwargs):
    """
    Get the path to a preconditioner

    Args:
        problem (str): The name of the problem that has been run to obtain the preconditioner
        nodes (int): Number of nodes that have been used

    Returns:
        str: Path to the preconditioner
    """
    return f'data/precons/{get_name(problem, nodes, **kwargs)}.pickle'


def store_precon(args, x, initial_guess, **kwargs):
    """
    Store the preconditioner of a specific optimization run.

    Args:
        problem (str): The name of the problem that has been run to obtain the preconditioner
        nodes (int): Number of nodes that have been used

    Returns:
        None
    """
    data = {}

    # defaults
    data['QI'] = None
    data['diags'] = np.zeros_like(x)
    data['first_row'] = np.zeros_like(x)

    # write the configuration in the data array
    configs = ['use_first_row', 'normalized', 'LU', 'IE']
    for c in configs:
        data[c] = kwargs.get(c, False)

    # write the data based on configuration
    if data['use_first_row']:
        data['diags'] = x[0: len(x) // 2]
        data['first_row'] = x[len(x) // 2::]
    elif data['LU'] or data['IE']:
        data['QI'] = args['QI']
    else:
        data['diags'] = x.copy()

    data['num_nodes'] = len(data['diags'])
    data['time'] = time.time()
    data['x'] = x.copy()
    data['args'] = args
    data['kwargs'] = kwargs
    data['initial_guess'] = initial_guess
    data['quad_type'] = args.get('quad_type', 'RADAU-RIGHT')

    with open(get_path(args['name'], data['num_nodes'], **kwargs), 'wb') as file:
        pickle.dump(data, file)

    name = get_name(args['name'], data['num_nodes'], **kwargs) 
    print(f'Stored preconditioner "{name}"')


def store_serial_precon(problem, nodes, **kwargs):
    """
    Store a serial preconditioner with placeholder values for the results of the optimization process

    Args:
        problem (str): The name of the problem that has been run to obtain the preconditioner
        nodes (int): Number of nodes that have been used

    Returns:
        None
    """
    args = get_params(problem, **kwargs)
    store_precon(args=args, x=np.zeros((nodes)), initial_guess=np.zeros((nodes)), **kwargs)


def load_precon(problem, nodes, **kwargs):
    '''
    Load a stored preconditioner

    Args:
        problem (str): The name of the problem that has been run to obtain the preconditioner
        nodes (int): Number of nodes that have been used

    Returns:
        dict: Preconditioner and options that have been used in obtaining it
    '''
    with open(get_path(problem, nodes, **kwargs), 'rb') as file:
        data = pickle.load(file)
    return data


def get_collocation_nodes(args, num_nodes):
    """
    Get the nodes of the collocation problem

    Args:
        args (dict): Parameters for running the problem
        num_nodes (int): Number of collocation nodes
    """
    coll = CollBase(num_nodes, quad_type=args.get('quad_type', 'RADAU-RIGHT'))
    return coll.nodes
