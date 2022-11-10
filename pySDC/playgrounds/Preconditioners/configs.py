"""
Get configurations as well as functions for where to store and load for various runs here
"""
import numpy as np
import time
import pickle
import matplotlib.pyplot as plt

from pySDC.core.Collocation import CollBase
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order

from pySDC.projects.Resilience.vdp import run_vdp
from pySDC.projects.Resilience.piline import run_piline
from pySDC.projects.Resilience.advection import run_advection
from pySDC.projects.Resilience.heat import run_heat
from pySDC.projects.Resilience.dahlquist import run_dahlquist
from pySDC.playgrounds.Preconditioners.diagonal_precon_sweeper import DiagPrecon, DiagPreconIMEX
from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity, AdaptivityResidual


params_default = {'quad_type': 'RADAU-RIGHT', 'sweeper_params': {'num_nodes': 3}, 'QI': None}

params_vdp = {
    **params_default,
    'prob': run_vdp,
    'sweeper': DiagPrecon,
    'serial_sweeper': generic_implicit,
    'e_tol': 2e-5,
    'problem_params': {},
    'r_tol': 1.0,
    'k': 1,
    'e': 1,
    'e_em': 1.0,
    'name': 'vdp',
}


params_piline = {
    **params_default,
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


params_advection = {
    **params_default,
    'prob': run_advection,
    'sweeper': DiagPrecon,
    'serial_sweeper': generic_implicit,
    'e_tol': 1e-9,
    'problem_params': {'freq': -1, 'sigma': 6e-2, 'type': 'backward', 'order': 5, 'nvars': 2**9, 'c': 1.0},
    #'problem_params': {'freq': -1, 'sigma': 2e-2, 'type': 'backward', 'order': 5, 'nvars': 2**8, 'c':10.},
    #'problem_params': {'freq': -1, 'sigma': 1e-2, 'type': 'backward', 'order': 5, 'nvars': 2**6, 'c':1.},
    'r_tol': 2e-11,
    'k': 475,
    'e': 5.98e-8,
    'e_em': 5.91e-10,
    'name': 'advection',
    'derivative': 1,
    'L': 1.0,
    'Tend': 2e-1,
}


params_heat = {
    **params_default,
    'prob': run_heat,
    'sweeper': DiagPrecon,
    'serial_sweeper': generic_implicit,
    'e_tol': 1e-9,
    'problem_params': {
        'freq': -1,
        'order': 2,
        'type': 'forward',
        'nvars': 2**7,
        'nu': 1.0,
        'sigma': 6e-2,
    },
    'k': 0,
    'name': 'heat',
    'derivative': 2,
    'L': 1.0,
    'Tend': 2e-1,
}


re = np.linspace(-300, 1, 302)
im = np.linspace(-1, 100, 102)
lambdas = np.array([[complex(re[i], im[j]) for i in range(len(re))] for j in range(len(im))]).reshape(
    (len(re) * len(im))
)
# lambdas = np.append(re, im * 1j)


params_Dahlquist = {
    **params_default,
    'prob': run_dahlquist,
    'sweeper': DiagPrecon,
    'serial_sweeper': generic_implicit,
    'e_tol': 5e-4,
    'problem_params': {
        'lambdas': lambdas,
        'u0': 1.0 + 0.0j,
    },
    'k': 0,
    'name': 'Dahlquist',
    'derivative': None,
    'L': None,
    'Tend': 1,
}


problems = {
    'vdp': params_vdp,
    'piline': params_piline,
    'advection': params_advection,
    'heat': params_heat,
    'Dahlquist': params_Dahlquist,
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


def get_serial_preconditioner(params, **kwargs):
    """
    Replace sweeper with serial sweeper and add the corresponding preconditioner

    Args:
        params (dict): Parameters for running the problem

    Returns:
        dict: Updated params
    """
    allowed = ['LU', 'IE', 'MIN', 'MIN3']
    for precon in allowed:
        if kwargs.get(precon, False):
            params['sweeper'] = params['serial_sweeper']
            params['QI'] = precon
    return params


def get_params_for_stiffness_plot(problem, **kwargs):
    """
    Get params for different equations to reproduce plots from Robert's paper "Parallelizing spectral deferred
    corrections across the method"

    Args:
        problem (str): The name of the problem that has been run to obtain the preconditioner

    Returns:
        dict: Parameters for running a problem
        numpy.ndarray: List of values for the problem parameter
    """
    params = get_params(problem, **kwargs)

    params_heat = {
        'problem_params': {
            'freq': 2,
            'nvars': 63,
            'bc': 'dirichlet-zero',
            'type': 'center',
        },
        'range': np.logspace(-3, 2, 6),
        'parameter': 'nu',
    }

    params_advection = {
        'problem_params': {
            'freq': 2,
            'nvars': 64,
            'bc': 'periodic',
            'type': 'center',
            'order': 2,
        },
        'range': np.logspace(-3, 2, 6),
        'parameter': 'c',
    }

    params_vdp = {
        'problem_params': {'nvars': 2, 'newton_tol': 1e-9, 'newton_maxiter': 20, 'u0': np.array([2.0, 0.0])},
        'range': np.array([0.1 * 2**i for i in range(0, 10)]),
        'parameter': 'mu',
    }

    params_all = {
        'heat': params_heat,
        'advection': params_advection,
        'vdp': params_vdp,
    }

    assert problem in params_all.keys(), f"Don\'t have parameters for stiffness plot for problem \"{problem}\""
    special_params = params_all[problem]

    if not kwargs.get('adaptivity', True):
        params['convergence_controllers'] = {}
        params['step_params'] = {'maxiter': 100}
        params['level_params'] = {
            'restol': 1e-8,
            'dt': 0.1,
        }

    params['force_sweeper_params'] = {'initial_guess': 'spread'}

    for key in ['problem_params', 'level_params']:
        params[key] = {**params.get(key, {}), **special_params.get(key, {})}

    params['Tend'] = params['level_params']['dt']

    return params, special_params['parameter'], special_params['range']


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

    params = problems[problem].copy()

    params['convergence_controllers'] = get_convergence_controllers(params, **kwargs)

    params = get_serial_preconditioner(params, **kwargs)
    return params


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
        if kwargs[c]:
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


def store_precon(params, x, initial_guess, **kwargs):
    """
    Store the preconditioner of a specific optimization run.

    Args:
        problem (str): The name of the problem that has been run to obtain the preconditioner
        x (numpy.ndarray): The entries of the preconditioner
        initial_guess (numpy.ndarray): Initial guess to start the minimization problem

    Returns:
        None
    """
    data = {}

    # write the configuration in the data array
    configs = ['use_first_row', 'normalized', 'LU', 'IE', 'MIN', 'random_IG', 'MIN3']
    for c in configs:
        data[c] = kwargs.get(c, False)

    # get the sweeper parameters
    sweeper_params, sweeper = prepare_sweeper(x, params, **kwargs)

    # write the data into a dictionary
    data['time'] = time.time()
    data['x'] = x.copy()
    data['params'] = params
    data['kwargs'] = kwargs
    data['initial_guess'] = initial_guess
    data['sweeper_params'] = sweeper_params
    data['sweeper'] = sweeper

    # write to file
    with open(get_path(params['name'], sweeper_params['num_nodes'], **kwargs), 'wb') as file:
        pickle.dump(data, file)

    name = get_name(params['name'], sweeper_params['num_nodes'], **kwargs)
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
    params = get_params(problem, **kwargs)
    store_precon(params=params, x=np.zeros((nodes)), initial_guess=np.zeros((nodes)), **kwargs)


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


def get_collocation_nodes(params, num_nodes):
    """
    Get the nodes of the collocation problem

    Args:
        params (dict): Parameters for running the problem
        num_nodes (int): Number of collocation nodes
    """
    coll = CollBase(num_nodes, quad_type=params.get('quad_type', 'RADAU-RIGHT'))
    return coll.nodes


colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def prepare_sweeper(x, params, use_first_row=False, normalized=False, random_IG=False, **kwargs):
    """
    Prepare the sweeper with diagonal elements before running the problem

    Args:
        x (numpy.ndarray): The entries of the preconditioner
        params (dict): Parameters for setting up the run
        use_first_row (bool): Use the first row of the preconditioner or not
        normalize (bool): Normalize the quadrature weights or not
        random_IG (bool): Use random initial guess in the sweeper

    Returns
        dict: Sweeper parameters
    """
    # process options
    if use_first_row:
        if normalized:
            raise NotImplementedError

        diags = np.array(x[0 : len(x) // 2])
        first_row = np.array(x[len(x) // 2 : :])
        num_nodes = len(x) // 2
    else:
        if normalized:
            diags = np.array(np.append(x, -sum(x) + 1))
            first_row = np.zeros_like(diags)
            num_nodes = len(x) + 1
        else:
            diags = np.array(x)
            first_row = np.zeros_like(diags)
            num_nodes = len(x)

    if random_IG:
        initial_guess = 'random'
    else:
        initial_guess = 'spread'

    # setup the sweeper
    if None not in x:
        sweeper_params = {
            **params.get('sweeper_params', {}),
            'num_nodes': num_nodes,
            'diagonal_elements': diags,
            'first_row': first_row,
            'QI': params.get('QI', 'LU'),
            'quad_type': params.get('quad_type', 'RADAU-RIGHT'),
            'initial_guess': initial_guess,
        }
    else:
        sweeper_params = {}

    return sweeper_params, params['sweeper']
