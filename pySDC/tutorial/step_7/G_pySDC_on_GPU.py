from pathlib import Path

from mpi4py import MPI

from pySDC.helpers.stats_helper import get_sorted
from pySDC.implementations.controller_classes.controller_MPI import controller_MPI
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_unforced
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh


def get_description(useGPU, ml):
    """
    Set up the heat equation, on a GPU or not, on one space level or two.

    The only thing that differs between the CPU and the GPU version is `useGPU`. The problem class
    is the same one either way: `setup_GPU` swaps the array library, the sparse library and the
    datatypes, so the body of the class goes on calling `self.xp.sin` and does not care.

    Args:
        useGPU (bool): Run on a GPU
        ml (bool): Use two space levels rather than one

    Returns:
        dict: description of the problem to be solved
    """
    level_params = {'restol': 1e-10, 'dt': 1e-2}

    sweeper_params = {'quad_type': 'RADAU-RIGHT', 'num_nodes': 3, 'QI': 'LU'}

    problem_params = {
        'nu': 0.1,
        'freq': 2,
        'bc': 'periodic',
        'nvars': [128, 64] if ml else 128,
        'useGPU': useGPU,
    }

    step_params = {'maxiter': 20}

    description = {
        'problem_class': heatNd_unforced,
        'problem_params': problem_params,
        'sweeper_class': generic_implicit,
        'sweeper_params': sweeper_params,
        'level_params': level_params,
        'step_params': step_params,
    }

    if ml:
        # the space transfer works on GPU arrays as well: the interpolation and restriction
        # matrices are assembled with SciPy and then moved to the device once
        description['space_transfer_class'] = mesh_to_mesh
        description['space_transfer_params'] = {'rorder': 2, 'iorder': 4, 'periodic': True}

    return description


def run(description, comm=None, num_procs=1, Tend=8e-2):
    """
    Run to `Tend` and report how it went.

    Args:
        description (dict): description of the problem to be solved
        comm (mpi4py.Intracomm): time communicator, for the parallel-in-time run
        num_procs (int): number of time steps to treat in parallel, for the serial controller
        Tend (float): time to run to

    Returns:
        float: error against the exact solution
        int: total number of iterations
    """
    controller_params = {'logger_level': 30}

    if comm is None:
        controller = controller_nonMPI(
            num_procs=num_procs, controller_params=controller_params, description=description
        )
        prob = controller.MS[0].levels[0].prob
    else:
        # the parallel-in-time controller sends the solution from one time rank to the next as a
        # GPU array, which needs MPI to have been told to expect device pointers -- see the README
        controller = controller_MPI(controller_params=controller_params, description=description, comm=comm)
        prob = controller.S.levels[0].prob

    uinit = prob.u_exact(0.0)
    uend, stats = controller.run(u0=uinit, t0=0.0, Tend=Tend)

    error = abs(prob.u_exact(Tend) - uend)
    iterations = sum(count for _, count in get_sorted(stats, type='niter', comm=comm))

    return error, iterations


def main():
    """
    Solve the same heat equation with SDC, MLSDC and PFASST, all of it on GPUs.

    The three differ only in what they are given: one space level and one time step at a time is
    SDC, two space levels is MLSDC, and two space levels spread over several time ranks is PFASST.
    """
    comm = MPI.COMM_WORLD

    # every rank runs the two serial variants -- they are the reference the parallel one is judged
    # against, and running them everywhere keeps the ranks in step
    runs = [
        ('SDC   ', *run(get_description(useGPU=True, ml=False))),
        ('MLSDC ', *run(get_description(useGPU=True, ml=True))),
        ('PFASST', *run(get_description(useGPU=True, ml=True), comm=comm)),
    ]

    if comm.rank == 0:
        Path('data').mkdir(parents=True, exist_ok=True)
        with open('data/step_7_G_out.txt', 'a') as f:
            for name, error, iterations in runs:
                out = f'{name} on {comm.size} GPU(s): error {error:.4e}, {iterations} iterations in total'
                f.write(out + '\n')
                print(out)

    # all three solve the same problem, so they had better agree on the answer
    errors = [error for _, error, _ in runs]
    assert max(errors) < 1e-8, f'Some run was not accurate enough: {errors}'
    assert abs(errors[1] - errors[2]) < 1e-10, 'PFASST and MLSDC disagree, which they should not'


if __name__ == '__main__':
    main()
