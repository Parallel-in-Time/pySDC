"""
This script shows how to run ParaDiag with actual MPI parallelism across the time-steps.

Part C ran ParaDiag with the "virtually parallel" controller, which holds all steps in one process and
is what you want for developing and debugging. Here we use the MPI controller instead: one time-step
per rank, with the communicator spanning the block that ParaDiag diagonalizes across.

The point of this part is that nothing about the *method* changes. The description and the controller
parameters are the same ones Part C would use; only the controller class differs. So you can develop a
setup serially and then run it in parallel without touching it.

We always integrate the same total number of time-steps and only vary how many of them are done in
parallel. With four steps in total and a block size of one, two or four, that means four, two or one
block respectively, so the controller has to window through the time domain block by block. Windowing
works the same way in both controllers, which is what we check: for every block size we run the MPI
controller and the virtually parallel one and compare.

Two properties of ParaDiag are worth keeping in mind when going parallel, because they are different
from PFASST:

- All steps of a block have to iterate together. In PFASST an early step can converge and drop out of
  the iteration, which is what makes it pipelined. ParaDiag cannot do that: the transform in time
  needs every step, so a step that stopped early would leave the others waiting forever.
- The block is therefore always full. If the end time does not divide into whole blocks, ParaDiag
  solves past it rather than truncating, and says so.
"""

import os
import subprocess

# we always do this many time-steps in total, no matter how many of them run in parallel
num_steps_total = 4


def get_description():
    """
    Set up the same advection problem as in Part C.

    Returns:
        dict: the description for the ParaDiag controller
    """
    from pySDC.implementations.problem_classes.AdvectionEquation_ND_FD import advectionNd
    from pySDC.implementations.sweeper_classes.ParaDiagSweepers import QDiagonalization

    level_params = {}
    level_params['dt'] = 0.1
    level_params['restol'] = 1e-6

    sweeper_params = {}
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 3
    sweeper_params['initial_guess'] = 'copy'

    # Part C uses GMRES here to count linear solver work. We only care about the parallelism, and the
    # complex shifted systems ParaDiag produces are hard for GMRES, so we solve them directly instead.
    problem_params = {'nvars': 64, 'order': 8, 'c': 1, 'solver_type': 'direct'}

    step_params = {}
    step_params['maxiter'] = 99

    description = {}
    description['problem_class'] = advectionNd
    description['problem_params'] = problem_params
    description['sweeper_class'] = QDiagonalization
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    return description


def get_controller_params():
    """
    Set up controller parameters for ParaDiag.

    `alpha` may also be a list or a callable of the iteration index if you want to start with a
    well-conditioned value and tighten it later. We keep it fixed here.

    Returns:
        dict: the controller parameters
    """
    controller_params = {}
    controller_params['logger_level'] = 30
    controller_params['alpha'] = 1e-4

    # the advection problem is linear, so we do not need the extra communication for average Jacobians
    controller_params['average_jacobian'] = False

    return controller_params


def format_result(mode, block_size, niter, error):
    """
    One line of output, in the same shape for both controllers so they can be compared.

    Args:
        mode (str): 'MPI' or 'virtual'
        block_size (int): number of time-steps done in parallel
        niter (list): number of iterations of each step
        error (float): error against the exact solution at the end

    Returns:
        str: the formatted line
    """
    num_blocks = num_steps_total // block_size
    return (
        f'{mode:>7s}: block size {block_size}, {num_blocks} block(s) of {block_size} step(s), '
        f'iterations {niter}, error {error:.4e}'
    )


def run_virtual(block_size):
    """
    Run the same setup with the virtually parallel controller.

    Args:
        block_size (int): number of time-steps in one block

    Returns:
        tuple: the end value, the iteration counts and the error
    """
    import numpy as np
    from pySDC.helpers.stats_helper import get_sorted
    from pySDC.implementations.controller_classes.controller_ParaDiag_nonMPI import controller_ParaDiag_nonMPI

    controller_params = get_controller_params()
    controller_params['mssdc_jac'] = False

    controller = controller_ParaDiag_nonMPI(
        controller_params=controller_params, description=get_description(), num_procs=block_size
    )

    # ParaDiag diagonalizes in time, so the solution becomes complex
    for S in controller.MS:
        S.levels[0].prob.init = tuple([*S.levels[0].prob.init[:2]] + [np.dtype('complex128')])

    P = controller.MS[0].levels[0].prob
    dt = controller.MS[0].levels[0].params.dt
    Tend = num_steps_total * dt

    uend, stats = controller.run(u0=P.u_exact(0.0), t0=0.0, Tend=Tend)
    niter = [int(me[1]) for me in get_sorted(stats, type='niter', sortby='time')]
    return uend, niter, abs(uend - P.u_exact(Tend))


def main(cwd):
    """
    A simple test program to test the MPI-parallel ParaDiag controller

    Args:
        cwd (str): current working directory
    """

    # try to import MPI here, will fail if things go wrong (and not in the subprocess part)
    try:
        import mpi4py

        del mpi4py
    except ImportError as e:
        raise ImportError('ParaDiag with MPI needs mpi4py') from e

    import numpy as np

    # Set python path once
    my_env = os.environ.copy()
    my_env['PYTHONPATH'] = '../../..:.'
    my_env['COVERAGE_PROCESS_START'] = 'pyproject.toml'

    # one time-step per rank, so the number of ranks is the block size
    block_sizes = [1, 2, 4]

    # set up new/empty file for output
    fname = 'step_9_D_out.txt'
    f = open(cwd + '/../../../data/' + fname, 'w')
    f.close()

    # run the MPI controller with different block sizes, always doing num_steps_total steps in total
    for block_size in block_sizes:
        print('Running ParaDiag with block size %2i...' % block_size)
        cmd = ('mpirun -np ' + str(block_size) + ' python playground_ParaDiag_MPI.py ../../../../data/' + fname).split()
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=my_env, cwd=cwd)
        p.wait()
        assert p.returncode == 0, 'ERROR: did not get return code 0, got %s with %2i processes' % (
            p.returncode,
            block_size,
        )

    # now do the same with the virtually parallel controller and append the results
    f = open(cwd + '/../../../data/' + fname, 'a')
    virtual = {}
    for block_size in block_sizes:
        uend, niter, error = run_virtual(block_size)
        virtual[block_size] = uend
        out = format_result('virtual', block_size, niter, error)
        f.write(out + '\n')
        print(out)
    f.close()

    # windowing must not change the answer: every block size integrates to the same end time
    reference = virtual[block_sizes[0]]
    for block_size in block_sizes[1:]:
        assert np.allclose(
            virtual[block_size], reference, atol=1e-9
        ), 'ERROR: virtual ParaDiag gives different results for different block sizes'


if __name__ == "__main__":
    main('.')
