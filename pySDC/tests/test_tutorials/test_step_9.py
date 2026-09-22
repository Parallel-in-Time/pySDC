import pytest


def _read(name):
    with open('data/' + name) as f:
        return [line for line in f.read().splitlines() if line.strip()]


def _parse(lines):
    """Turn the formatted output lines back into numbers, keyed by (mode, alpha)."""
    out = {}
    for line in lines:
        mode, rest = line.split(':', 1)
        alpha = rest.split('alpha ')[1].split('->')[0].strip()
        niter = int(rest.split('->')[1].split('iterations')[0])
        error = float(rest.split('error ')[1].split(',')[0])
        final_alpha = float(rest.split('final alpha ')[1])
        out[(mode.strip(), alpha)] = {'niter': niter, 'error': error, 'final_alpha': final_alpha}
    return out


def _collect_mpi_output(stem, sizes):
    """
    Stitch the per-rank-count files the MPI runs wrote into the one the tutorial documents.

    Each run writes its own file so the passes never have to agree on which of them truncates --
    they are separate processes, and the serial one runs last.
    """
    with open(f'data/{stem}_out.txt', 'w') as out:
        for n in sizes:
            with open(f'data/{stem}_np{n}.txt') as part:
                out.write(part.read())


@pytest.mark.base
def test_step_9_A():
    import pySDC.tutorial.step_9.A_paradiag_for_linear_problems


@pytest.mark.base
def test_step_9_B():
    import pySDC.tutorial.step_9.B_paradiag_for_nonlinear_problems


@pytest.mark.base
@pytest.mark.parametrize('problem', ['advection', 'vdp'])
def test_step_9_C(problem):

    from pySDC.tutorial.step_9.C_paradiag_in_pySDC import compare_ParaDiag_and_PFASST

    compare_ParaDiag_and_PFASST(n_steps=16, problem=problem)


@pytest.mark.mpi4py
def test_step_9_D():
    """Part D is serial: the alpha comparison with the virtually parallel controller."""
    from pySDC.tutorial.step_9.D_adaptive_alpha import alpha_settings, main as main_D

    main_D()

    results = _parse(_read('step_9_D_out.txt'))
    assert len(results) == len(alpha_settings), 'ERROR: expected one line per alpha, got %s' % len(results)

    # the adaptive strategy should need no more iterations than the best fixed alpha we tried
    fixed = [v['niter'] for (mode, a), v in results.items() if a != 'adaptive']
    assert results[('virtual', 'adaptive')]['niter'] <= min(
        fixed
    ), 'ERROR: adaptive alpha needed more iterations than the best fixed one'


# the block sizes Part E is run at; the marker and the comparison below must not drift apart
BLOCK_SIZES = [1, 2, 4]


@pytest.mark.mpi4py
@pytest.mark.parallel(BLOCK_SIZES)
def test_step_9_E_MPI():
    """One rank per time-step, so the communicator size is the block size."""
    from pathlib import Path
    from mpi4py import MPI
    from pySDC.tutorial.step_9.E_paradiag_MPI import main

    comm = MPI.COMM_WORLD
    fname = f'step_9_E_np{comm.size}.txt'
    if comm.rank == comm.size - 1:
        Path('data').mkdir(parents=True, exist_ok=True)
        open('data/' + fname, 'w').close()
    comm.Barrier()

    main(fname)


@pytest.mark.mpi4py
def test_step_9_E():
    """
    The MPI controller has to agree with Part D, and windowing must not change the answer.

    Runs after the rank passes above, which is where the per-block-size files come from.
    """
    from pySDC.tutorial.step_9.D_adaptive_alpha import alpha_settings, num_steps_total

    block_sizes = BLOCK_SIZES
    _collect_mpi_output('step_9_E', block_sizes)

    per_block = {n: _parse(_read(f'step_9_E_np{n}.txt')) for n in block_sizes}
    reference = _parse(_read('step_9_D_out.txt'))

    for n in block_sizes:
        assert len(per_block[n]) == len(alpha_settings), 'ERROR: expected one line per alpha at block size %s' % n

    # alpha is a property of the method, so at the block size Part D used the two controllers must agree
    for alpha in alpha_settings:
        mpi = per_block[num_steps_total][(f'MPI on {num_steps_total}', str(alpha))]
        virtual = reference[('virtual', str(alpha))]
        assert mpi['niter'] == virtual['niter'], 'ERROR: MPI and virtual differ in iterations for alpha %s' % alpha
        assert abs(mpi['error'] - virtual['error']) < 1e-12, (
            'ERROR: MPI and virtual differ in error for alpha %s' % alpha
        )

    # Windowing must not change what is being solved. Where the iteration count also matches, the
    # answers are identical; where it does not -- a loose alpha can cost a block one extra iteration
    # -- they still agree far inside the discretisation error of ~3e-5, the difference being leftover
    # iteration error rather than a different solution.
    for alpha in alpha_settings:
        errors = [per_block[n][(f'MPI on {n}', str(alpha))]['error'] for n in block_sizes]
        iters = {per_block[n][(f'MPI on {n}', str(alpha))]['niter'] for n in block_sizes}
        tol = 1e-12 if len(iters) == 1 else 1e-6
        assert max(errors) - min(errors) < tol, 'ERROR: errors differ between block sizes for alpha %s: %s' % (
            alpha,
            errors,
        )
