import pytest


@pytest.mark.libpressio
@pytest.mark.parallel([1, 4])
@pytest.mark.parametrize("thresh", [1e-6, 1e-8])
def test_compression_proof_of_concept_MPI(thresh):
    from mpi4py import MPI

    run_single_test(thresh=thresh, useMPI=True, num_procs=MPI.COMM_WORLD.size)


@pytest.mark.libpressio
@pytest.mark.parametrize("thresh", [1e-6, 1e-8])
@pytest.mark.parametrize("num_procs", [1, 4])
def test_compression_proof_of_concept(thresh, num_procs):
    """The non-MPI controller runs the whole block itself, so this stays on one rank."""
    run_single_test(thresh=thresh, useMPI=False, num_procs=num_procs)


def run_single_test(thresh, useMPI, num_procs):
    print(f'Running with error bound {thresh} and {num_procs}. MPI: {useMPI}')
    import matplotlib.pyplot as plt
    import os
    from pySDC.projects.compression.order import plot_order_in_time

    fig, ax = plt.subplots(figsize=(3, 2))
    plot_order_in_time(ax=ax, thresh=thresh, useMPI=useMPI, num_procs=num_procs)
    if os.path.exists('data'):
        ax.set_title(f'{num_procs} procs, {"MPI" if useMPI else "non MPI"}')
        fig.savefig(f'data/compression_order_time_advection_d={thresh:.2e}_n={num_procs}_MPI={useMPI}.png', dpi=200)
