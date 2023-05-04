import pytest


@pytest.mark.libpressio
@pytest.mark.parametrize("thresh", [1e-6, 1e-8])
@pytest.mark.parametrize("useMPI", [True, False])
@pytest.mark.parametrize("num_procs", [1, 4])
def test_compression_proof_of_concept(thresh, useMPI, num_procs):
    if useMPI:
        import subprocess
        import os

        # Setup environment
        my_env = os.environ.copy()

        cmd = f"mpirun -np {num_procs} python {__file__} -t {thresh} -M {useMPI} -n {num_procs}".split()

        p = subprocess.Popen(cmd, env=my_env, cwd=".")

        p.wait()
        assert p.returncode == 0, 'ERROR: did not get return code 0, got %s with %2i processes' % (
            p.returncode,
            num_procs,
        )
    else:
        run_single_test(thresh=thresh, useMPI=useMPI, num_procs=num_procs)


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


if __name__ == '__main__':
    import sys

    # defaults for arguments
    num_procs = 1
    useMPI = False
    thresh = -1

    # parse command line arguments
    for i in range(len(sys.argv)):
        if sys.argv[i] == '-n':
            num_procs = int(sys.argv[i + 1])
        elif sys.argv[i] == '-M':
            useMPI = True if sys.argv[i + 1] == 'True' else False
        elif sys.argv[i] == '-t':
            thresh = float(sys.argv[i + 1])

    # execute test
    if '--use-subprocess' in sys.argv:
        test_compression_proof_of_concept(thresh=thresh, useMPI=useMPI, num_procs=num_procs)
    else:
        run_single_test(thresh=thresh, useMPI=useMPI, num_procs=num_procs)
