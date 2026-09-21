import pytest


@pytest.mark.mpi4py
def test_main():
    """The serial variants, which is all `main` does now."""
    from pySDC.projects.parallelSDC.AllenCahn_parallel import main

    main()


@pytest.mark.mpi4py
@pytest.mark.parallel(3)
@pytest.mark.parametrize('variant', ['sl_parallel', 'ml_parallel'])
def test_parallel_variants(variant):
    """
    The node-parallel variants need one rank per collocation node.

    `main` used to launch these itself, with `mpirun -np 3 python -c "from ... import *; ..."`.
    """
    from pySDC.projects.parallelSDC.AllenCahn_parallel import run_variant

    run_variant(variant=variant)
