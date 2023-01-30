import pytest


@pytest.mark.base
def test_problematic_main():
    from pySDC.projects.DAE.run.fully_implicit_dae_playground import main

    main()
