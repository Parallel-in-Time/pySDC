import pytest


@pytest.mark.base
def test_main():
    from pySDC.projects.DAE.run.fully_implicit_dae_playground import main

    main()
