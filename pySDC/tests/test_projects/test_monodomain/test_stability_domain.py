import pytest


@pytest.mark.monodomain
def test_main():
    from pySDC.projects.Monodomain.run_scripts.run_TestODE import main

    main(100, False)
