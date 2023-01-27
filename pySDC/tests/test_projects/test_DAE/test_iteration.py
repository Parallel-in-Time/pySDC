import pytest


@pytest.mark.base
def test_main():
    from pySDC.projects.DAE.run.run_iteration_test import main

    main()
