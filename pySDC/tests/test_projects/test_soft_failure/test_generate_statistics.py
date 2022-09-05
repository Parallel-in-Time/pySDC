import pytest


@pytest.mark.base
def test_generate_statistics():
    from pySDC.projects.soft_failure.generate_statistics import main

    main()
