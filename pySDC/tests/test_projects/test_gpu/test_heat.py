import pytest


@pytest.mark.cupy
def test_main():
    from pySDC.projects.GPU.heat import main

    main()
