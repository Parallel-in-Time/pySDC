import pytest


@pytest.mark.petsc
@pytest.mark.timeout(600)
def test_grayscott():
    from pySDC.projects.SDC_showdown.SDC_timing_GrayScott import main

    main(cwd='pySDC/projects/SDC_showdown/')
