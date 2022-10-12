import pytest


@pytest.mark.petsc
def test_visualize_pySDC_with_PETSc():
    from pySDC.projects.TOMS.visualize_pySDC_with_PETSc import main

    main(cwd='pySDC/projects/TOMS/')
