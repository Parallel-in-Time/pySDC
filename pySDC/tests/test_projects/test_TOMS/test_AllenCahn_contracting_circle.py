import pytest


@pytest.mark.slow
def test_AllenCahn_contracting_circle():
    from pySDC.projects.TOMS.AllenCahn_contracting_circle import main

    main(cwd='pySDC/projects/TOMS/')
