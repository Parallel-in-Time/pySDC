import pytest

from pySDC.projects.TOMS.AllenCahn_contracting_circle import main

@pytest.mark.slow
def test_AllenCahn_contracting_circle():
    main(cwd='pySDC/projects/TOMS/')
