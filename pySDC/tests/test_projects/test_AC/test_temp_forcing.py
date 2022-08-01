import pytest


@pytest.mark.mpifft
def test_main_serial():
    from pySDC.projects.AllenCahn_Bayreuth.run_temp_forcing_verification import main

    main(cwd='pySDC/projects/AllenCahn_Bayreuth/')
