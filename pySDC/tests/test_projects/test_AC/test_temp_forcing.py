from pySDC.projects.AllenCahn_Bayreuth.run_temp_forcing_verification import main


def test_main_serial():
    main(cwd='pySDC/projects/AllenCahn_Bayreuth/')
