import pytest
import warnings



@pytest.mark.base
def test_problematic_main():
    from pySDC.projects.DAE.run.fully_implicit_dae_playground import main

    main()
    

@pytest.mark.base
def test_synch_gen_playground_main():
    from pySDC.projects.DAE.run.synchronous_machine_playground import main

    warnings.filterwarnings('ignore')
    main()
    warnings.resetwarnings()

