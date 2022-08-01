def test_create_plots():
    from pySDC.projects.node_failure.postproc_hard_faults_detail import create_plots

    create_plots(setup='HEAT', cwd='pySDC/projects/node_failure/')
    create_plots(setup='ADVECTION', cwd='pySDC/projects/node_failure/')
