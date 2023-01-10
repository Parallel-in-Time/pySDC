import pytest


@pytest.mark.base
def test_main():
    from pySDC.projects.AsympConv.PFASST_conv_tests import main

    main()


@pytest.mark.base
def test_Linf():
    from pySDC.projects.AsympConv.PFASST_conv_Linf import run_advection, run_diffusion

    QI = 'LU'
    run_diffusion(QI=QI, max_proc_exp=4)
    run_advection(QI=QI, max_proc_exp=4)

    QI = 'LU2'
    run_diffusion(QI=QI, max_proc_exp=4)
    run_advection(QI=QI, max_proc_exp=4)


@pytest.mark.base
def test_plot_results():
    from pySDC.projects.AsympConv.PFASST_conv_Linf import plot_results

    plot_results(cwd='pySDC/projects/AsympConv/')
