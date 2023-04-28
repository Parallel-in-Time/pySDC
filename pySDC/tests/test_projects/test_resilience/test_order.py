import pytest


@pytest.mark.base
@pytest.mark.parametrize("ks", [[2], [3], [4]])
@pytest.mark.parametrize("serial", [True, False])
def test_order_fixed_step_size(ks, serial):
    from pySDC.projects.Resilience.accuracy_check import plot_all_errors, plt

    fig, ax = plt.subplots()
    plot_all_errors(ax, ks, serial, Tend_fixed=1.0)


@pytest.mark.base
@pytest.mark.parametrize("ks", [[2], [3]])
@pytest.mark.parametrize("serial", [True, False])
def test_order_adaptive_step_size(ks, serial):
    print(locals())
    from pySDC.projects.Resilience.accuracy_check import plot_all_errors, plt

    fig, ax = plt.subplots()
    plot_all_errors(ax, ks, serial, Tend_fixed=5e-1, var='e_tol', dt_list=[1e-5, 5e-6], avoid_restarts=False)
