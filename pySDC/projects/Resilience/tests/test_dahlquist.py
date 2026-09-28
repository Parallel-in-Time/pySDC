import pytest


@pytest.mark.base
@pytest.mark.parametrize('QI, A_stable', [('IE', True), ('LU', True), ('PIC', False)])
def test_stability(QI, A_stable):
    """Implicit preconditioners keep SDC stable in the whole sampled left half plane, the explicit one does not."""
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pySDC.projects.Resilience.dahlquist import run_dahlquist, plot_stability, plot_contraction, plot_increment

    stats, _, _ = run_dahlquist(custom_description={'sweeper_params': {'QI': QI, 'initial_guess': 'spread'}})

    fig, axs = plt.subplots(1, 3)
    assert plot_stability(stats, ax=axs[0], iter=[1, 2, 3]) == A_stable
    plot_contraction(stats, fig=fig, ax=axs[1], iter=[0, 4])
    plot_increment(stats, fig=fig, ax=axs[2], iter=[0, 4])
    plt.close(fig)
