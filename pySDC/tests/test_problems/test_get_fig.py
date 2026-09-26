import pytest


def get_problems():
    from pySDC.implementations.problem_classes.Brusselator import Brusselator
    from pySDC.implementations.problem_classes.Burgers import Burgers1D, Burgers2D
    from pySDC.implementations.problem_classes.RayleighBenard import RayleighBenard

    return {
        'Brusselator': lambda: Brusselator(nvars=(16, 16)),
        'Burgers1D': lambda: Burgers1D(N=16),
        'Burgers2D': lambda: Burgers2D(nx=16, nz=16),
        'RayleighBenard': lambda: RayleighBenard(nx=16, nz=8),
    }


@pytest.mark.base
@pytest.mark.parametrize('name', ['Brusselator', 'Burgers1D', 'Burgers2D', 'RayleighBenard'])
def test_get_fig_leaves_rcParams_alone(name):
    """
    get_fig used to switch on the constrained layout for every figure made afterwards, so that e.g. tight_layout()
    on a figure with a colorbar raised. The layout belongs to the figure get_fig makes, and to nothing else.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    problem = get_problems()[name]()
    # starting from the option switched off, and restoring all settings afterwards, whatever get_fig does to them
    with matplotlib.rc_context({'figure.constrained_layout.use': False}):
        before = dict(matplotlib.rcParams)
        try:
            fig = problem.get_fig()
            assert type(fig.get_layout_engine()).__name__ == 'ConstrainedLayoutEngine'
            changed = {key for key in before if repr(before[key]) != repr(matplotlib.rcParams[key])}
            assert not changed, f'{name}.get_fig changed global matplotlib settings: {changed}'
        finally:
            plt.close('all')
