import pytest

from pySDC.projects.FEniCS_MLSDC.setups import EXAMPLES


@pytest.mark.fenics
@pytest.mark.parametrize('example', EXAMPLES)
@pytest.mark.parametrize('nlevels', [1, 2, 3])
def test_description_is_mass_only(example, nlevels):
    """Every setup must take the mass route and keep the collocation nodes across levels."""
    from pySDC.implementations.sweeper_classes.imex_1st_order_mass import imex_1st_order_mass
    from pySDC.implementations.sweeper_classes.generic_implicit_mass import generic_implicit_mass
    from pySDC.implementations.transfer_classes.BaseTransfer_mass import base_transfer_mass
    from pySDC.projects.FEniCS_MLSDC.setups import get_description

    description, controller_params, t0, Tend = get_description(example, nlevels=nlevels)

    assert description['sweeper_class'] in (imex_1st_order_mass, generic_implicit_mass)
    assert len(set(description['sweeper_params']['num_nodes'])) == 1, 'nodes must not be coarsened'
    assert len(description['sweeper_params']['num_nodes']) == nlevels
    assert len(description['problem_params']['refinements']) == nlevels
    assert Tend > t0 and controller_params['logger_level'] == 30

    if nlevels > 1:
        assert description['base_transfer_class'] is base_transfer_mass
        assert description['base_transfer_params']['finter'] is False
    else:
        assert 'base_transfer_class' not in description


@pytest.mark.fenics
def test_overrides():
    from pySDC.projects.FEniCS_MLSDC.setups import get_description

    description, _, t0, Tend = get_description('heat', nlevels=2, dt=0.5, restol=1e-5, maxiter=7, nsteps=3)
    assert description['level_params'] == {'restol': 1e-5, 'dt': 0.5}
    assert description['step_params']['maxiter'] == 7
    assert Tend == pytest.approx(1.5) and t0 == 0.0


@pytest.mark.fenics
def test_tolerance_lookup():
    from pySDC.projects.FEniCS_MLSDC.setups import get_tolerance

    assert all(get_tolerance(e) > 0 for e in EXAMPLES)


@pytest.mark.fenics
@pytest.mark.parametrize(
    'kwargs, match',
    [({'example': 'nope'}, 'unknown example'), ({'example': 'heat', 'nlevels': 4}, 'nlevels must be')],
)
def test_bad_input_raises(kwargs, match):
    from pySDC.projects.FEniCS_MLSDC.setups import get_description

    with pytest.raises(ValueError, match=match):
        get_description(**kwargs)


@pytest.mark.fenics
def test_unknown_example_tolerance_raises():
    from pySDC.projects.FEniCS_MLSDC.setups import get_tolerance

    with pytest.raises(ValueError, match='unknown example'):
        get_tolerance('nope')
