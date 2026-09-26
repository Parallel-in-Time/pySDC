import pytest

from pySDC.projects.FEM_with_FEniCS.setups import (
    COARSENINGS,
    EXAMPLES,
    FAMILIES,
    get_coarsenings,
    get_families,
    get_order_study,
)

#: Every combination an example actually has a problem class for. Not the full product: the
#: vortex is 2d and the DG problem classes are 1d.
CASES = [(e, f, c) for e in EXAMPLES for f in get_families(e) for c in get_coarsenings(e)]
BOTH_COARSENINGS = [(e, f) for e in EXAMPLES for f in get_families(e) if len(get_coarsenings(e)) > 1]
HAS_DG = [e for e in EXAMPLES if 'DG' in get_families(e)]


@pytest.mark.fenics
@pytest.mark.parametrize('example, family, coarsening', CASES)
@pytest.mark.parametrize('nlevels', [1, 2, 3])
def test_description_is_mass_only(example, family, coarsening, nlevels):
    """Every setup must take the mass route and keep the collocation nodes across levels."""
    from pySDC.implementations.sweeper_classes.imex_1st_order_mass import imex_1st_order_mass
    from pySDC.implementations.sweeper_classes.generic_implicit_mass import generic_implicit_mass
    from pySDC.implementations.transfer_classes.BaseTransfer_mass import base_transfer_mass
    from pySDC.projects.FEM_with_FEniCS.setups import get_description

    description, controller_params, t0, Tend = get_description(
        example, nlevels=nlevels, family=family, coarsening=coarsening
    )

    assert description['sweeper_class'] in (imex_1st_order_mass, generic_implicit_mass)
    assert len(set(description['sweeper_params']['num_nodes'])) == 1, 'nodes must not be coarsened'
    assert len(description['sweeper_params']['num_nodes']) == nlevels
    assert description['problem_params']['family'] == family
    assert Tend > t0 and controller_params['logger_level'] == 30

    if nlevels > 1:
        assert description['base_transfer_class'] is base_transfer_mass
        assert description['base_transfer_params']['finter'] is False
    else:
        assert 'base_transfer_class' not in description


@pytest.mark.fenics
@pytest.mark.parametrize('example, family', BOTH_COARSENINGS)
@pytest.mark.parametrize('nlevels', [1, 2, 3])
def test_only_the_chosen_direction_coarsens(example, family, nlevels):
    """h-coarsening steps down the mesh at fixed order, p-coarsening the other way round."""
    from pySDC.projects.FEM_with_FEniCS.setups import get_description

    h_params = get_description(example, nlevels=nlevels, family=family, coarsening='h')[0]['problem_params']
    p_params = get_description(example, nlevels=nlevels, family=family, coarsening='p')[0]['problem_params']

    refinements, order = h_params['refinements'], h_params['order']
    assert len(refinements) == nlevels and all(b < a for a, b in zip(refinements, refinements[1:]))
    assert isinstance(order, int)

    refinements, order = p_params['refinements'], p_params['order']
    assert isinstance(refinements, int) and refinements == h_params['refinements'][0]
    assert len(order) == nlevels and all(b < a for a, b in zip(order, order[1:]))

    if nlevels == 1:
        assert h_params['refinements'][0] == p_params['refinements']
        assert h_params['order'] == p_params['order'][0], 'SDC must not depend on the coarsening'


@pytest.mark.fenics
@pytest.mark.parametrize('example', HAS_DG)
def test_dg_picks_the_dg_problem_class(example):
    from pySDC.projects.FEM_with_FEniCS.problem_classes import DG_1D_FEniCS
    from pySDC.projects.FEM_with_FEniCS.setups import get_description

    cg = get_description(example, family='CG')[0]['problem_class']
    dg = get_description(example, family='DG')[0]['problem_class']

    assert dg is not cg and issubclass(dg, cg)
    assert dg.__module__ == DG_1D_FEniCS.__name__


@pytest.mark.fenics
def test_overrides():
    from pySDC.projects.FEM_with_FEniCS.setups import get_description

    description, _, t0, Tend = get_description('heat', nlevels=2, dt=0.5, restol=1e-5, maxiter=7, nsteps=3)
    assert description['level_params'] == {'restol': 1e-5, 'dt': 0.5}
    assert description['step_params']['maxiter'] == 7
    assert Tend == pytest.approx(1.5) and t0 == 0.0


@pytest.mark.fenics
def test_tolerance_lookup():
    from pySDC.projects.FEM_with_FEniCS.setups import get_tolerance

    assert all(get_tolerance(e) > 0 for e in EXAMPLES)


@pytest.mark.fenics
@pytest.mark.parametrize(
    'kwargs, match',
    [
        ({'example': 'nope'}, 'unknown example'),
        ({'example': 'heat', 'nlevels': 4}, 'nlevels must be'),
        ({'example': 'heat', 'family': 'RT'}, 'unknown family'),
        ({'example': 'heat', 'coarsening': 'hp'}, 'unknown coarsening'),
    ],
)
def test_bad_input_raises(kwargs, match):
    from pySDC.projects.FEM_with_FEniCS.setups import get_description

    with pytest.raises(ValueError, match=match):
        get_description(**kwargs)


@pytest.mark.fenics
def test_pfasst_procs_lookup():
    from pySDC.projects.FEM_with_FEniCS.setups import get_pfasst_procs

    for example in EXAMPLES:
        procs = get_pfasst_procs(example)
        assert procs[0] == 1 and all(b > a for a, b in zip(procs, procs[1:]))


@pytest.mark.fenics
@pytest.mark.parametrize('lookup', ['get_tolerance', 'get_pfasst_procs', 'get_families', 'get_coarsenings'])
def test_unknown_example_lookup_raises(lookup):
    import pySDC.projects.FEM_with_FEniCS.setups as setups

    with pytest.raises(ValueError, match='unknown example'):
        getattr(setups, lookup)('nope')


@pytest.mark.fenics
@pytest.mark.parametrize(
    'kwargs, match',
    [
        ({'example': 'vortex', 'family': 'DG'}, 'no DG problem class'),
        ({'example': 'vortex', 'coarsening': 'p'}, 'only set up for'),
    ],
)
def test_unsupported_combination_raises(kwargs, match):
    """An example without a problem class for the combination has to say so, not build a broken one."""
    from pySDC.projects.FEM_with_FEniCS.setups import get_description

    with pytest.raises(ValueError, match=match):
        get_description(**kwargs)


@pytest.mark.fenics
def test_every_case_is_buildable():
    """Whatever an example declares it supports, `get_description` has to produce."""
    from pySDC.projects.FEM_with_FEniCS.setups import get_description

    for example, family, coarsening in CASES:
        for nlevels in (1, 2, 3):
            description = get_description(example, nlevels=nlevels, family=family, coarsening=coarsening)[0]
            assert description['problem_class'] is not None, f'{example} [{family}, {coarsening}] has no problem class'


@pytest.mark.fenics
@pytest.mark.parametrize('example', [e for e in EXAMPLES if get_order_study(e)])
def test_order_study_holds_the_dof_count(example):
    """
    The order comparison is only meaningful if every order gives the same dofs on every level --
    otherwise it measures the size of the coarse space rather than its quality.
    """
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.projects.FEM_with_FEniCS.setups import get_description

    dofs = {}
    for order in get_order_study(example):
        description, controller_params, _, _ = get_description(example, nlevels=3, order=order)
        controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)
        dofs[order] = [level.prob.init.dim() for level in controller.MS[0].levels]

    reference = dofs[get_order_study(example)[0]]
    for order, got in dofs.items():
        assert got == reference, f'{example} at order {order} gives {got}, not {reference}'
    assert len(set(reference)) == len(reference), f'{example} has two levels of the same size: {reference}'


@pytest.mark.fenics
def test_order_override_refuses_a_negative_refinement():
    """A shift past zero would silently give two identical levels, because dolfin skips the loop."""
    from pySDC.projects.FEM_with_FEniCS.setups import get_description

    with pytest.raises(ValueError, match='negative refinement'):
        get_description('vortex', nlevels=3, order=4)
