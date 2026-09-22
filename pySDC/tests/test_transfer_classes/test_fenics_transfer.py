import numpy as np
import pytest

PROBLEM_PARAMS = {'c_nvars': 32, 't0': 0.0, 'family': 'CG', 'nu': 0.1, 'c': 1.0}

# Levels can differ in mesh refinement, in polynomial degree, or in both.
NESTED_SPACES = [((1, 0), (1, 1)), ((2, 0), (1, 1)), ((1, 0), (4, 1)), ((1, 1), (4, 1))]

SAME_MESH = {'refinements': (1, 1), 'order': (4, 1)}


def get_transfer(refinements=(1, 0), order=(1, 1)):
    """
    Set up `mesh_to_mesh_fenics` between two function spaces.

    Returns:
        tuple: the transfer object, the fine problem and the coarse problem
    """
    from pySDC.implementations.problem_classes.HeatEquation_1D_FEniCS_matrix_forced import fenics_heat
    from pySDC.implementations.transfer_classes.TransferFenicsMesh import mesh_to_mesh_fenics

    fine = fenics_heat(refinements=refinements[0], order=order[0], **PROBLEM_PARAMS)
    coarse = fenics_heat(refinements=refinements[1], order=order[1], **PROBLEM_PARAMS)
    return mesh_to_mesh_fenics(fine, coarse, {}), fine, coarse


@pytest.mark.fenics
@pytest.mark.parametrize('refinements, order', NESTED_SPACES)
def test_fenics_prolongation_is_exact(refinements, order):
    """
    The coarse space is a subspace of the fine one, whether the levels differ in mesh size or in
    polynomial degree. Interpolating a coarse function up therefore loses nothing, and restricting
    it has to give it back.
    """
    transfer, _, coarse = get_transfer(refinements=refinements, order=order)

    u_coarse = coarse.u_exact(0.0)
    recovered = transfer.restrict(transfer.prolong(u_coarse))

    assert abs(recovered - u_coarse) < 1e-13, 'Restriction did not undo the prolongation'


@pytest.mark.fenics
@pytest.mark.parametrize('refinements, order', NESTED_SPACES)
def test_fenics_restriction_is_nodal_injection(refinements, order):
    """
    Restriction interpolates rather than projects, so the coarse degrees of freedom are read off the
    fine function directly and restricting the fine interpolant is exact.
    """
    transfer, fine, coarse = get_transfer(refinements=refinements, order=order)

    restricted = transfer.restrict(fine.u_exact(0.0))

    assert abs(restricted - coarse.u_exact(0.0)) < 1e-13, 'Restriction is not nodal injection'


@pytest.mark.fenics
@pytest.mark.parametrize('refinements, order', NESTED_SPACES)
def test_fenics_project_recovers_the_coarse_space(refinements, order):
    """
    `project` is the other way down, used by `BaseTransfer_mass`. On a function that came from the
    coarse space it has to return that function, however the levels differ.
    """
    transfer, _, coarse = get_transfer(refinements=refinements, order=order)

    u_coarse = coarse.u_exact(0.0)

    assert abs(transfer.project(transfer.prolong(u_coarse)) - u_coarse) < 1e-13, 'project lost the coarse function'


@pytest.mark.fenics
@pytest.mark.parametrize('refinements, order', [((1, 1), (4, 1)), ((1, 0), (4, 1))])
def test_fenics_project_differs_from_restrict(refinements, order):
    """
    `project` is an L2 projection and `restrict` is interpolation, so on a function the coarse space
    cannot represent they have to differ. Otherwise `BaseTransfer_mass`, the only caller of
    `project`, would be getting interpolation by accident.
    """
    transfer, fine, _ = get_transfer(refinements=refinements, order=order)

    u_fine = fine.u_exact(0.0)

    assert abs(transfer.restrict(u_fine) - transfer.project(u_fine)) > 1e-8, 'project is just restrict'


@pytest.mark.fenics
@pytest.mark.parametrize('direction', ['restrict', 'prolong', 'project'])
def test_fenics_rhs_components(direction):
    """Each component of a right hand side has to be treated exactly as a single mesh would be."""
    from pySDC.implementations.datatype_classes.fenics_mesh import fenics_mesh, rhs_fenics_mesh

    transfer, fine, coarse = get_transfer(**SAME_MESH)
    source_prob = coarse if direction == 'prolong' else fine

    rhs = source_prob.eval_f(source_prob.u_exact(0.0), 0.0)
    assert type(rhs) is rhs_fenics_mesh, 'Expected the problem to produce a multi-component right hand side'

    transferred = getattr(transfer, direction)(rhs)
    assert type(transferred) is rhs_fenics_mesh, f'Expected rhs_fenics_mesh back, got {type(transferred).__name__}'

    for component in ['impl', 'expl']:
        single = getattr(transfer, direction)(fenics_mesh(getattr(rhs, component)))
        assert (
            abs(fenics_mesh(getattr(transferred, component)) - single) < 1e-13
        ), f'Component {component!r} was not treated like a single mesh'

    assert abs(fenics_mesh(transferred.impl) - fenics_mesh(transferred.expl)) > 1e-8, 'The components got mixed up'


@pytest.mark.fenics
@pytest.mark.parametrize('direction', ['restrict', 'prolong', 'project'])
def test_fenics_rejects_unknown_types(direction):
    """Anything that is not a FEniCS datatype has to be refused."""
    from pySDC.core.errors import TransferError

    transfer, _, _ = get_transfer()
    with pytest.raises(TransferError):
        getattr(transfer, direction)(np.zeros(3))


if __name__ == '__main__':
    test_fenics_restriction_is_nodal_injection((1, 0), (1, 1))
