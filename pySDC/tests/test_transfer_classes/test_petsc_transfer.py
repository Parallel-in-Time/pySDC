import numpy as np
import pytest

PROBLEM_PARAMS = {'nu': 1.0, 'freq': 2, 'sol_tol': 1e-12}


def get_transfer(nvars_coarse=9, refine=1):
    """
    Set up `mesh_to_mesh_petsc_dmda` between a DMDA grid and its `refine`-times refined version.

    Returns:
        tuple: the transfer object, the fine problem and the coarse problem
    """
    from pySDC.implementations.problem_classes.HeatEquation_2D_PETSc_forced import heat2d_petsc_forced
    from pySDC.implementations.transfer_classes.TransferPETScDMDA import mesh_to_mesh_petsc_dmda

    cnvars = [nvars_coarse, nvars_coarse]
    fine = heat2d_petsc_forced(cnvars=cnvars, refine=refine, **PROBLEM_PARAMS)
    coarse = heat2d_petsc_forced(cnvars=cnvars, refine=0, **PROBLEM_PARAMS)
    return mesh_to_mesh_petsc_dmda(fine, coarse, {}), fine, coarse


def sine(prob):
    """`sin(pi x) sin(pi y)` sampled on the problem's grid, flattened the way PETSc stores it."""
    n = prob.init.getSizes()[0]
    x = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, x, indexing='ij')
    values = prob.dtype_u(prob.init)
    values.getArray()[:] = (np.sin(np.pi * X) * np.sin(np.pi * Y)).T.flatten()
    return values


@pytest.mark.petsc
def test_petsc_prolongation_reproduces_a_constant():
    """Whatever else the DMDA interpolation does, it has to leave a constant alone."""
    transfer, _, coarse = get_transfer()

    prolonged = transfer.prolong(coarse.dtype_u(coarse.init, val=3.0))

    assert np.allclose(prolonged.getArray(), 3.0), 'Interpolation does not reproduce a constant'


@pytest.mark.petsc
def test_petsc_restriction_is_injection():
    """The restriction is built as an injection, so it has to pick the coincident fine values."""
    transfer, fine, coarse = get_transfer()

    restricted = transfer.restrict(sine(fine))

    assert np.allclose(restricted.getArray(), sine(coarse).getArray()), 'Restriction is not injection'


@pytest.mark.petsc
def test_petsc_prolongation_order():
    """
    The DMDA interpolation is bilinear, so it is second order. A comment in the transfer class calls
    it accurate for constants only, which undersells it.
    """
    errors = []
    for nvars_coarse in [5, 9, 17, 33]:
        transfer, fine, coarse = get_transfer(nvars_coarse=nvars_coarse)
        prolonged = transfer.prolong(sine(coarse))
        errors.append(np.max(np.abs(prolonged.getArray() - sine(fine).getArray())))

    errors = np.array(errors)
    order = np.log(errors[:-1] / errors[1:]) / np.log(2)
    assert np.isclose(np.median(order), 2, atol=0.3), f'Expected order 2, got {np.median(order):.2f}'


def get_grayscott_transfer(nvars_coarse=8, ratio=2):
    """
    The same transfer between two Gray-Scott grids. Its `dtype_f` is `petsc_vec_comp2`, which the
    heat problem above never produces.

    Returns:
        tuple: the transfer object, the fine problem and the coarse problem
    """
    from pySDC.implementations.problem_classes.GrayScott_2D_PETSc_periodic import petsc_grayscott_multiimplicit
    from pySDC.implementations.transfer_classes.TransferPETScDMDA import mesh_to_mesh_petsc_dmda

    params = {'Du': 1.0, 'Dv': 0.01, 'A': 0.09, 'B': 0.086}
    fine = petsc_grayscott_multiimplicit(nvars=[nvars_coarse * ratio] * 2, **params)
    coarse = petsc_grayscott_multiimplicit(nvars=[nvars_coarse] * 2, **params)
    return mesh_to_mesh_petsc_dmda(fine, coarse, {}), fine, coarse


@pytest.mark.petsc
@pytest.mark.parametrize('datatype', ['petsc_vec_imex', 'petsc_vec_comp2'])
@pytest.mark.parametrize('direction', ['restrict', 'prolong'])
def test_petsc_multiple_components(datatype, direction):
    """
    Each component has to go through the operator on its own, exactly as a single vector would.

    Note that the transfer builds its output from the problem's `dtype_f` rather than from the type
    it was handed, so each datatype has to be tested against a problem that actually produces it.
    """
    from pySDC.implementations.datatype_classes import petsc_vec as dt

    if datatype == 'petsc_vec_imex':
        transfer, fine, coarse = get_transfer()
        components = ['impl', 'expl']
    else:
        transfer, fine, coarse = get_grayscott_transfer()
        components = ['comp1', 'comp2']

    cls = getattr(dt, datatype)
    source_prob = fine if direction == 'restrict' else coarse
    assert source_prob.dtype_f is cls, 'Picked a problem that does not produce this datatype'

    source = cls(source_prob.init)
    single = source_prob.dtype_u(source_prob.init)
    single.getArray()[:] = np.random.default_rng(seed=4).random(single.getArray().shape)
    for factor, component in enumerate(components, start=1):
        getattr(source, component).getArray()[:] = factor * single.getArray()

    transferred = getattr(transfer, direction)(source)
    assert type(transferred) is cls, f'Expected {datatype} back, got {type(transferred).__name__}'

    reference = getattr(transfer, direction)(single)
    for factor, component in enumerate(components, start=1):
        assert np.allclose(
            getattr(transferred, component).getArray(), factor * reference.getArray()
        ), f'Component {component!r} was not treated like a single vector'


@pytest.mark.petsc
@pytest.mark.parametrize('direction', ['restrict', 'prolong'])
def test_petsc_rejects_unknown_types(direction):
    """Anything that is not a PETSc datatype has to be refused."""
    from pySDC.core.errors import TransferError

    transfer, _, _ = get_transfer()
    with pytest.raises(TransferError):
        getattr(transfer, direction)(np.zeros(3))


if __name__ == '__main__':
    test_petsc_prolongation_order()
