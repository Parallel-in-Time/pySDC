import numpy as np
import pytest

# The mesh-based space transfer classes that run on plain NumPy. `no_coarse` keeps the resolution,
# all others coarsen by a factor of two.
MESH_TRANSFER_CLASSES = ['mesh_to_mesh', 'mesh_to_mesh_fft', 'mesh_to_mesh_fft2d', 'no_coarse']


def get_problem(nvars):
    """
    Minimal stand-in for a pySDC problem. All the mesh transfer classes need from a problem is
    ``nvars`` and ``init``.

    Args:
        nvars (int or tuple): number of degrees of freedom

    Returns:
        Instance of a pySDC problem class
    """
    from pySDC.core.problem import Problem
    from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh

    class DummyProblem(Problem):
        dtype_u = mesh
        dtype_f = imex_mesh

        def __init__(self, nvars):
            super().__init__(init=(nvars, None, np.dtype('float64')))
            self._makeAttributeAndRegister('nvars', localVars=locals())

    return DummyProblem(nvars)


def get_transfer(name, iorder=4, rorder=4):
    """
    Set up a transfer class of the given name between two dummy problems.

    Returns:
        tuple: the transfer object, the fine problem and the coarse problem
    """
    if name == 'mesh_to_mesh':
        from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh as cls

        nvars_fine, nvars_coarse = 32, 16
    elif name == 'mesh_to_mesh_fft':
        from pySDC.implementations.transfer_classes.TransferMesh_FFT import mesh_to_mesh_fft as cls

        nvars_fine, nvars_coarse = 32, 16
    elif name == 'mesh_to_mesh_fft2d':
        from pySDC.implementations.transfer_classes.TransferMesh_FFT2D import mesh_to_mesh_fft2d as cls

        nvars_fine, nvars_coarse = (32, 32), (16, 16)
    elif name == 'no_coarse':
        from pySDC.implementations.transfer_classes.TransferMesh_NoCoarse import mesh_to_mesh as cls

        nvars_fine, nvars_coarse = 32, 32
    else:
        raise NotImplementedError(f'Don\'t know how to set up transfer class {name!r}')

    fine, coarse = get_problem(nvars_fine), get_problem(nvars_coarse)
    return cls(fine, coarse, {'iorder': iorder, 'rorder': rorder, 'periodic': True}), fine, coarse


@pytest.mark.base
@pytest.mark.parametrize('name', MESH_TRANSFER_CLASSES)
@pytest.mark.parametrize('direction', ['restrict', 'prolong'])
def test_multicomponent_mesh(name, direction):
    """
    An `imex_mesh` is also an instance of `mesh`, so a transfer class that dispatches on
    `isinstance` silently takes the single-component branch and either crashes or mangles the data.
    Check that both components survive and that each one gets the same treatment as a plain mesh.
    """
    from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh

    transfer, fine, coarse = get_transfer(name)
    source, target = (fine, coarse) if direction == 'restrict' else (coarse, fine)
    transfer_func = getattr(transfer, direction)

    rng = np.random.default_rng(seed=3)
    u = imex_mesh(source.init)
    u.impl[:] = rng.random(u.impl.shape)
    u.expl[:] = rng.random(u.expl.shape)

    transferred = transfer_func(u)

    assert type(transferred) is imex_mesh, f'Expected an imex_mesh back, got {type(transferred)}'
    assert transferred.shape == imex_mesh(target.init).shape, 'Got unexpected shape'

    for component in imex_mesh.components:
        single = mesh(source.init)
        single[:] = getattr(u, component)
        assert np.allclose(
            getattr(transferred, component), transfer_func(single)
        ), f'Component {component!r} was not treated like a plain mesh'

    # a transfer that hits the wrong branch can still return the right shape by accident, so make
    # sure the two components did not get mixed up
    assert not np.allclose(transferred.impl, transferred.expl), 'The two components ended up identical'


@pytest.mark.base
@pytest.mark.parametrize('name', ['mesh_to_mesh_fft', 'mesh_to_mesh_fft2d'])
@pytest.mark.parametrize('ratio', [2, 4])
def test_spectral_transfer_is_exact(name, ratio):
    """
    Fourier interpolation of a function the coarse grid resolves is exact, all the way up to the
    coarse Nyquist mode. Injection back down then has to return the original coarse data.
    """
    from pySDC.implementations.datatype_classes.mesh import mesh

    nvars_coarse = 16
    nvars_fine = nvars_coarse * ratio
    ndim = 2 if name.endswith('2d') else 1

    if name == 'mesh_to_mesh_fft':
        from pySDC.implementations.transfer_classes.TransferMesh_FFT import mesh_to_mesh_fft as cls
    else:
        from pySDC.implementations.transfer_classes.TransferMesh_FFT2D import mesh_to_mesh_fft2d as cls

    shape_fine = nvars_fine if ndim == 1 else (nvars_fine,) * 2
    shape_coarse = nvars_coarse if ndim == 1 else (nvars_coarse,) * 2
    fine, coarse = get_problem(shape_fine), get_problem(shape_coarse)
    transfer = cls(fine, coarse, {})

    def wave(nvars, wavenumber):
        # a product of cosines, so that the Nyquist mode stays recoverable from its samples in 2d as
        # well: a tilted wave such as cos(k(x+y)) aliases with sin(kx)sin(ky) there
        x = np.cos(wavenumber * np.linspace(0, 2 * np.pi, nvars, endpoint=False))
        return x if ndim == 1 else np.outer(x, x)

    # include the coarse Nyquist mode: it is where hand-rolled zero padding of the spectrum goes wrong
    for wavenumber in [1, 3, nvars_coarse // 2]:
        u_coarse, u_fine = mesh(coarse.init), mesh(fine.init)
        u_coarse[:] = wave(nvars_coarse, wavenumber)
        u_fine[:] = wave(nvars_fine, wavenumber)

        assert abs(transfer.prolong(u_coarse) - u_fine) < 1e-12, f'Prolongation is not exact for k={wavenumber}'
        assert abs(transfer.restrict(u_fine) - u_coarse) < 1e-12, f'Restriction is not exact for k={wavenumber}'


@pytest.mark.base
@pytest.mark.parametrize('iorder', [2, 4, 8])
@pytest.mark.parametrize('ndim', [1, 2])
def test_mesh_to_mesh_operator_algebra(iorder, ndim):
    """
    Check the algebraic properties of the interpolation and restriction matrices: restriction is the
    scaled adjoint of interpolation, and both reproduce constants on a periodic grid.
    """
    from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh

    nvars_fine, nvars_coarse = (32, 16) if ndim == 1 else ((32, 32), (16, 16))
    fine, coarse = get_problem(nvars_fine), get_problem(nvars_coarse)
    transfer = mesh_to_mesh(fine, coarse, {'iorder': iorder, 'rorder': iorder, 'periodic': True})

    P = transfer.Pspace.toarray()
    R = transfer.Rspace.toarray()

    assert np.allclose(R, 0.5**ndim * P.T), 'Restriction is not the scaled adjoint of interpolation'
    assert np.allclose(P @ np.ones(P.shape[1]), 1.0), 'Interpolation does not reproduce a constant'
    assert np.allclose(R @ np.ones(R.shape[1]), 1.0), 'Restriction does not reproduce a constant'


@pytest.mark.base
def test_mesh_to_mesh_injection():
    """`rorder = 0` asks for plain injection instead of a weighted average."""
    from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh

    transfer = mesh_to_mesh(get_problem(32), get_problem(16), {'iorder': 4, 'rorder': 0, 'periodic': True})
    R = transfer.Rspace

    assert R.nnz == R.shape[0], 'Injection has to take exactly one fine value per coarse value'
    assert np.allclose(R.toarray() @ np.ones(R.shape[1]), 1.0), 'Injection does not reproduce a constant'


@pytest.mark.base
@pytest.mark.parametrize('ndim', [1, 2])
def test_mesh_to_mesh_identity_without_coarsening(ndim):
    """Equal resolution on both levels has to give the identity in both directions."""
    from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh

    nvars = 16 if ndim == 1 else (16, 16)
    prob = get_problem(nvars)
    transfer = mesh_to_mesh(prob, get_problem(nvars), {'iorder': 4, 'rorder': 4, 'periodic': True})

    size = np.prod(nvars)
    assert np.allclose(transfer.Pspace.toarray(), np.eye(size)), 'Interpolation is not the identity'
    assert np.allclose(transfer.Rspace.toarray(), np.eye(size)), 'Restriction is not the identity'


if __name__ == '__main__':
    test_multicomponent_mesh('mesh_to_mesh_fft2d', 'prolong')
