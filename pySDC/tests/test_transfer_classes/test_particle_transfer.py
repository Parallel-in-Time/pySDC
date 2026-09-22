import numpy as np
import pytest

INIT = ((3, 5), None, np.dtype('float64'))


def get_transfer():
    """`particles_to_particles` keeps the resolution, so it needs nothing from the problems."""
    from pySDC.implementations.transfer_classes.TransferParticles_NoCoarse import particles_to_particles

    return particles_to_particles(fine_prob=None, coarse_prob=None, params={})


def arrays_of(data):
    """The arrays a particle datatype carries, by name. An `acceleration` is itself the array."""
    names = [name for name in ['pos', 'vel', 'q', 'm', 'elec', 'magn'] if hasattr(data, name)]
    return {name: getattr(data, name) for name in names} or {'values': data}


@pytest.mark.base
@pytest.mark.parametrize('datatype', ['particles', 'fields', 'acceleration'])
@pytest.mark.parametrize('direction', ['restrict', 'prolong'])
def test_particles_to_particles(datatype, direction):
    """
    The transfer does not coarsen, so it has to hand back the same values in the same type, and it
    has to hand back a copy: writing to the result must not reach into the level it came from.
    """
    from pySDC.implementations.datatype_classes import particles as dt

    cls = getattr(dt, datatype)
    source = cls(INIT)
    rng = np.random.default_rng(seed=7)
    for array in arrays_of(source).values():
        array[:] = rng.random(array.shape)

    transferred = getattr(get_transfer(), direction)(source)
    assert type(transferred) is cls, f'Expected {cls.__name__} back, got {type(transferred).__name__}'

    before = {name: np.array(array) for name, array in arrays_of(source).items()}
    for name, array in arrays_of(transferred).items():
        assert np.allclose(array, before[name]), f'{name!r} did not survive the transfer'
        # note that `array += 1` would not stick: `mesh` drops `out` in its `__array_ufunc__`
        array[:] = array + 1.0

    for name, array in arrays_of(source).items():
        assert np.allclose(array, before[name]), f'Writing to the result changed {name!r} on the source'


@pytest.mark.base
@pytest.mark.parametrize('direction', ['restrict', 'prolong'])
def test_particles_to_particles_rejects_unknown_types(direction):
    """Anything that is not a particle datatype has to be refused rather than silently copied."""
    from pySDC.core.errors import TransferError

    with pytest.raises(TransferError):
        getattr(get_transfer(), direction)(np.zeros(3))


if __name__ == '__main__':
    test_particles_to_particles('particles', 'restrict')
