"""
The NumPy and CuPy datatypes have to behave the same way.

Every CPU/GPU bug found in this project so far has been CuPy quietly differing from NumPy on
something nobody checked: `copy` returning the base class rather than the subclass, `asarray`
refusing a conversion NumPy performs, `zeros_like` dropping the subclass, an argument renamed in
one library and not the other. Each one passed every CPU test and failed only on hardware.

This builds the same profile of both datatypes -- for each operation, whether the result is still
a datatype and whether it still carries its communicator -- and requires the two to agree.
Divergences that are real and accepted are listed in `KNOWN_DIVERGENCES`, so that they are
declared rather than discovered, and so that a future CuPy fixing one makes this fail and get
tidied up.
"""

import copy as copy_module

import pytest

SENTINEL = 'a communicator would live here'


def profile(datatype, xp):
    """What each operation on a datatype gives back: still a datatype? still carrying the comm?"""

    def fresh():
        # every operation gets its own instance, so that an in-place one cannot colour the next
        me = datatype(init=((8,), None, xp.dtype('float64')))
        me[:] = 1.0
        me.comm = SENTINEL
        return me

    operations = {
        'slice': lambda a: a[:4],
        'whole slice': lambda a: a[:],
        'add scalar': lambda a: a + 1.0,
        'multiply': lambda a: 2.0 * a,
        'subtract self': lambda a: a - a,
        'in-place add': lambda a: a.__iadd__(0.0),
        'copy': lambda a: a.copy(),
        'copy-construct': lambda a: type(a)(a),
        'deepcopy': copy_module.deepcopy,
        'view as itself': lambda a: a.view(type(a)),
        'reshape': lambda a: a.reshape((2, 4)),
        'ravel': lambda a: a.ravel(),
        'flatten': lambda a: a.flatten(),
        'astype': lambda a: a.astype('float64'),
        'zeros_like': xp.zeros_like,
        'sum (reduction)': lambda a: a.sum(),
        'comparison': lambda a: a > 0.5,
    }

    described = {}
    for name, operation in operations.items():
        result = operation(fresh())
        described[name] = (
            isinstance(result, datatype),
            getattr(result, 'comm', None) == SENTINEL,
        )
    return described


#: Where the two genuinely differ, with the reason. Each entry is a promise that the difference is
#: understood, not that it is welcome.
KNOWN_DIVERGENCES = {
    # `cupy.zeros_like` has no `subok` -- it raises `subok is not supported yet` -- so it cannot be
    # asked to keep the subclass the way `numpy.zeros_like` does by default.
    'zeros_like': 'CuPy has no subok, so the subclass is lost',
    # `cupy.ndarray.astype` ignores `subok` for the same reason. Nothing in pySDC calls `astype` on
    # a datatype -- the calls there are on sparse operators and plain arrays -- so this costs
    # nothing today, but it would bite whoever writes the first one.
    'astype': 'CuPy has no subok, so the subclass is lost',
}


@pytest.mark.cupy
def test_the_two_datatypes_behave_the_same():
    """Build both profiles in one process and require them to match."""
    import cupy as cp
    import numpy as np

    from pySDC.implementations.datatype_classes.cupy_mesh import cupy_mesh
    from pySDC.implementations.datatype_classes.mesh import mesh

    on_CPU = profile(mesh, np)
    on_GPU = profile(cupy_mesh, cp)

    assert set(on_CPU) == set(on_GPU), 'the two profiles cover different operations'
    assert set(KNOWN_DIVERGENCES) <= set(on_CPU), 'a known divergence names no operation'

    # an anchor, so that the comparison below cannot pass by both backends being broken alike
    for name in ['slice', 'copy', 'subtract self']:
        assert on_CPU[name] == (True, True), f'the NumPy datatype no longer survives {name}'

    differing = {
        name: {'numpy': on_CPU[name], 'cupy': on_GPU[name]}
        for name in on_CPU
        if on_CPU[name] != on_GPU[name] and name not in KNOWN_DIVERGENCES
    }
    assert not differing, (
        'CuPy and NumPy disagree on (is still a datatype, still carries the communicator). '
        f'Either fix the CuPy side or record it in KNOWN_DIVERGENCES with a reason: {differing}'
    )


@pytest.mark.cupy
def test_the_known_divergences_are_still_real():
    """If CuPy ever fixes one of these, this fails and the entry should go."""
    import cupy as cp
    import numpy as np

    from pySDC.implementations.datatype_classes.cupy_mesh import cupy_mesh
    from pySDC.implementations.datatype_classes.mesh import mesh

    on_CPU = mesh(init=((8,), None, np.dtype('float64')))
    on_GPU = cupy_mesh(init=((8,), None, cp.dtype('float64')))

    assert isinstance(np.zeros_like(on_CPU), mesh), 'NumPy stopped keeping the subclass'
    assert not isinstance(cp.zeros_like(on_GPU), cupy_mesh), (
        'CuPy now keeps the subclass through `zeros_like`: drop the entry from KNOWN_DIVERGENCES '
        'and let the contract test cover it'
    )
    assert isinstance(on_CPU.astype('float32'), mesh), 'NumPy stopped keeping the subclass'
    assert not isinstance(
        on_GPU.astype('float32'), cupy_mesh
    ), 'CuPy now keeps the subclass through `astype`: drop the entry from KNOWN_DIVERGENCES'

    with pytest.raises(TypeError):
        # CuPy refuses the implicit conversion NumPy performs; this is what broke #708
        np.asarray(on_GPU)
