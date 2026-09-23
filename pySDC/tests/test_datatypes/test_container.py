import pytest


def make_pair_class():
    """A multi-component container over the smallest component type that can stand in for a mesh."""
    from pySDC.implementations.datatype_classes.container import MultiComponentContainer
    from pySDC.core.errors import DataError

    class Value(object):
        def __init__(self, init, val=0.0):
            self.val = init.val if isinstance(init, Value) else float(val)

        def __add__(self, other):
            return Value(None, self.val + other.val)

        def __sub__(self, other):
            return Value(None, self.val - other.val)

        def __rmul__(self, other):
            if not isinstance(other, (int, float)):
                raise DataError(f'cannot multiply {type(other)} with {type(self)}')
            return Value(None, other * self.val)

    class Pair(MultiComponentContainer):
        components = ['a', 'b']
        component_type = Value

    return Pair


@pytest.mark.base
def test_components_are_built_and_copied():
    Pair = make_pair_class()

    x = Pair(None, val=1.0)
    assert x.a.val == 1.0 and x.b.val == 1.0

    x.b.val = 2.0
    y = Pair(x)
    assert y.a.val == 1.0 and y.b.val == 2.0

    # a copy has to be a copy, not a second name for the same components
    y.a.val = 9.0
    assert x.a.val == 1.0


@pytest.mark.base
def test_arithmetic_is_component_wise():
    Pair = make_pair_class()

    x = Pair(None, val=1.0)
    y = Pair(None, val=2.0)

    assert (x + y).a.val == 3.0
    assert (y - x).b.val == 1.0
    assert (3.0 * x).a.val == 3.0

    # and leaves the operands alone
    assert x.a.val == 1.0 and y.a.val == 2.0


@pytest.mark.base
def test_arithmetic_with_a_foreign_type():
    from pySDC.core.errors import DataError

    Pair = make_pair_class()
    x = Pair(None, val=1.0)

    for operation in [lambda: x + 1.0, lambda: x - 'Dat kölsche Dom']:
        with pytest.raises(DataError):
            operation()

    # scaling is the component type's business, so it is the one that refuses
    with pytest.raises(DataError):
        'Dat kölsche Dom' * x
