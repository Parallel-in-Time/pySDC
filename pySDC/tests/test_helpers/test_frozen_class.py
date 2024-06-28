import pytest


@pytest.mark.base
def test_frozen_class():
    from pySDC.helpers.pysdc_helper import FrozenClass

    class Dummy(FrozenClass):
        pass

    me = Dummy()
    me.add_attr('foo')

    me.foo = 0

    you = Dummy()
    you.foo = 1
    assert me.foo != you.foo, 'Attribute is shared between class instances'

    me = Dummy()
    assert me.foo is None, 'Attribute persists after reinstantiation'

    me.add_attr('foo')
    assert len(Dummy.attrs) == 1, 'Attribute was added too many times'


if __name__ == '__main__':
    test_frozen_class()
