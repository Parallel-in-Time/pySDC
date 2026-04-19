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
    assert Dummy.attrs.count('foo') == 1, 'Attribute was added too many times'

    class Dummy2(FrozenClass):
        pass

    Dummy2.add_attr('bar')
    you = Dummy2()
    you.bar = 5
    # With the fix, 'bar' should NOT be accessible on Dummy instances at all
    # because it was only added to Dummy2, not Dummy
    with pytest.raises(AttributeError):  # bar is not an attribute of Dummy
        _ = me.bar


@pytest.mark.base
def test_frozen_class_isolation():
    """
    Test that attributes added to one frozen class don't leak to other frozen classes.
    This specifically tests the fix for the issue where attrs was shared across all subclasses.
    """
    from pySDC.helpers.pysdc_helper import FrozenClass

    # Simulate Level and Step status classes from the issue
    class LevelStatus(FrozenClass):
        def __init__(self):
            self.residual = None
            self._freeze()

    class StepStatus(FrozenClass):
        def __init__(self):
            self.iter = None
            self._freeze()

    # Add attribute to LevelStatus only
    LevelStatus.add_attr('error_embedded_estimate')

    # Verify attrs are separate
    assert 'error_embedded_estimate' in LevelStatus.attrs
    assert 'error_embedded_estimate' not in StepStatus.attrs

    level_status = LevelStatus()
    step_status = StepStatus()

    # Should work - error_embedded_estimate was added to LevelStatus
    level_status.error_embedded_estimate = 0.01
    assert level_status.error_embedded_estimate == 0.01

    # Should fail - error_embedded_estimate was NOT added to StepStatus
    with pytest.raises(TypeError, match="is a frozen class, cannot add attribute"):
        step_status.error_embedded_estimate = 0.02


if __name__ == '__main__':
    test_frozen_class()
    test_frozen_class_isolation()
    print("All tests passed!")
