import pytest


class _Marker:
    def __init__(self, name):
        self.name = name


def _item(conftest, name, markers):
    class _Item:
        nodeid = f'pySDC/tests/{name}.py::test_it'
        path = conftest.HERE / f'{name}.py'

        def iter_markers(self):
            return [_Marker(me) for me in markers]

    return _Item()


@pytest.mark.base
def test_unmarked_test_is_an_error():
    """A test that carries none of the markers CI selects would never run, so collection has to stop."""
    from pySDC.tests import conftest

    with pytest.raises(pytest.exit.Exception, match='no CI job runs them') as excinfo:
        conftest.pytest_collection_modifyitems(None, [_item(conftest, 'test_plain', ['parametrize'])])
    assert excinfo.value.returncode == 4


@pytest.mark.base
@pytest.mark.parametrize('marker', ['base', 'mpi4py', 'cupy', 'benchmark'])
def test_marked_test_is_accepted(marker):
    from pySDC.tests import conftest

    conftest.pytest_collection_modifyitems(None, [_item(conftest, 'test_marked', [marker, 'parametrize'])])


@pytest.mark.base
def test_tests_outside_the_tree_are_not_checked():
    """Project tests are run whole, without `-m`, so they need no marker."""
    from pySDC.tests import conftest

    class _Outside:
        nodeid = 'pySDC/projects/Some/tests/test_it.py::test_it'
        path = conftest.HERE.parent / 'projects' / 'Some' / 'tests' / 'test_it.py'

        def iter_markers(self):
            return []

    conftest.pytest_collection_modifyitems(None, [_Outside()])
