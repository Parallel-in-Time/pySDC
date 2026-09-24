import pytest


@pytest.mark.benchmark
def test_benchmark_collocation(benchmark):
    """Time the collocation tests of `pySDC/tests/test_collocation.py` over all their parameters."""
    from pySDC.tests.test_collocation import (
        NODE_TYPES,
        QUAD_TYPES,
        test_canintegratepolynomials,
        test_partialquadraturewithQ,
        test_partialquadraturewithS,
        test_relateQandSmat,
    )

    def wrapper():
        for num_nodes in range(2, 13):
            for node_type in NODE_TYPES:
                for quad_type in QUAD_TYPES:
                    test_canintegratepolynomials(num_nodes, node_type, quad_type)
                    test_relateQandSmat(num_nodes, node_type, quad_type)
                    test_partialquadraturewithQ(num_nodes, node_type, quad_type)
                    test_partialquadraturewithS(num_nodes, node_type, quad_type)

    benchmark(wrapper)
