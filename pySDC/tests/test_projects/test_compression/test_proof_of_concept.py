import pytest


@pytest.mark.libpressio
@pytest.mark.parametrize("thresh", [1e-6, 1e-8, 1e-10])
def test_compression_proof_of_concept(thresh):
    import matplotlib.pyplot as plt
    from pySDC.projects.compression.order import plot_order_in_time

    fig, ax = plt.subplots()
    plot_order_in_time(ax, thresh)
