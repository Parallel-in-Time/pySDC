import pytest


@pytest.mark.base
def test_main():
    from pySDC.projects.parallelSDC.preconditioner_playground import main, plot_iterations

    main()
    plot_iterations()
