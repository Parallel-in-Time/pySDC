"""
The project's figures, drawn from the current code and checked to still show what they claim.

They land in ``data/``, from where the README and the website pick them up.
"""

import os

import pytest


@pytest.mark.base
def test_mixed_precision_figures():
    """A normalised half-precision solve keeps the fp64 floor; the naive one, its control, does not."""
    from pySDC.projects.DeltaSDC.plot_mixed_precision import main

    for name, curves in main().items():
        floor = {label: min(history) for label, history in curves.items()}
        assert floor['fp16 solve, normalized'] < 10 * floor['fp64 solve'], name
        assert floor['fp16 solve, naive'] > 1e4 * floor['fp64 solve'], f'{name}: the control no longer fails'
        assert os.path.isfile(f'data/mixed_precision_{name}.png')


@pytest.mark.base
def test_delivered_accuracy_figure():
    """The iteration count is flat down to about five digits, then rises; half precision costs about one."""
    from pySDC.projects.DeltaSDC.plot_delivered_accuracy import main

    result = main()
    counts = dict(zip(result['etas'], result['iterations'], strict=True))
    assert all(count == result['exact'] for eta, count in counts.items() if eta <= 1e-5)
    assert counts[1e-2] > result['exact'], 'a two-digit solve should cost iterations'
    assert result['float16'] <= result['exact'] + 2
    assert os.path.isfile('data/delivered_accuracy.png')
