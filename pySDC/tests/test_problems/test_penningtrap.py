import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

# Computes the particle interactions in a fresh interpreter, where `None` in sys.modules makes `import numba` fail
WITHOUT_NUMBA = """
import json, sys
import numpy as np
sys.modules['numba'] = None
from pySDC.implementations.problem_classes.PenningTrap_3D import penningtrap
pos = np.array(json.loads(sys.argv[1]))
print(json.dumps(penningtrap.fast_interactions(pos.shape[1], pos, 0.1, np.ones(pos.shape[1])).tolist()))
"""


@pytest.mark.base
def test_interactions_without_numba():
    from pySDC.implementations.problem_classes.PenningTrap_3D import penningtrap

    pos = np.random.default_rng(0).random((3, 5))
    compiled = penningtrap.fast_interactions(5, pos, 0.1, np.ones(5))

    run = subprocess.run(
        [sys.executable, '-c', WITHOUT_NUMBA, json.dumps(pos.tolist())],
        capture_output=True,
        text=True,
        check=True,
        cwd=Path(__file__).parents[3],
    )
    # The last line: in CI, a sitecustomize.py that sets up coverage announces itself on stdout first
    assert np.allclose(json.loads(run.stdout.splitlines()[-1]), compiled, rtol=1e-12)
