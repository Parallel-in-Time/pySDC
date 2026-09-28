import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]


@pytest.mark.base
def test_readme_cites_the_preferred_citation():
    """The README's "How to cite" is written by hand; CITATION.cff, which everything else is made from, is not"""
    cff = (ROOT / 'CITATION.cff').read_text(encoding='utf-8')
    preferred = cff[cff.index('preferred-citation:') :]
    doi = re.search(r'^\s+doi:\s*(\S+)', preferred, re.M).group(1)
    readme = (ROOT / 'README.md').read_text(encoding='utf-8')
    assert f'https://doi.org/{doi}' in readme, f'README.md does not cite {doi}, the preferred citation of CITATION.cff'
