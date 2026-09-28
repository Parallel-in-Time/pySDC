#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 20:14:06 2024
"""

import os
import warnings
import pytest

from pySDC.helpers.testing import DataChecker


@pytest.mark.base
def test_DataChecker(tmp_path):
    result = [1, 2, 3, 4]
    correct = result
    wrong = [1, 2, 3, 3]

    # DataChecker keeps its json files next to the path it is given: a fresh directory, so that the
    # test neither finds a stale reference nor leaves files in the source tree
    script = str(tmp_path / 'script.py')
    d1 = DataChecker(script)

    warnings.filterwarnings("error")
    try:
        d1.storeAndCheck('r1', result)
    except UserWarning:
        pass
    else:
        raise AssertionError("no reference data does not raise warning")
    d1.writeToJSON()
    os.rename(d1._dataFile, d1._dataRefFile)

    d2 = DataChecker(script)
    try:
        d2.storeAndCheck('r1', result)
    except UserWarning:
        raise AssertionError("warning raised with reference data available") from None
    d2.writeToJSON()
    warnings.resetwarnings()

    d3 = DataChecker(script)

    try:
        d3.storeAndCheck('r1', wrong)
    except AssertionError:
        pass
    else:
        raise AssertionError("wrong data does not raise assertion error")

    try:
        d3.storeAndCheck('r2', correct)
    except AssertionError:
        pass
    else:
        raise AssertionError("wrong key does not raise assertion error")

    try:
        d3.storeAndCheck('r1', correct[:-1])
    except AssertionError:
        pass
    else:
        raise AssertionError("data with incorrect size does not raise assertion error")
