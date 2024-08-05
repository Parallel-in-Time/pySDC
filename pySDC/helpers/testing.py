#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 14:41:43 2024

Helpers module for testing utilities
"""
import os
import json
import warnings
import numpy as np


class DataChecker:
    """
    Object allowing to quickly generate and check generated data from scripts.

    Here is an example of use for one given script:

    >>> from pySDC.helpers.testing import DataChecker
    >>>
    >>> data = DataChecker(__file__)
    >>> # ... some computations ...
    >>> data.storeAndCheck('solution', solution)
    >>> # ... some other computations ...
    >>> data.storeAndCheck('errors', errors)
    >>> # ... ploting some figures
    >>> data.writeToJSON() # end of script

    The `storeAndCheck` method use a unique key as first argument, and the
    data as second argument, into list format.
    Calling this method will store the data into a cache variable, and compare
    with reference data stored in `_dataRef.json` (if it exists, see below ...).
    Finally, the `writeToJSON` method saves the cache variable into a json
    file `_data.json`.

    When there is no `_dataRef.json` in the script directory,
    executing the script output some warning telling that there is no
    reference data to compare with.
    To remove those warnings (and properly test data), just rename the
    `_data.json` into `_dataRef.json`.
    Then re-running the script will then compare the newly generated data with
    the reference data stored into `_dataRef.json`, and raise an error if there
    is some differences.

    Important
    ---------

    - using the `__file__` built-in variable is necessary to save data in the
      script directory, independently of the working directory.
    - several script in the same directory can use the DataChecker, which
      implies that the key provided to `storeAndCheck` must be unique across
      all directory scripts. If not, the same key from previous directory will
      simply be overwritten in the cached variable and the `_data.ref` file.
    """

    def __init__(self, filePath):
        path = '/' + os.path.join(*filePath.split('/')[:-1])
        self._data = {}  # cache for data
        self._dataRef = None  # cache for reference data
        self._dataFile = os.path.join(path, '_data.json')
        self._dataRefFile = os.path.join(path, '_dataRef.json')

    def storeAndCheck(self, key, data, rtol=1e-5, atol=1e-8):
        """
        Store data into cache, and check with eventual reference data

        Parameters
        ----------
        key : str
            Unique key (project wide) for the data.
        data : list or array-like
            The data that has to be stored.
        rtol : float
            Relative tolerance
        atol : float
            Absolute tolerance
        """
        self._data[key] = list(data)
        if self._dataRef is None:
            try:
                with open(self._dataRefFile, "r") as f:
                    self._dataRef = json.load(f)
            except FileNotFoundError:
                warnings.warn(f"no reference data to check key:{key}")
                return  # pragma: no cover

        assert key in self._dataRef, f"key:{key} not in reference data"

        data = self._data[key]
        ref = self._dataRef[key]

        assert len(data) == len(ref), f"data with len:{len(data)}) cannot be compared to ref with len:{len(ref)})"
        assert np.allclose(
            data, ref, atol=atol, rtol=rtol, equal_nan=True
        ), f"{key}, difference between data:{data} and ref:{ref}"

    def writeToJSON(self):
        """Write cached data into a json file"""
        if os.path.isfile(self._dataFile):
            with open(self._dataFile, "r") as f:
                self._data.update(json.load(f))
        with open(self._dataFile, "w") as f:
            json.dump(self._data, f)
