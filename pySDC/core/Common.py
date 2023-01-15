#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description
-----------

Module containing utility classe(s) from which inherit some of the pySDC base
classes.
"""
from pySDC.core.Errors import ReadOnlyError


class RegisterParams(object):
    """Base class to register parameters"""

    def _register(self, *names, readOnly=False):
        """
        Register class attributes as

        Parameters
        ----------
        *names : TYPE
            DESCRIPTION.
        readOnly : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        """
        if readOnly:
            if not hasattr(self, '_readOnly'):
                self._readOnly = set()
            self._readOnly = self._readOnly.union(names)
        else:
            if not hasattr(self, '_parNames'):
                self._parNames = set()
            self._parNames = self._parNames.union(names)

    @property
    def params(self):
        return {name: getattr(self, name) for name in self._readOnly.union(self._parNames)}

    def __setattr__(self, name, value):
        try:
            if name in self._readOnly:
                raise ReadOnlyError(name)
        except AttributeError:
            pass
        super().__setattr__(name, value)
