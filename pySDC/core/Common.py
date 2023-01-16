#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description
-----------

Module containing utility classe(s) from which inherit some of the pySDC base
classes.
"""
from pySDC.core.Errors import ReadOnlyError


class _MetaRegisterParams(type):
    """Metaclass for RegisterParams base class"""

    def __new__(cls, name, bases, dct):
        obj = super().__new__(cls, name, bases, dct)
        obj._parNamesReadOnly = set()
        obj._parNames = set()
        return obj


class RegisterParams(metaclass=_MetaRegisterParams):
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
            self._parNamesReadOnly = self._parNamesReadOnly.union(names)
        else:
            self._parNames = self._parNames.union(names)

    @property
    def params(self):
        return {name: getattr(self, name) for name in self._parNamesReadOnly.union(self._parNames)}

    def __setattr__(self, name, value):
        if name in self._parNamesReadOnly:
            raise ReadOnlyError(name)
        super().__setattr__(name, value)
