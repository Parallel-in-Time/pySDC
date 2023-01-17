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
    """
    Base class to register parameters.

    Attributes
    ----------
    params : dict (property)
        Dictionnary containing names and values of registered parameters.
    _parNames : set of str
        Names of all the registered parameters.
    _parNamesReadOnly : set of str
        Names of all the parameters registered as read-only.
    """

    def _makeAttributeAndRegister(self, *names, localVars=None, readOnly=False):
        """
        Register a list of attribute name as parameters of the class.

        Parameters
        ----------
        *names : list of str
            The name of the parameters to be registered (should be class attributes).
        localVars : dict
            Dictionnary containing key=names and values=paramValues for each
            parNames given in names. Can be provided, for instance, using
            `locals()` built-in dictionary. MUST BE provided as soon as
            names contains anything.
        readOnly : bool, optional
            Wether or not store the parameters as read-only attributes
        """
        if len(names) > 1 and localVars is None:
            raise ValueError("a dictionnary must be provided in localVars with parameters values")
        # Set parameters as attributes
        for name in names:
            try:
                super().__setattr__(name, localVars[name])
            except KeyError:
                raise ValueError(f'value for {name} not given in localVars')
        # Register as class parameter
        if readOnly:
            self._parNamesReadOnly = self._parNamesReadOnly.union(names)
        else:
            self._parNames = self._parNames.union(names)

    @property
    def params(self):
        """Dictionnary containing names and values of registered parameters"""
        return {name: getattr(self, name) for name in self._parNamesReadOnly.union(self._parNames)}

    def __setattr__(self, name, value):
        if name in self._parNamesReadOnly:
            raise ReadOnlyError(name)
        super().__setattr__(name, value)
