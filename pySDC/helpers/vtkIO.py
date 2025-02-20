#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper functions for VTK files IO (to be used with Paraview or PyVista)
"""
import os
import vtk
from vtkmodules.util import numpy_support
import numpy as np


def writeToVTR(fileName: str, data, coords, varNames):
    """
    Write a data array containing variables from a 3D rectilinear grid into a VTR file.

    Parameters
    ----------
    fileName : str
        Name of the VTR file, can be with or without the .vtr extension.
    data : np.4darray
        Array containing all the variables with [nVar, nX, nY, nZ] shape.
    coords : list[np.1darray]
        Coordinates in each direction.
    varNames : list[str]
        Variable names.

    Returns
    -------
    fileName : str
        Name of the VTR file.
    """
    data = np.asarray(data)
    nVar, *gridSizes = data.shape

    assert len(gridSizes) == 3, "function can be used only for 3D grid data"
    assert nVar == len(varNames), "varNames must have as many variable as data"
    assert [len(np.ravel(coord)) for coord in coords] == gridSizes, "coordinate size incompatible with data shape"

    nX, nY, nZ = gridSizes
    vtr = vtk.vtkRectilinearGrid()
    vtr.SetDimensions(nX, nY, nZ)

    def vect(x):
        return numpy_support.numpy_to_vtk(num_array=x, deep=True, array_type=vtk.VTK_FLOAT)

    x, y, z = coords
    vtr.SetXCoordinates(vect(x))
    vtr.SetYCoordinates(vect(y))
    vtr.SetZCoordinates(vect(z))

    def field(u):
        return numpy_support.numpy_to_vtk(num_array=u.ravel(order='F'), deep=True, array_type=vtk.VTK_FLOAT)

    pointData = vtr.GetPointData()
    for name, u in zip(varNames, data):
        uVTK = field(u)
        uVTK.SetName(name)
        pointData.AddArray(uVTK)

    writer = vtk.vtkXMLRectilinearGridWriter()
    if not fileName.endswith(".vtr"):
        fileName += ".vtr"
    writer.SetFileName(fileName)
    writer.SetInputData(vtr)
    writer.Write()

    return fileName


def readFromVTR(fileName: str):
    """
    Read a VTR file into a numpy 4darray

    Parameters
    ----------
    fileName : str
        Name of the VTR file, can be with or without the .vtr extension.

    Returns
    -------
    data : np.4darray
        Array containing all the variables with [nVar, nX, nY, nZ] shape.
    coords : list[np.1darray]
        Coordinates in each direction.
    varNames : list[str]
        Variable names.
    """
    if not fileName.endswith(".vtr"):
        fileName += ".vtr"
    assert os.path.isfile(fileName), f"{fileName} does not exist"

    reader = vtk.vtkXMLRectilinearGridReader()
    reader.SetFileName(fileName)
    reader.Update()

    vtr = reader.GetOutput()
    gridSizes = vtr.GetDimensions()
    assert len(gridSizes) == 3, "can only read 3D data"

    def vect(x):
        return numpy_support.vtk_to_numpy(x)

    coords = [vect(vtr.GetXCoordinates()), vect(vtr.GetYCoordinates()), vect(vtr.GetZCoordinates())]
    pointData = vtr.GetPointData()
    varNames = [pointData.GetArrayName(i) for i in range(pointData.GetNumberOfArrays())]
    data = [numpy_support.vtk_to_numpy(pointData.GetArray(name)).reshape(gridSizes, order="F") for name in varNames]
    data = np.array(data)

    return data, coords, varNames
