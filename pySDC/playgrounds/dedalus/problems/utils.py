#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities
"""
import os
import h5py
import glob
import random
import numpy as np
import scipy.optimize as sco
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pySDC.helpers.fieldsIO import Rectilinear
from qmat.nodes import NodesGenerator
from qmat.lagrange import LagrangeApproximation


def computeMeanSpectrum(uValues, xGrid=None, zGrid=None, verbose=False):
    """ uValues[nT, nVar, nX, (nY,) nZ] """
    uValues = np.asarray(uValues)
    nT, nVar, *gridSizes = uValues.shape
    dim = len(gridSizes)
    assert nVar == dim
    if verbose:
        print(f"Computing Mean Spectrum on u[{', '.join([str(n) for n in uValues.shape])}]")

    energy_spectrum = []
    if dim == 2:

        for i in range(2):
            u = uValues[:, i]                           # (nT, Nx, Nz)
            spectrum = np.fft.rfft(u, axis=-2)          # over Nx -->  #(nT, k, Nz)
            spectrum *= np.conj(spectrum)               # (nT, k, Nz)
            spectrum /= spectrum.shape[-2]              # normalize with Nx --> (nT, k, Nz)
            spectrum = np.mean(spectrum.real, axis=-1)  # mean over Nz --> (nT,k)
            energy_spectrum.append(spectrum)

    elif dim == 3:

        # Check for a cube with uniform dimensions
        nX, nY, nZ = gridSizes
        assert nX == nY
        size = nX // 2

        # Interpolate in z direction
        assert xGrid is not None and zGrid is not None
        if verbose: print(" -- interpolating from zGrid to a uniform mesh ...")
        from qmat.lagrange import LagrangeApproximation
        P = LagrangeApproximation(zGrid, weightComputation="STABLE").getInterpolationMatrix([0.1, 0.5, 0.9])
        uValues = (P @ uValues.reshape(-1, nZ).T).T.reshape(nT, dim, nX, nY, 3)

        # Compute 2D mode disks
        k1D = np.fft.fftfreq(nX, 1/nX)**2
        kMod = k1D[:, None] + k1D[None, :]
        kMod **= 0.5
        idx = kMod.copy()
        idx *= (kMod < size)
        idx -= (kMod >= size)

        idxList = range(int(idx.max()) + 1)
        flatIdx = idx.ravel()

        # Fourier transform and square of Im,Re
        if verbose: print(" -- 2D FFT on u, v & w ...")
        uHat = np.fft.fftn(uValues, axes=(-3, -2))

        if verbose: print(" -- square of Im,Re ...")
        ffts = [uHat[:, i] for i in range(nVar)]
        reParts = [uF.reshape((nT, nX*nY, 3)).real**2 for uF in ffts]
        imParts = [uF.reshape((nT, nX*nY, 3)).imag**2 for uF in ffts]

        # Spectrum computation
        if verbose: print(" -- computing spectrum ...")
        spectrum = np.zeros((nT, size, 3))
        for i in idxList:
            if verbose: print(f" -- k{i+1}/{len(idxList)}")
            kIdx = np.argwhere(flatIdx == i)
            tmp = np.empty((nT, *kIdx.shape, 3))
            for re, im in zip(reParts, imParts):
                np.copyto(tmp, re[:, kIdx])
                tmp += im[:, kIdx]
                spectrum[:, i] += tmp.sum(axis=(1, 2))
        spectrum /= 2*(nX*nY)**2

        energy_spectrum.append(spectrum)
        if verbose: print(" -- done !")

    return energy_spectrum


def getModes(grid):
    nX = np.size(grid)
    k = np.fft.rfftfreq(nX, 1/nX) + 0.5
    return k


def rbc3dInterpolation(coarseFields):
    """
    Interpolate a RBC3D field to twice its space resolution

    Parameters
    ----------
    coarseFields : np.4darray
        The fields values on the coarse grid, with shape [nV,nX,nY,nZ].
        The last dimension (z) uses a chebychev grid, while x and y are
        uniform periodic.

    Returns
    -------
    fineFields : np.4darray
        The interpolated fields, with shape [nV,2*nX,2*nY,2*nZ]
    """
    coarseFields = np.asarray(coarseFields)
    assert coarseFields.ndim == 4, "requires 4D array"

    nV, nX, nY, nZ = coarseFields.shape

    # Chebychev grids and interpolation matrix for z
    zC = NodesGenerator("CHEBY-1", "GAUSS").getNodes(nZ)
    zF = NodesGenerator("CHEBY-1", "GAUSS").getNodes(2*nZ)
    Pz = LagrangeApproximation(zC, weightComputation="STABLE").getInterpolationMatrix(zF)

    # Fourier interpolation in x and y
    print(" -- computing 2D FFT ...")
    uFFT = np.fft.fftshift(np.fft.fft2(coarseFields, axes=(1, 2)), axes=(1, 2))
    print(" -- padding in Fourier space ...")
    uPadded = np.zeros_like(uFFT, shape=(nV, 2*nX, 2*nY, nZ))
    uPadded[:, nX//2:-nX//2, nY//2:-nY//2] = uFFT
    print(" -- computing 2D IFFT ...")
    uXY = np.fft.ifft2(np.fft.ifftshift(uPadded, axes=(1, 2)), axes=(1, 2)).real*4

    # Polynomial interpolation in z
    print(" -- interpolating in z direction ...")
    fineFields = (Pz @ uXY.reshape(-1, nZ).T).T.reshape(nV, 2*nX, 2*nY, 2*nZ)

    return fineFields

def decomposeRange(iBeg, iEnd, step, maxSize):
    if iEnd is None:
        raise ValueError("need to provide iEnd for range decomposition")
    nIndices = len(range(iBeg, iEnd, step))
    subRanges = []

    # Iterate over the original range and create sub-ranges
    iStart = iBeg
    while nIndices > 0:
        iStop = iStart + (maxSize - 1) * step
        if step > 0 and iStop > iEnd:
            iStop = iEnd
        elif step < 0 and iStop < iEnd:
            iStop = iEnd

        subRanges.append((iStart, iStop + 1 - (iStop==iEnd), step))
        nIndices -= maxSize
        iStart = iStop + step if nIndices > 0 else iEnd

    return subRanges


class OutputFiles():
    """
    Object to load and manipulate hdf5 Dedalus generated solution output
    """
    def __init__(self, folder):
        self.folder = folder
        fileNames = glob.glob(f"{self.folder}/*.h5")
        fileNames.sort(key=lambda f: int(f.split("_s")[-1].split(".h5")[0]))
        self.files = fileNames
        self._file = None   # temporary buffer to store the HDF5 file
        self._iFile = None  # index of the HDF5 stored in the temporary buffer
        vData0 = self.file(0)['tasks']['velocity']
        self.x = np.array(vData0.dims[2]["x"])
        self.dim = dim = len(vData0.dims)-2
        if dim == 2:
            self.z = np.array(vData0.dims[3]["z"])
            self.y = self.z
        elif dim == 3:
            self.y = np.array(vData0.dims[3]["y"])
            self.z = np.array(vData0.dims[4]["z"])
        else:
            raise NotImplementedError(f"{dim = }")


    def file(self, iFile:int):
        if iFile != self._iFile:
            try:
                self._file.close()
            except: pass
            self._iFile = iFile
            self._file = h5py.File(self.files[iFile], mode='r')
        return self._file

    def __del__(self):
        try:
            self._file.close()
        except: pass

    @property
    def nFiles(self):
        return len(self.files)

    @property
    def nX(self):
        return self.x.size

    @property
    def nY(self):
        return self.y.size

    @property
    def nZ(self):
        return self.z.size

    @property
    def shape(self):
        if self.dim == 2:
            return (4, self.nX, self.nZ)
        elif self.dim == 3:
            return (4, self.nX, self.nY, self.nZ)

    @property
    def k(self):
        if self.dim == 2:
            return getModes(self.x)
        elif self.dim == 3:
            return getModes(self.x), getModes(self.y)

    def vData(self, iFile:int):
        return self.file(iFile)['tasks']['velocity']

    def bData(self, iFile:int):
        return self.file(iFile)['tasks']['buoyancy']

    def pData(self, iFile:int):
        return self.file(iFile)['tasks']['pressure']

    def times(self, iFile:int=None):
        if iFile is None:
            return np.concatenate([self.times(i) for i in range(self.nFiles)])
        return np.array(self.vData(iFile).dims[0]["sim_time"])

    @property
    def nFields(self):
        return [self.nTimes(i) for i in range(self.nFiles)]

    def fields(self, iField):
        offset = np.cumsum(self.nFields)
        iFile = np.argmax(iField < offset)
        iTime = iField - sum(offset[:iFile])
        data = self.file(iFile)["tasks"]
        fields = [
            data["velocity"][iTime, 0],
            data["velocity"][iTime, 1],
            ]
        if self.dim == 3:
            fields += [data["velocity"][iTime, 2]]
        fields += [
            data["buoyancy"][iTime],
            data["pressure"][iTime]
            ]
        return np.array(fields)

    def nTimes(self, iFile:int):
        return self.times(iFile).size

    def readField(self, iFile, name, iBeg=0, iEnd=None, step=1, verbose=False):
        if verbose: print(f"Reading {name} from hdf5 file {iFile}")
        if name == "velocity":
            fData = self.vData(iFile)
        elif name == "buoyancy":
            fData = self.bData(iFile)
        elif name == "pressure":
            fData = self.pData(iFile)
        else:
            raise ValueError(f"cannot read {name} from file")
        shape = fData.shape
        if iEnd is None: iEnd = shape[0]
        rData = range(iBeg, iEnd, step)
        data = np.zeros((len(rData), *shape[1:]))
        for i, iData in enumerate(rData):
            if verbose: print(f" -- field {i+1}/{len(rData)}, idx={iData}")
            data[i] = fData[iData]
        if verbose: print(" -- done !")
        return data


    def getMeanProfiles(self, iFile:int,
                        buoyancy=False, bRMS=False, pressure=False,
                        iBeg=0, iEnd=None, step=1, verbose=False):
        """

        Args:
            iFile (int): file index
            buoyancy (bool, optional): return buoyancy profile. Defaults to False.
            pressure (bool, optional): return pressure profile. Defaults to False.

        Returns:
           profilr (list): mean profiles of velocity, buoyancy and pressure
        """
        profile = []
        axes = 1 if self.dim==2 else (1, 2)
        velocity = self.readField(0, "velocity", iBeg, iEnd, step, verbose)

        # Horizontal mean velocity amplitude
        uH = velocity[:, :self.dim-1]
        meanH = ((uH**2).sum(axis=1)**0.5).mean(axis=axes)
        profile.append(meanH)

        # Vertical mean velocity
        uV = velocity[:, -1]
        meanV = np.mean(abs(uV), axis=axes)
        profile.append(meanV)

        if bRMS or buoyancy:
            b = self.readField(0, "buoyancy", iBeg, iEnd, step, verbose)
        if buoyancy:
            profile.append(np.mean(b, axis=axes))
        if bRMS:
            diff = b - b.mean(axis=axes)[(slice(None), *[None]*(self.dim-1), slice(None))]
            rms = (diff**2).mean(axis=axes)**0.5
            profile.append(rms)
        if pressure:
            p = self.readField(0, "pressure", iBeg, iEnd, step, verbose)
            profile.append(np.mean(p, axis=axes))          # (time_index, Nz)
        return profile


    def getLayersQuantities(self, iFile, iBeg=0, iEnd=None, step=1, verbose=False):
        uMean, _, bRMS = self.getMeanProfiles(
            iFile, bRMS=True, iBeg=iBeg, iEnd=iEnd, step=step, verbose=verbose)
        uMean = uMean.mean(axis=0)
        bRMS = bRMS.mean(axis=0)

        from qmat.lagrange import LagrangeApproximation
        z = self.z
        nFine = int(1e4)
        zFine = np.linspace(0, 1, num=nFine)
        P = LagrangeApproximation(z).getInterpolationMatrix(zFine)

        uMeanFine = P @ uMean
        bRMSFine = P @ bRMS

        deltaU = zFine[np.argmax(uMeanFine[:nFine//2])]
        deltaT = zFine[np.argmax(bRMSFine[:nFine//2])]

        return zFine, uMeanFine, bRMSFine, deltaU, deltaT


    def getMeanSpectrum(self, iFile:int, iBeg=0, iEnd=None, step=1, verbose=False, batchSize=5):
        """
        Mean spectrum from a given output file

        Parameters
        ----------
        iFile : int
            Index of the file to use.
        iBeg : int, optional
            Starting index for the fields to use. The default is 0.
        iEnd : int, optional
            Stopping index (non included) for the fields to use. The default is None.
        step : int, optional
            Index step for the fields to use. The default is 1.
        verbose : bool, optional
            Display infos message in stdout. The default is False.
        batchSize : int, optional
            Number of fields to regroup when computing one FFT. The default is 5.

        Returns
        -------
        spectra : np.ndarray[nT,size]
            The spectrum values for all nT fields.
        """
        spectra = []
        if iEnd is None:
            iEnd = self.nFields[iFile]
        subRanges = decomposeRange(iBeg, iEnd, step, batchSize)
        for iBegSub, iEndSub, stepSub in subRanges:
            if verbose:
                print(f" -- computing for fields in range ({iBegSub},{iEndSub},{stepSub})")
            velocity = self.readField(iFile, "velocity", iBegSub, iEndSub, stepSub, verbose)
            spectra += computeMeanSpectrum(velocity, verbose=verbose, xGrid=self.x, zGrid=self.z)
        return np.concatenate(spectra)


    def getFullMeanSpectrum(self, iBeg:int, iEnd=None):
        """
        Function to get full mean spectrum

        Args:
            iBeg (int): starting file index
            iEnd (int, optional): stopping file index. Defaults to None.

        Returns:
           sMean (np.ndarray): mean spectrum
           k (np.ndarray): wave number
        """
        if iEnd is None:
            iEnd = self.nFiles
        sMean = []
        for iFile in range(iBeg, iEnd):
            energy_spectrum = self.getMeanSpectrum(iFile)
            sx, sz = energy_spectrum                        # (1,time_index,k)
            sMean.append(np.mean((sx+sz)/2, axis=0))        # mean over time ---> (2, k)
        sMean = np.mean(sMean, axis=0)                      # mean over x and z ---> (k)
        np.savetxt(f'{self.folder}/spectrum.txt', np.vstack((sMean, self.k)))
        return sMean, self.k

    def toVTR(self, idxFormat="{:06d}"):
        """
        Convert all 3D fields from the OutputFiles object into a list
        of VTR files, that can be read later with Paraview or equivalent to
        make videos.

        Parameters
        ----------
        idxFormat : str, optional
            Formating string for the index suffix of the VTR file.
            The default is "{:06d}".

        Example
        -------
        >>> # Suppose the FieldsIO object is already writen into outputs.pysdc
        >>> import os
        >>> from pySDC.utils.fieldsIO import Rectilinear
        >>> os.makedirs("vtrFiles")  # to store all VTR files into a subfolder
        >>> Rectilinear.fromFile("outputs.pysdc").toVTR(
        >>>    baseName="vtrFiles/field", varNames=["u", "v", "w", "T", "p"])
        """
        assert self.dim == 3, "can only be used with 3D fields"
        from pySDC.helpers.vtkIO import writeToVTR

        baseName = f"{self.folder}/vtrFiles"
        os.makedirs(baseName, exist_ok=True)
        baseName += "/out"
        template = f"{baseName}_{idxFormat}"
        coords = [self.x, self.y, self.z]
        varNames = ["velocity_x", "velocity_y", "velocity_z", "buoyancy", "pressure"]
        for i in range(np.cumsum(self.nFields)[0]):
            u = self.fields(i)
            writeToVTR(template.format(i), u, coords, varNames)


    def interpolate(self, iFile:int, fileName:str, iField:int=-1):
        assert self.dim == 3, "interpolation only possible for 3D RBC fields"

        fields = self.file(iFile)["tasks"]

        velocity = fields["velocity"][iField]
        buoyancy = fields["buoyancy"][iField]
        pressure = fields["pressure"][iField]

        uCoarse = np.concat([velocity, buoyancy[None, ...], pressure[None, ...]])
        uFine = rbc3dInterpolation(uCoarse)

        nX, nY, nZ = uFine.shape[1:]
        xCoord = np.linspace(0, 1, nX, endpoint=False)
        yCoord = np.linspace(0, 1, nY, endpoint=False)
        zCoord = NodesGenerator("CHEBY-1", "GAUSS").getNodes(nZ)

        output = Rectilinear(np.float64, fileName)
        output.setHeader(5, [xCoord, yCoord, zCoord])
        output.initialize()
        output.addField(0, uFine)


def checkDNS(sMean:np.ndarray, k:np.ndarray, vRatio:int=4, nThrow:int=1):
    """
    Funciton to check DNS
    Args:
        sMean (np.ndarray): mean spectrum
        k (np.ndarray): wave number
        vRatio (int, optional): #to-do
        nThrow (int, optional): number of values to exclude fitting. Defaults to 1.

    Returns:
        status (bool): if under-resolved or not
        [a, b, c] (float): polynomial coefficients
        x, y (float): variable values
        nValues (int): # to-do
    """
    nValues = k.size//vRatio

    y = np.log(sMean[-nValues-nThrow:-nThrow])
    x = np.log(k[-nValues-nThrow:-nThrow])

    def fun(coeffs):
        a, b, c = coeffs
        return np.linalg.norm(y - a*x**2 - b*x - c)

    res = sco.minimize(fun, [0, 0, 0])
    a, b, c = res.x
    status = "under-resolved" if a > 0 else "DNS !"

    return status, [a, b, c], x, y, nValues

def generateChunkPairs(folder:str, N:int, M:int,
                       tStep:int=1, xStep:int=1, zStep:int=1,
                       shuffleSeed=None
):
    """
    Function to generate chunk pairs

    Args:
        folder (str): path to dedalus hdf5 data
        N (int): timesteps of dt
        M (int): size of chunk
        tStep (int, optional): time slicing. Defaults to 1.
        xStep (int, optional): x-grid slicing. Defaults to 1.
        zStep (int, optional): z-grid slicing. Defaults to 1.
        shuffleSeed (int, optional): seed for random shuffle. Defaults to None.

    Returns:
        pairs (list): chunk pairs
    """
    out = OutputFiles(folder)

    pairs = []
    vxData, vzData,  bData, pData = [], [],  [], []
    for iFile in range(0, out.nFiles):
        vxData.append(out.vData(iFile)[:, 0, ::xStep, ::zStep])
        vzData.append(out.vData(iFile)[:, 1, ::xStep, ::zStep])
        bData.append(out.bData(iFile)[:, ::xStep, ::zStep])
        pData.append(out.pData(iFile)[:, ::xStep, ::zStep])
    # stack all arrays
    vxData = np.concatenate(vxData)
    vzData = np.concatenate(vzData)
    bData = np.concatenate(bData)
    pData = np.concatenate(pData)

    assert vxData.shape[0] == vzData.shape[0]
    assert vzData.shape[0] == pData.shape[0]
    assert vzData.shape[0] == bData.shape[0]
    nTimes = vxData.shape[0]

    for i in range(0, nTimes-M-N+1, tStep):
        chunk1 = np.stack((vxData[i:i+M],vzData[i:i+M],bData[i:i+M],pData[i:i+M]), axis=1)
        chunk2 = np.stack((vxData[i+N:i+N+M],vzData[i+N:i+N+M], bData[i+N:i+N+M],
                           pData[i+N:i+N+M]), axis=1)
        # chunks are shape (M, 4, Nx//xStep, Nz//zStep)
        assert chunk1.shape == chunk2.shape
        pairs.append((chunk1, chunk2))

    # shuffle if a seed is given
    if shuffleSeed is not None:
        random.seed(shuffleSeed)
        random.shuffle(pairs)

    return pairs


def contourPlot(field, x, y, time=None,
                title=None, refField=None, refTitle=None, saveFig=False,
                closeFig=True, error=False, refScales=False):

    fig, axs = plt.subplots(1 if refField is None else 2)
    ax = axs if refField is None else axs[0]

    if refField is not None and refScales:
        scales = (np.min(refField), np.max(refField))

    def setup(ax):
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel("x")
        ax.set_ylabel("z")

    def setColorbar(field, im, ax, error=False):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        if refScales:
            im.cmap.set_under("white")
            im.cmap.set_over("white")
            im.set_clim(*scales)
            fig.colorbar(im, cax=cax, ax=ax, ticks=np.linspace(*scales, 3))
        else:
            fig.colorbar(im, cax=cax, ax=ax, ticks=np.linspace(np.min(field), np.max(field), 3))

    im = ax.pcolormesh(x, y, field)
    setColorbar(field, im, ax, error)
    timeSuffix = f' at t = {np.round(time,3)}s' if time is not None else ''
    ax.set_title(f'{title}{timeSuffix}')
    setup(ax)

    if refField is not None:
        im = axs[1].pcolormesh(x, y, refField)
        setColorbar(refField, im, axs[1])
        axs[1].set_title(f'{refTitle}{timeSuffix}')
        setup(axs[1])

    plt.tight_layout()
    if saveFig:
        plt.savefig(saveFig)
    if closeFig:
        plt.close(fig)
