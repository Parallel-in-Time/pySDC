#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MPI space-time decomposition
"""
from mpi4py import MPI
import sys

def initSpaceTimeCommunicators(nProcSpace=None, nProcTime=None, groupSpace=True):
    
    gComm = MPI.COMM_WORLD
    gRank = gComm.Get_rank()
    gSize = gComm.Get_size()

    def log(msg):
        if gRank == 0:
            print(msg)

    if (nProcTime is None) and (nProcSpace is None):
        # No parallelization specified, space takes everything
        nProcTime = 1
        nProcSpace = gSize
    elif nProcSpace is None:
        # Only time parallelization specified, space takes the rest
        nProcSpace = gSize//nProcTime
    elif nProcTime is None:
        # Only space parallelization specified, time takes the rest
        nProcTime = gSize//nProcSpace

    log("-- Starting MPI initialization ...")
    log(f" - nProcTotal = {gSize}")
    log(f" - nProcSpace = {nProcSpace}")
    log(f" - nProcTime = {nProcTime}")

    # Check for inadequate decomposition
    if (gSize != nProcSpace*nProcTime) and (gSize != 1):
        if gRank == 0:
            raise ValueError(
                f' product of nProcSpace ({nProcSpace}) with nProcTime ({nProcTime})'
                f' is not equal to the total number of MPI processes ({gSize})')
        else:
            sys.exit(0)
            
    
    # Information message
    if gSize == 1:
        log(" - no parallelization at all")
        nProcSpace = 1
        nProcSpace = 1
    else:
        if nProcSpace != 1:
            log(f" - space parallelization activated : {nProcSpace} mpi processes")
        else:
            log(" - no space parallelization")
        if nProcSpace != 1:
            log(f" - time parallelization activated : {nProcSpace} mpi processes")
        else:
            log(" - no time parallelization")
        log('-- finished MPI initialization --')

    # Construction of MPI communicators
    if groupSpace:
        tColor = gRank % nProcSpace
        tComm = gComm.Split(tColor, gRank)
        gComm.Barrier()
        sColor = (gRank-gRank % nProcSpace)/nProcSpace
        sComm = gComm.Split(sColor, gRank)
        gComm.Barrier()
    else:
        sColor = gRank % nProcTime
        sComm = gComm.Split(sColor, gRank)
        gComm.Barrier()
        tColor = (gRank-gRank % nProcTime)/nProcSpace
        tComm = gComm.Split(tColor, gRank)
        gComm.Barrier()

    return gComm, sComm, tComm


if __name__ == "__main__":

    gComm, sComm, tComm = initSpaceTimeCommunicators(nProcTime=4)

    gRank, gSize = gComm.Get_rank(), gComm.Get_size()
    sRank, sSize = sComm.Get_rank(), sComm.Get_size()
    tRank, tSize = tComm.Get_rank(), tComm.Get_size()

    from time import sleep
    sleep(0.1*gRank)

    import psutil
    coreNum = psutil.Process().cpu_num()

    print(f"Global rank {gRank} ({gSize}), space rank {sRank} ({sSize}),"
          f" time rank {tRank} ({tRank}) running on CPU core {coreNum}")
    