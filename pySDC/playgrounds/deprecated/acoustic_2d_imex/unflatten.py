import numpy as np


def unflatten(uin, dim, Nx, Nz):
    temp = np.zeros((dim, Nx * Nz))
    temp = np.asarray(np.split(uin, dim))
    uout = np.zeros((dim, Nx, Nz))
    for i in range(0, dim):
        uout[i, :, :] = np.split(temp[i, :], Nx)
    return uout
