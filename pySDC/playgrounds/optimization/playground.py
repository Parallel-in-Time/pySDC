import numpy as np
import scipy.optimize as opt
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right


m = 5
coll = CollGaussRadau_Right(num_nodes=m, tleft=0.0, tright=1.0)
Q = coll.Qmat[1:, 1:]

def func(x):
    return max(abs(np.linalg.eigvals(np.eye(m) - np.diag(x).dot(Q))))


def func2(x0):
    # x0 = 1E-01 * np.ones(m)
    d = opt.minimize(func, x0, method='Nelder-Mead', tol=1E-08)
    print(d.fun, np.diag(np.linalg.inv(np.diag(d.x))))
    return d.fun


y0 = 1.0 / np.asarray([0.2818591930905709, 0.2011358490453793, 0.06274536689514164, 0.11790265267514095, 0.1571629578515223])
d2 = opt.minimize(func2, y0, method='Nelder-Mead', tol=1E-08)
print(d2)
# print(np.linalg.inv(np.diag(d.x)))