from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.core.Sweeper import sweeper
from matplotlib import pyplot as plt
import scipy.optimize as opt
import numpy as np
import scipy

def rho(x):
    global Q, M
    return np.linalg.norm(Q - np.diag([x[i] for i in range(M)]), np.inf) / np.amin([abs(x[i]) for i in range(M)])


M = 3
prec_list = ['IE', 'LU', 'MIN', 'EE']
color_list = ['r', 'g', 'b', 'c']
ldt_list = np.arange(-100, 0, 0.1)
results = {}

for prec in prec_list:
    sw = sweeper({'collocation_class': CollGaussRadau_Right, 'num_nodes': M})
    if prec != 'EE':
        QDelta = sw.get_Qdelta_implicit(sw.coll, prec)[1:, 1:]
    else:
        QDelta = sw.get_Qdelta_explicit(sw.coll, prec)[1:, 1:]
    QDL = np.tril(QDelta, -1)
    QDD = np.diag(np.diag(QDelta))
    Q = sw.coll.Qmat[1:, 1:]

    a = np.linalg.norm(np.linalg.inv(QDD), np.inf)
    b = 1 - np.linalg.norm(Q-QDelta, np.inf) * np.linalg.norm(QDL, np.inf)
    print(prec, (a**3 + 1)/(1-b))
    continue

    result_Rnorm = np.zeros(len(ldt_list))
    result_Rrho = np.zeros(len(ldt_list))
    result_est = np.zeros(len(ldt_list))
    for i, ldt in enumerate(ldt_list):

        R = np.linalg.matrix_power(ldt * np.linalg.inv(np.eye(M) - ldt * QDelta).dot(Q - QDelta), 1)
        result_Rnorm[i] = np.linalg.norm(R, np.infty)
        result_Rrho[i] = np.amax(abs(np.linalg.eigvals(R)))
        # result_est[i] = abs(ldt) * np.linalg.norm(np.linalg.inv(np.eye(M) - ldt * QDelta), np.inf) * np.linalg.norm(Q - QDelta, np.inf)
        # result_est[i] = abs(ldt) * np.linalg.norm(np.linalg.inv(np.eye(M) - ldt * QDD), np.inf) * np.linalg.norm(np.linalg.inv(np.eye(M) - ldt * np.linalg.inv(np.eye(M) - ldt * QDD).dot(QDL)), np.inf) * np.linalg.norm(Q - QDelta, np.inf)
        result_est[i] = abs(ldt) * 1.0 / min(1 - ldt * np.diag(QDelta)) * (1 + 1/ldt) * np.linalg.norm(Q - QDelta, np.inf)
        # result_est[i] = np.linalg.norm(np.linalg.inv(np.eye(M) - ldt * np.linalg.inv(np.eye(M) - ldt * QDD).dot(QDL)), np.inf)
        # result_est[i] = abs(ldt) * np.linalg.norm(QDL, np.inf) / min(1 - ldt * np.diag(QDelta))

    results[prec] = [result_Rnorm, result_Rrho, result_est]
exit()
fig = plt.figure()

for prec, color in zip(prec_list, color_list):
    plt.plot(ldt_list, results[prec][0], label=prec, color=color)
    plt.plot(ldt_list, results[prec][2], label=prec, ls='--', color=color)

plt.ylim(0, 15)
plt.legend()

# fig = plt.figure()
#
# for prec in prec_list:
#
#
# plt.ylim(0,2)
# plt.legend()

plt.show()
exit()

