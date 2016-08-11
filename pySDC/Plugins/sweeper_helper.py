import scipy.linalg as LA
import numpy as np


def get_Qd(coll, Nnodes, qd_type):
    QDmat = np.zeros(coll.Qmat.shape)
    if qd_type == 'LU':
        QT = coll.Qmat[1:,1:].T
        [P, L, U] = LA.lu(QT, overwrite_a=True)
        QDmat[1:,1:] = U.T
    elif qd_type == 'IE':
        for m in range(coll.num_nodes + 1):
            QDmat[m, 1:m + 1] = coll.delta_m[0:m]
    elif qd_type == 'IEpar':
        for m in range(coll.num_nodes + 1):
            QDmat[m, m] = np.sum(coll.delta_m[0:m])
    elif qd_type == 'Qpar':
        QDmat = np.diag(np.diag(coll.Qmat))
    elif qd_type == 'GS':
        QDmat = np.tril(coll.Qmat)
    elif qd_type == 'PIC':
        # QDmat = coll.Qmat
        QDmat = np.zeros(coll.Qmat.shape)
    else:
        print('qd_type not implemented...', qd_type)
        exit()
    return QDmat