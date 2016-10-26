
import numpy as np
import matplotlib.pylab as plt
from collections import namedtuple
import pickle

if __name__ == "__main__":

    ID = namedtuple('ID', ['setup', 'qd_type', 'param'])
    file = open('results_iterations_precond.pkl','rb')
    results = pickle.load(file)

    qd_type_list = []
    setup_list = []
    for key in results.keys():
        if isinstance(key,ID):
            if not key.qd_type in qd_type_list:
                qd_type_list.append(key.qd_type)
        elif isinstance(key,str):
            setup_list.append(key)
    print('Found these type of preconditioners:',qd_type_list)
    print('Found these setups:',setup_list)


    for setup in setup_list:

        plt.figure()
        for qd_type in qd_type_list:
            niter_heat = np.zeros(len(results[setup][1]))
            for key in results.keys():
                if isinstance(key,ID):
                    if key.setup == setup and key.qd_type == qd_type:
                        xvalue = results[setup][1].index(key.param)
                        niter_heat[xvalue] = results[key]
            plt.semilogx(results[setup][1],niter_heat, label=qd_type, lw=2)

        plt.ylim([0,100])
        plt.legend()
        plt.title(setup)

    plt.show()
