import numpy as np
import matplotlib.pylab as plt
from collections import namedtuple
import pickle

if __name__ == "__main__":

    params = {'legend.fontsize': 20,
              'figure.figsize': (12, 8),
              'axes.labelsize': 20,
              'axes.titlesize': 20,
              'xtick.labelsize': 16,
              'ytick.labelsize': 16,
              'lines.linewidth': 3
              }
    plt.rcParams.update(params)

    ID = namedtuple('ID', ['c_nvars', 'dt'])
    file = open('fenics_heat_unforced_sdc.pkl','rb')
    results = pickle.load(file)
    # print(results)

    c_nvars_list = []
    dt_list = []
    for key in results.keys():
        if isinstance(key, ID):
            if not key.c_nvars in c_nvars_list:
                c_nvars_list.append(key.c_nvars)
            if not key.dt in dt_list:
                dt_list.append(key.dt)
    c_nvars_list = sorted(c_nvars_list)
    dt_list = sorted(dt_list)
    print('c_nvars_list:',c_nvars_list)
    print('dt_list:', dt_list)

    plt.figure()
    plt.xlim([min(dt_list)/2, max(dt_list)*2])
    plt.title('HEAT EQ')
    plt.xlabel('dt')
    plt.ylabel('rel. error')
    plt.grid()

    id = ID(c_nvars=c_nvars_list[0], dt=dt_list[-1])
    base_error = results[id][1]
    order_guide_time = [base_error/(2**(5*i))/5 for i in range(len(dt_list)-1,-1,-1)]
    plt.loglog(dt_list,order_guide_time,'k--',label='5th order (time)')

    id = ID(c_nvars=c_nvars_list[0], dt=dt_list[1])
    base_error = results[id][1]
    order_guide_space = [base_error / (2 ** (2 * i)) for i in range(0, len(c_nvars_list))]
    xvars = [dt_list[1] for i in range(0, len(order_guide_space))]
    plt.plot(xvars, order_guide_space, marker='o', color='k', markersize=10, ls='--', label='2nd order (space)')

    min_err = 1E99
    max_err = 0E00
    for c_nvars in c_nvars_list:

        err_classical_rel = []
        for dt in dt_list:
            id = ID(c_nvars=c_nvars, dt=dt)
            err = results[id][1]
            min_err = min(err,min_err)
            max_err = max(err,max_err)
            err_classical_rel.append(results[id][1])


        plt.loglog(dt_list,err_classical_rel,label=c_nvars)

    plt.ylim([min_err / 10, max_err * 10])
    plt.legend(loc=2, ncol=2, numpoints = 1)

    plt.show()