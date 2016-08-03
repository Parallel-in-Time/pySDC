import matplotlib.pylab as plt
from collections import namedtuple
import pickle
import os

if __name__ == "__main__":

    # Set up plotting parameters
    params = {'legend.fontsize': 20,
              'figure.figsize': (12, 8),
              'axes.labelsize': 20,
              'axes.titlesize': 20,
              'xtick.labelsize': 16,
              'ytick.labelsize': 16,
              'lines.linewidth': 3
              }
    plt.rcParams.update(params)

    file_list = ['fenics_heat_unforced_sdc.pkl', 'fenics_heat_unforced_mlsdc.pkl']

    for file in file_list:

        # Load pkl file with the results
        ID = namedtuple('ID', ['c_nvars', 'dt'])
        fileopen = open(file,'rb')
        results = pickle.load(fileopen)

        filename = os.path.splitext(file)[0]

        # assemble lists of time steps and number of elements
        c_nvars_list = []
        dt_list = []
        for key in results.keys():
            if isinstance(key, ID):
                # print(key,results[key])
                if not key.c_nvars in c_nvars_list:
                    c_nvars_list.append(key.c_nvars)
                if not key.dt in dt_list:
                    dt_list.append(key.dt)
        c_nvars_list = sorted(c_nvars_list)
        dt_list = sorted(dt_list)
        print('c_nvars_list:',c_nvars_list)
        print('dt_list:', dt_list)
        # continue
        # plot errors over dt, for all number of elements
        plt.figure()
        plt.xlim([min(dt_list)/2, max(dt_list)*2])
        plt.title(filename+' - errors')
        plt.xlabel('dt')
        plt.ylabel('rel. error')
        plt.grid()

        id = ID(c_nvars=c_nvars_list[0], dt=dt_list[-1])
        base_error = results[id][1]
        order_guide_time = [base_error/(2**(5*i))/5 for i in range(len(dt_list)-1,-1,-1)]
        plt.loglog(dt_list,order_guide_time,marker='', color='k', ls='--',label='5th order (time)')

        id = ID(c_nvars=c_nvars_list[0], dt=dt_list[1])
        base_error = results[id][1]
        order_guide_space = [base_error / (2 ** (2 * i)) for i in range(0, len(c_nvars_list))]
        xvars = [dt_list[1] for i in range(0, len(order_guide_space))]
        plt.plot(xvars, order_guide_space, marker='o', color='k', markersize=10, ls='--', label='2nd order (space)')

        min_err = 1E99
        max_err = 0E00
        for c_nvars in c_nvars_list:

            err_classical_rel = []
            xvars = []
            for dt in dt_list:
                id = ID(c_nvars=c_nvars, dt=dt)
                if id in results:
                    err = results[id][1]
                    min_err = min(err,min_err)
                    max_err = max(err,max_err)
                    err_classical_rel.append(err)
                    xvars.append(dt)


            plt.loglog(xvars,err_classical_rel,label=c_nvars)

        plt.ylim([min_err / 10, max_err * 10])
        plt.legend(loc=2, ncol=2, numpoints = 1)
        fname = 'errors_' + filename + '.pdf'
        plt.savefig(fname, rasterized=True, bbox_inches='tight')



        # plot number of iterations over dt, for all number of elements
        plt.figure()
        plt.xlim([min(dt_list) / 2, max(dt_list) * 2])
        plt.title(filename + ' - iteration counts')
        plt.xlabel('dt')
        plt.ylabel('number of iterations')
        plt.grid()

        min_niter = 100
        max_niter = 0

        niter_list = []
        niter_lower = []
        niter_upper = []
        xvars = []
        for dt in dt_list:

            niter = 0
            nentries = 0
            min_niter_loc = 100
            max_niter_loc = 0
            for c_nvars in c_nvars_list:
                id = ID(c_nvars=c_nvars, dt=dt)
                if id in results:
                    niter += results[id][0]
                    nentries += 1
                    min_niter_loc = min(results[id][0], min_niter_loc)
                    max_niter_loc = max(results[id][0], max_niter_loc)
            niter_mean = 1.0*niter/nentries
            niter_lower.append(niter_mean-min_niter_loc)
            niter_upper.append(max_niter_loc-niter_mean)
            min_niter = min(niter_mean, min_niter)
            max_niter = max(niter_mean, max_niter)
            niter_list.append(niter_mean)
            xvars.append(dt)

        plt.errorbar(xvars,niter_list,yerr=[niter_lower,niter_upper])#, label=c_nvars)


        plt.ylim([min_niter-1, max_niter+1])
        plt.xscale('log')
        fname = 'niter_' + filename + '.pdf'
        plt.savefig(fname, rasterized=True, bbox_inches='tight')

        # plt.legend(loc=2, ncol=2, numpoints=1)

    plt.show()