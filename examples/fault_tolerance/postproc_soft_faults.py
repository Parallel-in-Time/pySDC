import pickle as pkl
import numpy as np
import math
import os
from matplotlib import rc
import matplotlib.pyplot as plt


if __name__ == "__main__":

    # list = [ ('HEAT_soft_faults_corr10x_N1000.pkl','SDC, correction, a = 10',(10,20,30,40,50)) ,
    #          ('HEAT_soft_faults_nocorr_N1000.pkl',' SDC, no correction',(10,20,30,40,50)) ]
    list = [ ('HEAT_MLSDC_soft_faults_corr10x_N1000.pkl','MLSDC, correction, a = 10',(5,15,25,35,45)) ,
             ('HEAT_MLSDC_soft_faults_nocorr_N1000.pkl',' MLSDC, no correction', (5,15,25,35,45)) ]

    for file,title,bins in list:
        soft_stats = pkl.load(open(file,'rb'))

        soft_stats_cleaned = []
        min_iter = 99
        for item in soft_stats:
            if item[0] > 0 and not math.isnan(item[2][-1]):
                min_iter = min(item[1],min_iter)
                soft_stats_cleaned.append(item)
            elif item[0] == 0:
                ref_res = np.log10(item[2])

        nstats = len(soft_stats_cleaned)
        print(nstats,min_iter)

        #####

        mean_res = np.zeros(min_iter)

        for item in soft_stats_cleaned:
            mean_res[:] = mean_res[:] + np.log10(item[2][0:min_iter].clip(1E-07,1))

        mean_res[:] = mean_res[:]/nstats

        stddev_res = np.zeros(min_iter)
        for item in soft_stats_cleaned:
            stddev_res[:] = stddev_res[:] + (np.log10(item[2][0:min_iter].clip(1E-07,1)) - mean_res[:])**2

        stddev_res[:] = np.sqrt(stddev_res[:]/(nstats-1))

        # conf_coeff = 1.96
        # conf_max = mean_res[:] + conf_coeff*stddev_res[:]/np.sqrt(nstats)
        # conf_min = mean_res[:] - conf_coeff*stddev_res[:]/np.sqrt(nstats)
        conf_max = mean_res[:] + stddev_res[:]
        conf_min = mean_res[:] - stddev_res[:]

        fig, ax = plt.subplots(figsize=(10,10))

        plt.plot(mean_res,'r-',label='mean')
        plt.plot(ref_res,'k--',label='reference')
        plt.plot(conf_min,'b-',label='standard deviation')
        plt.plot(conf_max,'b-',)

        plt.legend()

        plt.xlabel('iteration')
        plt.ylabel('log10(residual)')
        plt.title(title)

        plt.tight_layout()

        fname = 'residual_'+os.path.splitext(file)[0]+'.png'

        plt.savefig(fname, rasterized=True, transparent=True, bbox_inches='tight')

        #####

        niter = [item[1] for item in soft_stats_cleaned]

        fig, ax = plt.subplots(figsize=(10,10))

        plt.hist(niter,bins=bins,normed=False,histtype='bar',rwidth=0.8)

        # plt.xlim((10,50))
        plt.ylim((0,400))

        # plt.xticks([15,25,35,45],['10-20','20-30','30-40','40-50'])

        plt.xlabel('number of iterations')
        plt.ylabel('number of runs')
        plt.title(title)

        plt.tight_layout()

        fname = 'iterations_'+os.path.splitext(file)[0]+'.png'

        plt.savefig(fname, rasterized=True, transparent=True, bbox_inches='tight')



    plt.show()



