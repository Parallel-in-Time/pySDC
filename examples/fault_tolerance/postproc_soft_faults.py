import pickle as pkl
import numpy as np
import math
import os
from matplotlib import rc
import matplotlib.pyplot as plt


if __name__ == "__main__":

    list = [ ('HEAT_SDC_soft_faults_nocorr_N1000.pkl',' SDC, no correction',(10,20,30,40,50)),
             ('HEAT_SDC_soft_faults_corr1x_N1000.pkl','SDC, correction, a = 1',(10,20,30,40,50)),
             ('HEAT_SDC_soft_faults_corr5x_N1000.pkl','SDC, correction, a = 5',(10,20,30,40,50)),
             ('HEAT_SDC_soft_faults_corr10x_N1000.pkl','SDC, correction, a = 10',(10,20,30,40,50)) ]

    # list = [ ('HEAT_MLSDC_soft_faults_corr10x_N1000.pkl','MLSDC, correction, a = 10',(5,15,25,35,45)) ,
    #          ('HEAT_MLSDC_soft_faults_nocorr_N1000.pkl',' MLSDC, no correction', (5,15,25,35,45)) ]
    # list = [ ('HEAT_PFASST_soft_faults_corr10x_N1000_NOCOARSE.pkl','PFASST, correction, a = 10', ()),
    #          ('HEAT_PFASST_soft_faults_nocorr_N1000_NOCOARSE.pkl','PFASST, no correction', ())]

    for file,title,bins in list:
        soft_stats = pkl.load(open(file,'rb'))

        soft_stats_cleaned = []
        min_iter = 99
        nfaults = 0
        ndetect = 0
        nhits = 0
        nmissed = 0
        niter = 0
        for item in soft_stats:
            if item[0] > 0 and not math.isnan(item[6]):
                min_iter = min(item[4],min_iter)
                soft_stats_cleaned.append(item)
                nfaults += item[0]
                ndetect += item[1]
                nhits += item[2]
                nmissed += item[3]
                niter += item[4]
            elif item[0] == 0:
                ref_res = np.log10(item[5])

        nruns = len(soft_stats)
        nruns_clean = len(soft_stats_cleaned)

        print()
        print('Setup:',file)
        print('Found %i successfull runs out of %i overall runs' %(nruns_clean,nruns))
        print('     Number of iterations (full/avg): %6i \t %4.2f' %(niter, niter/nruns_clean) )
        print('     Number of faults (full/avg):\t  %6i \t %4.2f' %(nfaults, nfaults/nruns_clean) )
        print('     Number of detections (full/avg): %6i \t %4.2f' %(ndetect,ndetect/nruns_clean) )
        if ndetect > 0:
            print('     Number of hits (full/avg/ref. to faults):\t\t\t   %6i \t %4.2f \t %4.2f' %(nhits, nhits/nruns_clean, nhits/nfaults) )
            print('     Number of false negatives (full/avg/ref. to faults):  %6i \t %4.2f \t %4.2f' %(nmissed, nmissed/nruns_clean, nmissed/nfaults) )
            print('     Number of false positives (full/avg/ref. to detects): %6i \t %4.2f \t %4.2f' %(ndetect-nhits, (ndetect-nhits)/nruns_clean, (ndetect-nhits)/ndetect) )


        continue

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

        # continue
        #####

        niter = [item[1] for item in soft_stats_cleaned]

        fig, ax = plt.subplots(figsize=(10,10))

        plt.hist(niter,normed=False,histtype='bar',rwidth=0.8)

        # plt.xlim((10,50))
        # plt.ylim((0,400))

        # plt.xticks([15,25,35,45],['10-20','20-30','30-40','40-50'])

        plt.xlabel('number of iterations')
        plt.ylabel('number of runs')
        plt.title(title)

        plt.tight_layout()

        fname = 'iterations_'+os.path.splitext(file)[0]+'.png'

        plt.savefig(fname, rasterized=True, transparent=True, bbox_inches='tight')



    plt.show()



