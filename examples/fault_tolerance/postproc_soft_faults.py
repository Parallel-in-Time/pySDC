import pickle as pkl
import numpy as np
import math
import os
from matplotlib import rc
import matplotlib.pyplot as plt
import scipy.stats as stats


if __name__ == "__main__":

    # list = [ ('HEAT_SDC_soft_faults_nocorr_N1000.pkl', 'SDC, no correction'),
    #          ('HEAT_SDC_soft_faults_corr10x_N1000.pkl', 'SDC, correction, a = 10'),
    #          ('HEAT_SDC_soft_faults_corr5x_N1000.pkl', 'SDC, correction, a = 5'),
    #          ('HEAT_SDC_soft_faults_corr1x_N1000.pkl','SDC, correction, a = 1') ]

    list = [ ('HEAT_MLSDC_soft_faults_nocorr_N1000.pkl', 'MLSDC, no correction'),
             ('HEAT_MLSDC_soft_faults_corr10x_N1000.pkl', 'MLSDC, correction, a = 10') ,
    #          ('HEAT_MLSDC_soft_faults_corr5x_N1000.pkl', 'MLSDC, correction, a = 5'),
             ('HEAT_MLSDC_soft_faults_corr1x_N1000.pkl','MLSDC, correction, a = 1') ]

    # list = [ ('HEAT_PFASST_soft_faults_corr10x_N1000_NOCOARSE.pkl','PFASST, correction, a = 10'),
    #          ('HEAT_PFASST_soft_faults_nocorr_N1000_NOCOARSE.pkl','PFASST, no correction')]

    iterhist = []
    legendhist = []

    for file,title in list:
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
                soft_stats_cleaned.append(item)
                nfaults += item[0]
                ndetect += item[1]
                nhits += item[2]
                nmissed += item[3]
                niter += item[4]
            elif item[0] == 0:
                ref_res = np.log10(item[5])
                min_iter = len(item[5])

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


        # for l in range(min_iter):
        #     iters = [np.log10(item[5][l].clip(1E-07,1)) for item in soft_stats_cleaned]
        #     param = stats.gamma.fit(iters)
        #     print(param[1],ref_res[l])

        # continue

        #####

        mean_res = np.zeros(min_iter)

        for item in soft_stats_cleaned:
            mean_res[:] = mean_res[:] + np.log10(item[5][0:min_iter].clip(1E-14,1))

        mean_res[:] = mean_res[:]/nruns_clean

        dev_res = np.zeros(min_iter)
        min_res = np.zeros(min_iter)
        max_res = np.zeros(min_iter)
        max_res[:] = -99
        for item in soft_stats_cleaned:
            dev_res[:] = dev_res[:] + np.sqrt((np.log10(item[5][0:min_iter].clip(1E-14,1)) - mean_res[:])**2)
            min_res[:] = np.minimum(min_res[:],np.log10(item[5][0:min_iter].clip(1E-14,1)))
            max_res[:] = np.maximum(max_res[:],np.log10(item[5][0:min_iter].clip(1E-14,1)))

        dev_res[:] = dev_res[:]/nruns_clean

        fig, ax = plt.subplots(figsize=(10,10))

        lw = 3
        ms = 8
        mw = 2

        plt.plot(ref_res,  linestyle='--',color='k',linewidth=lw, marker='o', markersize=ms, markeredgecolor='k', markeredgewidth=mw, label='reference')
        plt.plot(mean_res, linestyle='-', color='r',linewidth=lw, marker='o', markersize=ms, markeredgecolor='k', markeredgewidth=mw, label='mean')

        plt.plot([],[],color='grey',alpha=0.5,linewidth=10,label='range of residuals')
        # plt.fill_between(range(min_iter),mean_res-dev_res, mean_res+dev_res, alpha=0.5, facecolor='grey', edgecolor='none')
        plt.fill_between(range(min_iter), min_res, max_res, alpha=0.5, facecolor='grey', edgecolor='none')

        plt.legend(numpoints=1)

        plt.xlabel('iteration')
        plt.ylabel('log10(residual)')
        plt.title(title)

        plt.tight_layout()

        fname = 'residual_'+os.path.splitext(file)[0]+'.png'

        plt.savefig(fname, rasterized=True, transparent=True, bbox_inches='tight')

        # continue
        #####

        iterhist.append([item[4] for item in soft_stats_cleaned])
        legendhist.append(title)

    fig, ax = plt.subplots(figsize=(10,10))

    # bins = np.arange(min_iter,51,1)
    plt.hist(iterhist,bins=15,normed=False,histtype='bar',rwidth=0.7,log=True,label=legendhist)

    # plt.xlim((min_iter-1,51))
    plt.ylim((0,nruns))
    plt.legend()
    # print(bins)
    # plt.xticks(bins+1.5,bins)

    plt.xlabel('number of iterations')
    plt.ylabel('number of runs')
    # plt.title()

    plt.tight_layout()

    fname = 'iterations_MLSDC_histogram.png'

    plt.savefig(fname, rasterized=True, transparent=True, bbox_inches='tight')

    plt.show()



