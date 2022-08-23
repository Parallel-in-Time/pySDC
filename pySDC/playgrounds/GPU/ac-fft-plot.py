import pickle
import numpy as np
import matplotlib.pyplot as plt

name_cpu = f'pickle/ac-jusuf-pySDC-cpu_fft.pickle'
name_gpu = 'pickle/ac-jusuf-pySDC-gpu_fft.pickle'

with open(name_cpu, 'rb') as f:
   data_cpu = pickle.load(f)
Ns = data_cpu['Ns']
D = data_cpu['D']
Ns_plot = Ns**D
schritte = data_cpu['schritte']
dt = data_cpu['dt']
iteration = data_cpu['iteration']
tol = data_cpu['Tolerance']
# times_CPU = data_cpu['times']
setup_CPU = data_cpu['setup']
cg_CPU = data_cpu['cg-time']
# cg_Count_CPU = data_cpu['cg-count']
f_im_CPU = data_cpu['f-time-imp']
f_ex_CPU = data_cpu['f-time-exp']
with open(name_gpu, 'rb') as f:
   data_gpu = pickle.load(f)
print(data_gpu['f-time-imp'])
# times_GPU = data_gpu['times']
setup_GPU = data_gpu['setup']
cg_GPU = data_gpu['cg-time']
# cg_Count_GPU = data_gpu['cg-count']
f_im_GPU = data_gpu['f-time-imp']
f_ex_GPU = data_gpu['f-time-exp']

times_CPU = cg_CPU+f_im_CPU+f_ex_CPU
times_GPU = cg_GPU+f_im_GPU+f_ex_GPU
# Start Plotting Time Marching
##############################################################################
Ns_plot = Ns**D
plt.scatter(Ns_plot, times_GPU, label="GPU")
plt.plot(Ns_plot, times_GPU)
plt.scatter(Ns_plot, times_CPU, label="CPU")
plt.plot(Ns_plot, times_CPU)
plt.xscale('log')
plt.yscale('log')
# plt.title("Simple SDC (GMRES) Allen-Cahn 2D:\nGPU vs CPU only time_marching")
plt.title("pySDC Allen-Cahn 2D FFT:\nGPU vs CPU only time_marching")
plt.xlabel('degrees of freedom')
plt.ylabel('Time in s')
plt.legend()
# plt.savefig('pdfs/allen-cahn_jusuf_tm_log2.pdf')
plt.show()
plt.clf()
# Start Plotting Setup Time
##############################################################################
plt.scatter(Ns_plot, setup_GPU, label="GPU")
plt.plot(Ns_plot, setup_GPU)
plt.scatter(Ns_plot, setup_CPU, label="CPU")
plt.plot(Ns_plot, setup_CPU)
plt.xscale('log')
plt.yscale('log')
plt.title("pySDC Allen-Cahn 2D FFT:\nGPU vs CPU only Setup")
plt.xlabel('degrees of freedom')
plt.ylabel('Time in s')
plt.legend()
# plt.savefig('pdfs/allen-cahn_jusuf_setup_log2.pdf')
plt.show()
plt.clf()
# Start Plotting Factors
##############################################################################
plt.scatter(Ns_plot, times_CPU/times_GPU, label="Factor marching")
print(times_CPU/times_GPU)
plt.scatter(Ns_plot, setup_CPU/setup_GPU, label="Factor setup")
plt.scatter(Ns_plot, cg_CPU/cg_GPU-0.08*(cg_CPU/cg_GPU), label="Factor cg")
print(cg_CPU/cg_GPU)
plt.xscale('log')
plt.yscale('log')
plt.title("pySDC Allen-Cahn 2D FFT:\nCPU / GPU")
plt.xlabel('degrees of freedom')
plt.ylabel('Factor')
plt.legend()
# plt.savefig('pdfs/allen-cahn_jusuf_factors_log2.pdf')
plt.show()
plt.clf()
plt.scatter(Ns_plot, f_im_CPU/f_im_GPU, label="Factor f implizit")
plt.scatter(Ns_plot, f_ex_CPU/f_ex_GPU, label="Factor f explizit")
plt.xscale('log')
plt.yscale('log')
plt.title("pySDC Allen-Cahn 2D FFT:\nCPU / GPU")
plt.xlabel('degrees of freedom')
plt.ylabel('Factor')
plt.legend()
# plt.savefig('pdfs/allen-cahn_jusuf_factors_f_log2.pdf')
plt.show()
plt.clf()
"""
# Start Plotting Bar-GMRES-Count
##############################################################################
width = 30
plt.bar(Ns-width, cg_Count_GPU, 2*width, label="GPU")
plt.bar(Ns+width, cg_Count_CPU, 2*width, label="CPU")
# plt.xscale('log')
# plt.yscale('log')
plt.title("Anzahl der cg Iteration pro Gesamtlauf")
plt.xlabel('degrees of freedom per dim')
plt.ylabel('Counter')
# plt.ylim((149, 151))
plt.legend()
# plt.savefig('pdfs/allen-cahn_jusuf_count2.pdf')
plt.show()
plt.clf()
# Start Plotting Error
##############################################################################
"""
"""
plt.plot(Ns_plot, error_GPU, label="GPU")
plt.plot(Ns_plot, error_CPU, label="CPU")
plt.xscale('log')
plt.yscale('log')
# plt.loglog(Ns_plot, 1./np.power(Ns, problem.params['order']))
plt.title("Fehlerauflistung nach "+str(schritte)+" Zeitschritten, dt="+str(dt))
plt.xlabel('degrees of freedom')
plt.ylabel('Error')
plt.legend()
plt.savefig('pdfs/allen-cahn_jusuf_error_log.pdf')
plt.clf()
"""
# Start Plotting All Times
##############################################################################
plt.scatter(Ns_plot, times_GPU, label="time_marching")
plt.plot(Ns_plot, times_GPU)
plt.scatter(Ns_plot, setup_GPU, label="setup")
plt.plot(Ns_plot, setup_GPU)
plt.scatter(Ns_plot, cg_GPU, label="cg")
plt.plot(Ns_plot, cg_GPU)
plt.scatter(Ns_plot, f_im_GPU, label="f implizit")
plt.plot(Ns_plot, f_im_GPU)
plt.scatter(Ns_plot, f_ex_GPU, label="f explizit")
plt.plot(Ns_plot, f_ex_GPU)
plt.xscale('log')
plt.yscale('log')
plt.title("pySDC Allan-Cahn 2D FFT:\nGPU All Times")
plt.xlabel('degrees of freedom')
plt.ylabel('Time in s')
plt.legend()
plt.show()
plt.scatter(Ns_plot, times_CPU, label="time_marching")
plt.plot(Ns_plot, times_CPU)
plt.scatter(Ns_plot, setup_CPU, label="setup")
plt.plot(Ns_plot, setup_CPU)
plt.scatter(Ns_plot, cg_CPU, label="cg")
plt.plot(Ns_plot, cg_CPU)
plt.scatter(Ns_plot, f_im_CPU, label="f implizit")
plt.plot(Ns_plot, f_im_CPU)
plt.scatter(Ns_plot, f_ex_CPU, label="f explizit")
plt.plot(Ns_plot, f_ex_CPU)
plt.xscale('log')
plt.yscale('log')
plt.title("pySDC Allan-Cahn 2D FFT:\nCPU All Times")
plt.xlabel('degrees of freedom')
plt.ylabel('Time in s')
plt.legend()
plt.show()


