import numpy as np
from scipy.signal import resample

ratio = 2
nvars = 2048
cnvars = int(nvars / ratio)

xvalues = np.array([i * 1/nvars - 0.5 for i in range(nvars)])
uexf = 0.5 * (1 + np.tanh((0.25 - abs(xvalues)) / (np.sqrt(2) * 0.04)))
xvalues = np.array([i * 1/cnvars - 0.5 for i in range(cnvars)])
uexc = 0.5 * (1 + np.tanh((0.25 - abs(xvalues)) / (np.sqrt(2) * 0.04)))

tmpG = np.fft.rfft(uexc)
tmpF = np.zeros(cnvars + 1, dtype=np.complex128)
halfG = int(cnvars / 2)
tmpF[0: halfG] = tmpG[0: halfG]
tmpF[-1] = tmpG[-1]
uf = np.fft.irfft(tmpF) * ratio

print(np.amax(abs(uf-uexf)))

uf1 = np.fft.irfftn(tmpG, s=[nvars]) * ratio

# uf1 = resample(uexc, nvars)
print(np.amax(abs(uf1-uexf)))


