import numpy as np
import cse
from scipy.interpolate import splrep, splev
import matplotlib.pyplot as plt
import time

# experimental data - O₂ photodissociation cross section yielding O(¹D₂)
wavelength, xsO1D, xsO1Dstderr = np.loadtxt('data/ANU/xsf.dat',  unpack=True)
wavenumber = 1e8/wavelength  # wavelength convert Å to cm⁻¹
wavenumber, xsO1D = wavenumber[::-1], xsO1D[::-1]*1e-19  # reverse order

# limit experimental data for fit, to regions of significant cross section
limit = xsO1D > 10e-19

'''
# interpolate experimental data on a smaller grid
spl = splrep(wavenumber[limit], xsO1D[limit])
dw = 1
wn = np.arange(int(wavenumber[limit][0]/dw)*dw,
               int(wavenumber[limit][-1]/dw)*dw, dw)
expt = splev(wn, spl)
'''

wn = wavenumber[limit]
expt = xsO1D[limit]

# cse model ---------------------------------------------------
X = cse.Cse('O2', VT=['potentials/X3S-1.dat'], en=800)
EB = cse.Cse('O2', dirpath='potentials', suffix='.dat',
             VT=['B3S-1', 'E3S-1'], coup=[4000])
EBX = cse.Transition(EB, X, dipolemoment=[1, 0])

evcm = X._evcm
R = EB.R

# PEC analytical
'''
subR = np.logical_and(R > 1.3, R < 2.8)
Bfit = cse.tools.analytical.Wei_fit(R[subR], EB.VT[0, 0][subR]*evcm,
                                    voo=EB.VT[0, 0, -1]*evcm,
                                    adjust=['re', 'De', 'b', 'h'])
Bp = Bfit.paramdict
'''
subR = np.logical_and(R > 0.98, R < 1.31)
Efit = cse.tools.analytical.Wei_fit(R[subR], EB.VT[1, 1][subR]*evcm,
                                    voo=EB.VT[1, 1, -1]*evcm,
                                    adjust=['re', 'De', 'b', 'h'])
Ep = Efit.paramdict

# least-squares fit -----------------------------------------------
lb0 = EB.statelabel[0]
lb1 = EB.statelabel[1]

t0 = time.time()
fit = cse.tools.model_fit.Model_fit(EBX,
          data2fit={lb0:{'xs':(wn, expt)}}, # , lb1:{'peak': 80415}},
          VT_adj={# lb1:{'ΔV':(-50, -60, -40)},
                  # lb1:{'ΔR':(0.1, -0.5, 0.5)}},
                  lb1:{'Wei':Ep | {'Rm':0.98, 'Rn':1.31}},
                  lb0:{'spline':np.arange(0.9, 1.4, 0.1)}},
          coup_adj={lb1+'<->'+lb0:(1, 0.7, 1.3)},
          etdm_adj={lb0+'<-'+X.statelabel[0]:(0.5, 0.1, 1.5)},
          verbose=False)

dt = time.time() - t0
mins, secs = dt // 60, dt % 60

print(f'\nCalculation time: {int(mins):d} minutes and {int(secs)} seconds\n')

# plot -----------------------------------------------------------
fit.residual(fit.result.x)

fig, (axp, axx) = plt.subplots(1, 2, sharey=True)
axx.plot(xsO1D, wavenumber, 'C2', label=r'expt. O($^1D_2$)')
axx.plot(fit.csexs, wn, 'C3o', ms=2, label='cse')
axx.set_xlabel(r'photodissociation cross section (cm$^2$)')
axx.set_ylim(49000, 85000)
axx.legend()

for i, p in enumerate(fit.csemodel.us.statelabel):
    VT = fit.csemodel.us.VT[i, i]*evcm - X.cm
    axp.plot(R, VT, label=p)

axp.legend()
axp.set_xlabel(r'internuclear distance ($\AA$)')
axp.set_ylabel(r'wavenumber (cm$^{-1}$)')
axp.set_xlim(0.8, 2.9)

plt.suptitle(r'O$_2$ Schumann-Runge continuum $EB^3\Sigma_u^- $'
             r'$\leftarrow X^3\Sigma_g^-$')
plt.tight_layout(h_pad=0.2)

plt.savefig('figures/O2_SRC_EBX_O1D_fit.svg')
plt.show()
