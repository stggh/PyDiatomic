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
limit = xsO1D > 7e-19

wn = wavenumber[limit]
expt = xsO1D[limit]

# cse model ---------------------------------------------------
X = cse.Cse('O2', VT=['potentials/X3S-1.dat'], en=800)
EB = cse.Cse('O2', dirpath='potentials', suffix='.dat',
             VT=['B3S-1', 'E3S-1'], coup=[4000])
EB.VT[1, 1] -= 200/8065.541
EBX = cse.Transition(EB, X, dipolemoment=[1, 0])

# handy variables
lb0 = EB.statelabel[0]  # state label string (unicode) B³Σ₁⁻
lb1 = EB.statelabel[1]  # E³Σ₁⁻
evcm = X._evcm  # conversion eV to cm⁻¹
R = EB.R  # common internuclear distance grid

# PEC(s) - paramaterise from fitted Wei analytical curves
print(f'Wei fit to {lb1}')
subR = np.logical_and(R > 0.98, R < 1.31)
Efit = cse.tools.analytical.Wei_fit(R[subR], EB.VT[1, 1][subR]*evcm,
                                    voo=EB.VT[1, 1, -1]*evcm,
                                    adjust=['re', 'De', 'b', 'h'])
print(Efit.fitstr)

# least-squares fit -----------------------------------------------
t0 = time.time()
fit = cse.tools.model_fit.Model_fit(EBX,
          data2fit={lb0:{'xs':(wn, expt)}
                    #lb1:{'position': 82945}},
                    },
          VT_adj={#lb1:{'ΔV':-1000},
                  # lb1:{'ΔR':(0.1, -0.5, 0.5)}},
                  lb0:{'spline':np.arange(0.9, 1.4, 0.1)},
                  lb1:{'Wei':Efit.paramdict | {'Rm':0.98, 'Rn':1.31}}},
          coup_adj={lb1+'<->'+lb0:1},
          etdm_adj={lb0+'<-'+X.statelabel[0]:0.5},
          verbose=False)

dt = time.time() - t0
mins, secs = dt // 60, dt % 60

print(f'\nCalculation time: {int(mins):d} minutes and {int(secs)} seconds\n')

# plot -----------------------------------------------------------
fit.residual(fit.result.x)

fig, (axp, axx) = plt.subplots(1, 2, sharey=True)
axx.plot(xsO1D, wavenumber, 'C2', label=r'expt. O($^1D_2$)')
try:
    axx.plot(fit.csexs, wn, 'C3o', ms=2, label='cse')
except:
    pass
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
