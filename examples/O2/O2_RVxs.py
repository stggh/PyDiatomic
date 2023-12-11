import numpy as np
import cse
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev
from scipy.integrate import simpson

# CSE model ------------------------------------------------
O2X = cse.Cse('O2', VT=['potentials/X3S-1.dat'], en=800)
O2BE = cse.Cse('O2', VT=['B3S-1', 'E3S-1'], coup=[4000],
               dirpath='potentials', suffix='.dat')
O2BE.VT[1, 1] -= O2X.cm/O2X._evcm

O2BEX = cse.Transition(O2BE, O2X, dipolemoment=[1, 0],
                       transition_energy=np.arange(57550, 90000, dn:=100))
xs = O2BEX.xs[:, 0]  # '0' is 'B3S-1.dat' channel
wn = O2BEX.wavenumber

# ANU experimental data ------------------------------------
wavl1D, xs1D, exs1D = np.loadtxt('data/ANU/xsf.dat', unpack=True)
xs1D *= 1e-19
wn1D = 1e8/wavl1D

# convolution of calculation to match instrument resolution FWHM ~ 2 Å
sigma = 100
gx = np.arange(-3*sigma, 3*sigma, dn)
gaussian = np.exp(-(gx/sigma)**2/2)/sigma/np.sqrt(2*np.pi)

convy = np.convolve(xs, gaussian, mode='same')

yarea = simpson(xs, wn)
conarea = simpson(convy, wn)
convy *= (yarea/conarea)


# plot -----------------------------------------------------
plt.plot(wn, convy)
plt.plot(wn1D, xs1D)
plt.xlabel('Wavenumber (cm$^{-1}$)')
plt.ylabel('Cross section (cm$^{2}$)')
plt.ticklabel_format(axis='y', style='sci', scilimits=(-19, -19))
plt.title(r'O$_2$ $^3\Sigma_u^-$ Rydberg-valence interaction')

plt.savefig('figures/O2_RVxs.svg')
plt.show()
