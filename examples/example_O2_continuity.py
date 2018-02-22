# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pylab as plt
import time

import cse
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate.quadrature import simps

##########################################################################
#
# O2 dissociation limit continuity as per
#  Allison, Dalgarno, and Pasachoff  Planet Space Sci. 19, 1463-1473 (1971)
#  Fig. 3
#
# Stephen.Gibson@anu.edu.au
# 22 December 2016
##########################################################################

# experimental data ----------------
#  ANU - total photoabsorption cross section 130 - 171 nm
#  doi: 10.1016/0368-2048(96)02910-6
xst = np.loadtxt("data/O2xs-ANU.dat", unpack=True)
xst = (1.0e8/xst[0], xst[1]*1e-19)

#  Harvard - band oscillator strengths v'=1-12
#  doi: 10.1016/0032-0633(83)90085-5
fexp = np.loadtxt("data/O2osc-Yoshino.dat", unpack=True)

# energy ranges for calculation in cm-1
bands = np.array([49357.4, 50044.9, 50710, 51351.5, 51968.4, 52559.6,
                  53122.6, 53655.3, 54156.5, 54622.1, 55051, 55439.5,
                  55784.8, 56085.6, 56340.7, 56551.1, 56720.1, 56852.7,
                  56955.2, 57032.5, 57086.9, 57120.7])

continuum = np.arange(57300, 85000, 100)

# CSE model Schumann-Runge B ^3Sigma_u^- <- X ^3Sigma_g^- single channel ----
O2bands = cse.Xs(mu='O2', VTi=['potentials/X3S-1.dat'], eni=800,
                          VTf=['potentials/B3S-1.dat'],
                     dipolemoment=['transitionmoments/dipole_b_valence.dat'])

# transition energies may also be determined from the next 2 lines 
# O2bands.us.levels()
# bands = O2bands.us.energies - O2bands.gs.cm

lb = len(bands)
vib = np.arange(lb)

# CSE ^3Sigma_u^- valence and Rydbergs coupled channels -------
O2S = cse.Xs(mu='O2', VTi=['potentials/X3S-1.dat'], eni=800,
                      VTf=['potentials/3S-1v.dat',
                           'potentials/3S-1r.dat',
                           'potentials/3S-1r2.dat'],
                      coupf=[4033, 2023, 0],
                      dipolemoment=['transitionmoments/dvX.dat',
                                    'transitionmoments/drX.dat',
                                    'transitionmoments/dr2X.dat'])

# CSE ^3Pi_u coupled channels -------
O2P = cse.Xs(mu='O2', VTi=['potentials/X3S-1.dat'], eni=800,
                      VTf=['potentials/3P1v.dat',
                           'potentials/3P1r.dat',
                           'potentials/3P1r2.dat'],
                      coupf=[7034, 3403, 0],
                      dipolemoment=['transitionmoments/dvPX.dat',
                                    'transitionmoments/drPX.dat',
                                    'transitionmoments/dr2PX.dat'])

# ground X3Sigma_g^- state energy
print(" E(v\"=0, J=0) = {:8.2f} (cm-1)\n".format(O2S.gs.cm))

# (1) band oscillator strengths - uncoupled
print("band oscillator strengths (v', 0), v' = 0-21: ...")
tstart = time.time()
O2bands.calculate_xs(transition_energy=bands)
print("  in {:.1f} seconds\n".format(time.time()-tstart))
print(" v'    fosc     fexpt(Yoshino)")
for v, fosc in enumerate(O2bands.xs):
    if v in fexp[0]:
        print("{:2d}   {:8.2e}     {:8.2e}".format(v, fosc[0], fexp[1, v-1]))
    else:
        print("{:2d}   {:8.2e}".format(v, fosc[0]))

print("\nO2 B-X continuum photodissociation cross section:")
print("  {:5.0f} to {:5.0f} in {:.0f} cm-1 steps ...".
      format(continuum[0], continuum[-1], continuum[1]-continuum[0]))
tstart = time.time()
O2S.calculate_xs(transition_energy=continuum)
print("  in {:.1f} seconds\n".format(time.time()-tstart))

print("\nO2 3Pi-X continuum photodissociation cross section:")
print("  {:5.0f} to {:5.0f} in {:.0f} cm-1 steps ...".
      format(continuum[0], continuum[-1], continuum[1]-continuum[0]))
tstart = time.time()
O2P.calculate_xs(transition_energy=continuum)
print("    in {:.1f} seconds\n".format(time.time()-tstart))

print("calculation complete - see plot")


# evaluate derivative for fosc x dv/dE
fosc = O2bands.xs[:, 0]
spl = InterpolatedUnivariateSpline(bands, vib, k=1)
dvdE = spl.derivative()(bands)

plt.plot(continuum, O2S.xs[:, 0] + 2*O2P.xs[:, 0], 'C0--',
         label='$\sigma$ PyDiatomic')
plt.plot(bands, fosc * dvdE/1.13e12, 'C2+', label=r'$f_{osc}$ PyDiatomic')
plt.plot((57136.2, 57136.2), (1.0e-25, 1.0e-18), 'k--', lw=1)

plt.xlabel(r"wavenumber (cm$^{-1}$)")
plt.ylabel(r"cross section (cm$^{2}$)")
plt.title(r"O$_{2}$ $^3\Pi_u, B{ }^{3}\Sigma_{u}^{-} - X{}^{3}\Sigma_{g}^{-}$")

plt.plot(*xst, 'C1-', label="ANU expt.", zorder=2)

plt.errorbar(bands[1:13], fexp[1]*dvdE[1:13]/1.13e12,
             yerr=fexp[2]*dvdE[1:13]/1.13e12, color='C3', fmt='o', mfc='w',
             ms=4, label='Yoshino', zorder=1)

plt.yscale('log')
plt.annotate(r"$f_{v^{\prime}0} \frac{dv^{\prime}}{dE}/1.13 \times"
             " 10^{12}$", (49000, 3.0e-18), fontsize=12)
plt.annotate(r"$\sigma$", (70000, 5.0e-19), fontsize=12)
plt.legend(loc=8)

plt.savefig("output/example_O2_continuity.png", dpi=100)
plt.show()
