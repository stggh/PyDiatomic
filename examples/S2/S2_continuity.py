# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pylab as plt
import time

import cse
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d
from scipy.integrate import simps, quad

##########################################################################
# 
# S2 dissociation limit continuity as per 
#  Allison, Dalgarno, and Pasachoff  Planet Space Sci. 19, 1463-1473 (1971)
#  Fig. 3
#
# Stephen.Gibson@anu.edu.au
# 24 December 2016
##########################################################################

# B-X ------------------
continuum = np.arange(44477, 50000, 100)
bands = np.array([31675, 32102, 32526, 32945, 33358, 33768, 34173, 34567,
          34957, 35345, 35728, 36108, 36477, 36835, 37185, 37534,
          37877, 38210, 38537, 38862, 39187, 39498, 39801, 40098,
          40382, 40659, 40923, 41178, 41422, 41656, 41879, 42092,
          42294, 42486, 42667, 42838, 42998, 43149, 43289, 43421,
          43542, 43655, 43758, 43853, 43940, 44018, 44089, 44152,
          44208, 44258, 44301, 44338, 44370, 44436])

transition_energies = np.append(bands, continuum)

print(r"S2 B-X continuum")

lb = len(bands)
v = np.arange(lb)

S2 = cse.Xs('S2', VTi=['S2/X/X3S-1.dat'], eni=400,
                     VTf=['S2/B/B3S-1.dat'],
                     dipolemoment=['S2/abinitio/DBXA.dat'])

tstart = time.time()
S2.calculate_xs(transition_energy=transition_energies)
tend = time.time()
print("    in {:.1f} seconds\n".format(tend-tstart))
print(" E(v\"=0) = {:8.2f} (cm-1)\n".format(S2.gs.cm))

fosc = S2.xs[:lb, 0]

spl = InterpolatedUnivariateSpline(bands, v, k=1)
dvdEB = spl.derivative()(bands)

plt.semilogy(bands, fosc*dvdEB/1.13e12, '+', color='C0')
plt.semilogy(continuum, S2.xs[lb:], color='C0', label=r'$B ^3\Sigma_u^-$')
plt.semilogy((44476.3, 44476.3), (1.0e-22, 2.0e-16), '--', color='C0',
             lw=1)
plt.xlabel(r"wavenumber (cm$^{-1}$)")
plt.ylabel(r"cross section (cm$^{2}$)")
plt.title(r"S$_{2}$  $B{ }^{3}\Sigma_{u}^{-}, B^{\prime\prime}{}^{3}\Pi_{u} - X{}^{3}\Sigma_{g}^{-}$")

plt.annotate(r"$f_{v^{\prime}0} \frac{dv^{\prime}}{dE}/1.13 \times"
             " 10^{12}$", (32000, 1.2e-25), fontsize=12)
plt.annotate(r"$\sigma(^{3}\Sigma)$", (46500, 1.0e-19),
             color='#1f77b4', fontsize=12)

# B"-X ------------------

continuum = np.arange(35815, 50000, 100)

bands = np.array([31075, 31404, 31725, 32038, 32343, 32641, 32931, 33213, 33485,
          33746, 34000, 34241, 34461, 34660, 34842, 35008, 35156, 35286,
          35401, 35499, 35581, 35649, 35703, 35744, 35774, 35794])

transition_energies = np.append(bands, continuum)

S2 = cse.Xs('S2', VTi=['S2/X/X3S-1.dat'], eni=400,
                     VTf=['S2/P/Bpp3P1.dat'],
                     dipolemoment=['S2/abinitio/DBppXA.dat'])

print(r"S2 Bpp-X continuum")
lb = len(bands)
v = np.arange(lb)

tstart = time.time()
S2.calculate_xs(transition_energy=transition_energies)
tend = time.time()
print("    in {:.1f} seconds\n".format(tend-tstart))
print(" E(v\"=0) = {:8.2f} (cm-1)\n".format(S2.gs.cm))


fosc = S2.xs[:lb, 0]

spl = InterpolatedUnivariateSpline(bands, v, k=1)

dvdE = spl.derivative()(bands)

plt.semilogy(bands, fosc*dvdE/1.13e12, 'o', color='C3', mfc='none', ms=4)
plt.semilogy(continuum, S2.xs[lb:], color='C3', label=r"$B^{\prime\prime}{}^{3}\Pi_{u}$")
plt.semilogy((35812.63, 35812.63), (1.0e-22, 2.0e-16), '--', color='C3',
             lw=1)
plt.annotate(r"$\sigma(^{3}\Pi)$", (41500, 2.0e-23),
             color='C3', fontsize=12)

#---------- B-B"-X -----------------------
continuum = np.arange(35815, 40000, 1)

S2 = cse.Xs('S2', VTi=['S2/X/X3S-1.dat'], eni=400,
                     VTf=['S2/B/B_3S-1_temp14.dat', 'S2/P/Bpp3P1.dat',
                          'S2/C/C3S-1.dat', 'S2/D/D3P1.dat'],
                     coupf=[60, 4000, 0, 0, 0, 7000],
                     dipolemoment=['S2/abinitio/DBXA.dat',
                                   'S2/abinitio/DBXA.dat',
                                   0, 0])

print()
print(r"S2 B-Bpp-X continuum {:d} to {:d}, step {:d} (cm-1) ..."
      .format(continuum[0], continuum[-1], continuum[1]-continuum[0]))
      
tstart = time.time()
S2.calculate_xs(transition_energy=continuum)
tend = time.time()
print("    in {:.1f} seconds\n".format(tend-tstart))

totalxs = S2.xs[:, 0] # + 2*S2.xs[:, 1]
np.savetxt("S2total.dat", np.column_stack((continuum, totalxs)))

subE = totalxs > 2.0e-19
plt.semilogy(continuum[subE], totalxs[subE], color='C8', label=r'$B-B^{\prime\prime}$')
plt.annotate(r"$\sigma(^{3}\Sigma\leftrightarrow^{3}\Pi)$", (40000, 4.0e-25),
             color='C8', fontsize=12)

xs370Kdis = np.loadtxt("xs_370K_discrete.dat", unpack=True)
xs370Kpre = np.loadtxt("xs_370K_prediss.dat", unpack=True)

bandlimits = [33900, 34254, 34673, 35050, 35400]

v = []
f = []
#ef = []

# eBE < bandlimits[0]
Er = xs370Kdis[0] < bandlimits[0]
emin = xs370Kdis[0][Er][0]
emax = xs370Kdis[0][Er][-1]

v.append((emin+emax)/2)
# func = interp1d(xs370Kdis[0][Er], xs370Kdis[1][Er])
# fq, efq = quad(func, xs370Kdis[0][Er][0], xs370Kdis[0][Er][-1])
fq = simps(xs370Kdis[1][Er], xs370Kdis[0][Er])
f.append(fq)

emin = emax
for emax in bandlimits[1:]:
    Er = np.logical_and(xs370Kdis[0] >= emin, xs370Kdis[0] < emax)
    v.append((emin+emax)/2)
    f.append(simps(xs370Kdis[1][Er], xs370Kdis[0][Er]))
    # func = interp1d(xs370Kdis[0][Er], xs370Kdis[1][Er])
    # fq, efq = quad(func, xs370Kdis[0][Er][0], xs370Kdis[0][Er][-1])
    emin = emax

Er = xs370Kdis[0] >= bandlimits[-1]
emax = xs370Kdis[0][Er][-1]
v.append((emin+emax)/2)
f.append(simps(xs370Kdis[1][Er], xs370Kdis[0][Er]))

v = np.array(v)
fexp = np.array(f)
for i, vi in enumerate(v):
    print(' {:d}  {:8.3f}  {:5.2g}  {:5.2g}'\
          .format(i+5, vi, fexp[i]*1.0e4, dvdE[i]))

fexp *= 18.9e-4*dvdEB[5:11]/fexp[0]/1.13e12

subE = xs370Kdis[1] > 5.0e-18

# plt.semilogy(xs370Kdis[0][subE], xs370Kdis[1][subE], 'C1', label=r'Stark')
plt.plot(v, fexp, 'C4o', label=r'$f_{osc}\frac{dv^\prime}{dE}$ Stark')
plt.semilogy(*xs370Kpre, 'C1', label=r'Stark')
plt.legend(fontsize='smaller', labelspacing=0.1)


plt.savefig("S2.svg")
plt.show()
