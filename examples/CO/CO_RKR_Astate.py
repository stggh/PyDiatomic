import numpy as np
import cse
import matplotlib.pyplot as plt
import scipy.constants as const

# spectroscopic constants
vv, Gv, Bv = np.loadtxt('data/vGB-A1P.dat', unpack=True)

iso = 'CO'
μ = cse.cse_setup.reduced_mass(iso)[0]/const.m_u
Voo = 90674 # rel. to X-min.
Te = 63992.2875 # from Dunham fit
De = Voo - Te  # potential well depth
dv = 0.1
Rgrid = np.arange(0.005, 10.004, 0.005)

# RKR potential curve --------------
R, PEC, vib, RTP, PTP = cse.tools.RKR.rkr(μ, vv, Gv, Bv, De, Voo=Voo,
                                          limb='L', dv=dv, Rgrid=Rgrid)

# save curve
np.savetxt('potentials/rkrA1P1.dat', np.column_stack((R.T, PEC.T)),
           fmt='%8.5f %15.8e')

# evaluate eigenstates to compare with spectroscopic constants
print('\nRKR: Energy levels vs spectroscopic constants')
A = cse.Cse(iso, VT=['potentials/rkrA1P1.dat'])
A.levels(ntrial=10)
vmax = min(int(list(A.results.keys())[-1]), int(vv[-1]))
vv = np.array(vv[1:vmax], dtype=int)
Gv = Gv[1:vmax]
Bv = Bv[1:vmax]

# CSE -----------------------------------------------------
Gcse, Bcse, Dcse, _ = np.asarray(list(A.results.values())).T
dG = Gv - Gcse[vv]
dB = (Bv - Bcse[vv])*1e3

print(' v      Gv       ΔGv        Bv   ΔBv(x10³)   Dv(x10⁶)')
for v in vv[:-1]:
    print(f'{v:2d} {Gv[v]:11.2f} {dG[v]:6.2f}   {Bv[v]:8.2f} {dB[v]:6.2f}   '
            f'{Dcse[v]*1e6:8.2f}')
print()

# plot -----------------------------------------------------
vint = np.mod(vib, 1) > 0.99  # integer levels

fig, ax = plt.subplot_mosaic('''
                             pg
                             pb
                             ''', constrained_layout=True)

ax['p'].plot(R, PEC, label='RKR potential curve')
ax['p'].plot(RTP[vint], PTP[vint], 'o', mfc='w', ms=4, label='turning point')
ax['p'].legend()
ax['p'].axis(xmin=0.8, xmax=4, ymin=PEC.min()-1000, ymax=PEC[-1]+5000)
ax['p'].ticklabel_format(axis='y', style='sci', scilimits=(4, 4))
ax['p'].set_xlabel(r'internuclear distance $/$ $\AA$')
ax['p'].set_ylabel(r'potential energy (rel. to $X$-min.) $/$ cm$^{-1}$')

ax['g'].plot(vv, dG, 'C2')
ax['g'].set_ylabel(r'$\Delta G_v$ $/$ cm$^{-1}$')
ax['g'].set_ylim(0, 3)
ax['g'].tick_params('x', labelbottom=False)

ax['b'].plot(vv, dB, 'C3')
ax['b'].set_ylabel(r'$\Delta B_v$ $/$ cm$^{-1}$')
ax['b'].set_xlabel('vibrational quantum number')
ax['b'].set_ylim(-2e-1, 6e-1)
ax['b'].ticklabel_format(axis='y', style='sci', scilimits=(-1, -1))

plt.suptitle(r'CO $A^1\Pi$ $-$ Rydberg-Klein-Rees')

plt.savefig('figures/CO_RKR_Astate.svg')
plt.show()
