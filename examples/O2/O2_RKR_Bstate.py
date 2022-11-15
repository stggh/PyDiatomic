import numpy as np
import cse
import matplotlib.pyplot as plt
import scipy.constants as const

# spectroscopic constants
vv, Gv, Bv, Dv = np.loadtxt('data/vGBD-B3S.dat', unpack=True)

μ = cse.cse_setup.reduced_mass('O2')[0]/const.m_u
Voo = 57136.4 + 787.14  # rel. to X-min.
Te = 49792.2738  # from Dunham fit
De = Voo - Te  # potential well depth
dv = 0.1
Rgrid = np.arange(0.005, 10.004, 0.005)

# RKR potential curve --------------
R, PEC, vib, RTP, PTP = cse.tools.RKR.rkr(μ, vv, Gv, Bv, De, Voo=Voo,
                                          limb='L', dv=dv, Rgrid=Rgrid)

# save curve
np.savetxt('potentials/rkrB3S-1.dat', np.column_stack((R.T, PEC.T)),
           fmt='%8.5f %15.8e')

# evaluate eigenstates to compare with spectroscopic constants
print('\nRKR: Energy levels vs spectroscopic constants')
B = cse.Cse('O2', VT=['potentials/rkrB3S-1.dat'])
B.levels(ntrial=30)
vmax = int(list(B.results.keys())[-1]) + 2
vv = np.array(vv[1:vmax], dtype=int)
Gv = Gv[1:vmax]
Bv = Bv[1:vmax]
Dv = Dv[1:vmax]

# CSE -----------------------------------------------------
Gcse, Bcse, Dcse, _ = np.asarray(list(B.results.values())).T
dG = Gv - Gcse[vv]
dB = (Bv - Bcse[vv])*1e3
dD = (Dv - Dcse[vv]*1e6)

print(' v      Gv       ΔGv        Bv   ΔBv(x10³)   Dv(x10⁶)  ΔDv(x10⁶)')
for v in vv:
    print(f'{v:2d} {Gv[v]:11.2f} {dG[v]:6.2f}   {Bv[v]:8.2f} {dB[v]:6.2f}   '
            f'{Dv[v]:8.2f} {dD[v]:9.2f}')
print()

# plot -----------------------------------------------------
vint = np.mod(vib, 1) > 0.99  # integer levels

fig, ax = plt.subplot_mosaic('''
                             pg
                             pb
                             pd
                             ''', constrained_layout=True)

ax['p'].plot(R, PEC, label='RKR potential curve')
ax['p'].plot(RTP[vint], PTP[vint], 'o', mfc='w', ms=4, label='turning point')
ax['p'].legend()
ax['p'].axis(xmin=1.1, xmax=4, ymin=PEC.min()-1000, ymax=PEC[-1]+5000)
ax['p'].ticklabel_format(axis='y', style='sci', scilimits=(4, 4))
ax['p'].set_xlabel(r'internuclear distance $/$ $\AA$')
ax['p'].set_ylabel(r'potential energy (rel. to $X$-min.) $/$ cm$^{-1}$')

ax['g'].plot(vv, dG, 'C2')
ax['g'].set_ylabel(r'$\Delta G_v$ $/$ cm$^{-1}$')
ax['g'].set_ylim(-0.5, 1)
ax['g'].tick_params('x', labelbottom=False)

ax['b'].plot(vv, dB, 'C3')
ax['b'].set_ylabel(r'$\Delta B_v$ $/$ $10^{-3}$ cm$^{-1}$')
ax['b'].set_ylim(-0.6, 0.7)
ax['b'].tick_params('x', labelbottom=False)

ax['d'].plot(vv, dD, 'C4')
ax['d'].set_ylabel(r'$\Delta D_v$ $/$ $10^{-6}$ cm$^{-1}$')
ax['d'].set_xlabel(r'$v$')
ax['d'].set_ylim(-0.5, 0.2)

plt.suptitle(r'O$_2$ $B^3\Sigma_u^-$')

plt.savefig('figures/O2_RKR_Bstate.svg')
plt.show()
