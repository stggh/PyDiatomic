import numpy as np
import cse
import matplotlib.pyplot as plt
import scipy.constants as const

####################################################################
# Rydberg-Klein-Rees evaluation of a potential energy curve
# from spectroscopic constants
#
# Note  RKR calculation is repeated using corrected spectroscopic constants
#  to give ΔGv < 0.1 cm⁻¹ for v" < 35
#
# Stephen.Gibson@anu.edu.au  March 2023
####################################################################

# RKR potential energy curve --------------
def RKR(iso, vv, Gv, Bv, Rgrid, Voo, Te=0, dv=0.1, limb='L'):
    μ = cse.cse_setup.reduced_mass(iso)[0]/const.m_u
    De = Voo - Te  # potential well depth

    R, PEC, vib, RTP, PTP = cse.tools.RKR.rkr(μ, vv, Gv, Bv, De, Voo=Voo,
                                              limb=limb, dv=dv, Rgrid=Rgrid)

    return R, PEC, vib, RTP, PTP  # potential curve, turning points

# CSE diagnostics ---------------------------------------
def levels(iso, fn='potentials/rkrX1S0.dat'):
    X = cse.Cse(iso, VT=[fn])
    X.levels(ntrial=10)

    Gcse, Bcse, Dcse, _ = np.asarray(list(X.results.values())).T

    return Gcse, Bcse, Dcse

def compare(vv, Gv, Bv, Dv, Gcse, Bcse, Dcse, verbose=True):
    offset = 0
    if vv[0] < 0:
        offset = 1
    vint = np.array(vv[offset:], dtype=int)
    Gint = Gv[offset:]
    Bint = Bv[offset:]
    Dint = Dv[offset:]

    dG = Gint - Gcse[vint]
    dB = Bint - Bcse[vint]
    dD = Dint - Dcse[vint]

    if verbose:
        print('\nRKR: Energy levels vs spectroscopic constants')
        print(' v     Gv       ΔGv        Bv   ΔBv(x10⁴)   Dv(x10⁶)  ΔDv(x10⁸)')
        for v in vint:
            print(f'{v:2d} {Gint[v]:10.3f} {dG[v]:6.3f}   '
                  f'{Bint[v]:8.3f} {dB[v]*1e4:6.3f}   '
                  f'{Dint[v]*1e6:8.3f} {dD[v]*1e8:8.3f}')
        print()

    return vint, dG, dB, dD

# plot -----------------------------------------------------
def plot(R, PEC, vdv, RTP, PTP, vint, dG, dB):
    vi = vdv % 1 > 0.99  # integer (whole number) vibrational levels

    fig, ax = plt.subplot_mosaic('''
                                 pg
                                 pb
                                 pd
                                 ''', constrained_layout=True)

    # potential energy curves
    ax['p'].plot(R, PEC, label='RKR potential curve')
    ax['p'].plot(RTP[vi], PTP[vi], 'o', mfc='w', ms=4, label='turning point')
    ax['p'].legend(fontsize='small', labelspacing=0.3)
    ax['p'].axis(xmin=0.7, xmax=4, ymin=PEC.min()-1000, ymax=PEC[-1]+5000)
    ax['p'].ticklabel_format(axis='y', style='sci', scilimits=(4, 4))
    ax['p'].set_xlabel(r'internuclear distance $/$ $\AA$')
    ax['p'].set_ylabel(r'potential energy (rel. to $X$-min.) $/$ cm$^{-1}$')

    # ΔGv
    ax['g'].plot(vint, dG, 'C2')
    ax['g'].set_ylabel(r'$\Delta G_v$ $/$ cm$^{-1}$')
    # ax['g'].ticklabel_format(axis='y', style='sci', scilimits=(-2, -2))
    ax['g'].set_ylim(-0.1, 0.1)
    ax['g'].tick_params('x', labelbottom=False)

    # ΔBv
    ax['b'].plot(vint, dB, 'C3')
    ax['b'].ticklabel_format(axis='y', style='sci', scilimits=(-4, -4))
    ax['b'].set_ylabel(r'$\Delta B_v$ $/$ cm$^{-1}$')
    ax['b'].set_ylim(-3e-4, 5e-5)
    ax['b'].tick_params('x', labelbottom=False)

    # ΔDv
    ax['d'].plot(vint, dD, 'C4')
    ax['d'].set_xlabel('vibrational quantum number')
    ax['d'].set_ylabel(r'$\Delta D_v$ $/$ cm$^{-1}$')
    ax['d'].ticklabel_format(axis='y', style='sci', scilimits=(-8, -8))
    ax['d'].set_ylim(-1e-9, 4e-8)
    ax['d'].set_xlim(0, 35)

    plt.suptitle(r'CO $X^1\Sigma^+$ $-$ Rydberg-Klein-Rees')

    plt.savefig('figures/CO_RKR_Xstate.svg')
    plt.show()

# main ----------------------------------------------
# spectroscopic constants
iso = 'CO'
vv, Gv, Bv, Dv = np.loadtxt('data/vGBD-X.dat', unpack=True)
Dv *= 1e-6

Rgrid = np.arange(0.005, 10.004, 0.005)
Voo = 90674+Gv[0]  # cm⁻¹ Coxon+Hajigeorgiou JPC121 2992 (2004)

# RKR potential energy curve from spectroscopic constants Gv, Bv
print('\nRKR curve - initial calculation ...')
R, PEC, vdv, RTP, PTP = RKR(iso, vv, Gv, Bv, Rgrid, Voo)

# save curve
fn = 'potentials/rkrX1S0.dat'
np.savetxt(fn, np.column_stack((R.T, PEC.T)), fmt='%8.5f %15.8e')

print('\nEvaluate spectroscopic constants for RKR PEC: Cse.levels() ...')
Gcse, Bcse, Dcse = levels(iso, fn) 
vint, dG, dB, dD = compare(vv, Gv, Bv, Dv, Gcse, Bcse, Dcse)

# Repeat RKR calculation correcting spectroscopic constants by first order
# difference: P Pajunen J Chem Phys 92(12) 7484-7487 (1990) 
print('\nPajuen - RKR calculation using corrected spectroscopic constants ...')
Gc = Gv.copy()
Bc = Bv.copy()
Gc += dG  # correction to input Gv
Bc += dB  # Bv

R, PEC, vdv, RTP, PTP = RKR(iso, vv, Gc, Bc, Rgrid, Voo)

# save curve - final
fn = 'potentials/rkrX1S0.dat'
np.savetxt(fn, np.column_stack((R.T, PEC.T)), fmt='%8.5f %15.8e')
print(f'\nRKR PEC written to "{fn}"')

print('\nRe-evaluate spectroscopic constants for new RKR PEC: Cse.levels() ...')
Gcse, Bcse, Dcse = levels(iso, fn) 
print('\nCompare input with calculated ...')
vint, dG, dB, dD = compare(vv, Gv, Bv, Dv, Gcse, Bcse, Dcse)

print('All done! See plot')
plot(R, PEC, vdv, RTP, PTP, vint, dG, dB)
