import numpy as np
import cse
import matplotlib.pyplot as plt
import scipy.constants as const

####################################################################
# Rydberg-Klein-Rees evaluation of a potential energy curve
# from spectroscopic constants
#
# Note here RKR is repeated using corrected spectroscopic constants
# ΔGv < 0.1 cm⁻¹ for v" < 24
#
# Stephen.Gibson@anu.edu.au  2022
####################################################################

# RKR potential energy curve --------------
def RKR(iso, vv, Gv, Bv, Rgrid, Voo, Te=0, dv=0.1, limb='L'):
    μ = cse.cse_setup.reduced_mass(iso)[0]/const.m_u
    De = Voo - Te  # potential well depth

    R, PEC, vib, RTP, PTP = cse.tools.RKR.rkr(μ, vv, Gv, Bv, De, Voo=Voo,
                                              limb=limb, dv=dv, Rgrid=Rgrid)

    return R, PEC, vib, RTP, PTP  # potential curve, turning points

# CSE diagnostics ---------------------------------------
def levels(iso, fn='potentials/rkrX3S-1.dat'):
    X = cse.Cse(iso, VT=[fn])
    X.levels(ntrial=30)

    Gcse, Bcse, Dcse, _ = np.asarray(list(X.results.values())).T

    return Gcse, Bcse, Dcse

def compare(vv, Gv, Bv, Gcse, Bcse, Dcse, verbose=True):
    offset = 0
    if vv[0] < 0:
        offset = 1
    vint = np.array(vv[offset:], dtype=int)
    Gint = Gv[offset:]
    Bint = Bv[offset:]

    dG = Gint - Gcse[vint]
    dB = Bint - Bcse[vint]

    if verbose:
        print('\nRKR: Energy levels vs spectroscopic constants')
        print(' v     Gv       ΔGv        Bv   ΔBv(x10⁴)   Dv(x10⁶)')
        for v in vint:
            print(f'{v:2d} {Gint[v]:10.2f} {dG[v]:6.2f}   '
                  f'{Bint[v]:8.2f} {dB[v]*1e4:6.2f}   '
                  f'{Dcse[v]*1e6:8.2f}')
        print()

    return vint, dG, dB

# plot -----------------------------------------------------
def plot(R, PEC, vdv, RTP, PTP, vint, dG, dB):
    vi = vdv % 1 > 0.99  # integer (whole number) vibrational levels

    fig, ax = plt.subplot_mosaic('''
                                 pg
                                 pb
                                 ''', constrained_layout=True)

    # potential energy curves
    ax['p'].plot(R, PEC, label='RKR potential curve')
    ax['p'].plot(RTP[vi], PTP[vi], 'o', mfc='w', ms=4, label='turning point')
    ax['p'].legend(fontsize='small', labelspacing=0.3)
    ax['p'].axis(xmin=0.8, xmax=3, ymin=PEC.min()-1000, ymax=PEC[-1]+5000)
    ax['p'].ticklabel_format(axis='y', style='sci', scilimits=(4, 4))
    ax['p'].set_xlabel(r'internuclear distance $/$ $\AA$')
    ax['p'].set_ylabel(r'potential energy (rel. to $X$-min.) $/$ cm$^{-1}$')

    # ΔGv
    ax['g'].plot(vint, dG, 'C2')
    ax['g'].set_ylabel(r'$\Delta G_v$ $/$ cm$^{-1}$')
    # ax['g'].ticklabel_format(axis='y', style='sci', scilimits=(-2, -2))
    ax['g'].set_ylim(-0.3, 0.3)
    ax['g'].tick_params('x', labelbottom=False)

    # ΔBv
    ax['b'].plot(vint, dB, 'C3')
    ax['b'].set_xlabel('vibrational quantum number')
    ax['b'].set_ylabel(r'$\Delta B_v$ $/$ cm$^{-1}$')
    ax['b'].ticklabel_format(axis='y', style='sci', scilimits=(-4, -4))
    ax['b'].set_ylim(-2e-4, 5e-4)

    plt.suptitle(r'O$_2$ $X^3\Sigma_g^-$')

    plt.savefig('figures/O2_RKR_Xstate.svg')
    plt.show()

# main ----------------------------------------------
# spectroscopic constants
iso = 'O2'
vv, Gv, Bv = np.loadtxt('data/SRI/vGB-X3S-Cosby.dat', unpack=True)

Rgrid = np.arange(0.005, 10.004, 0.005)
Voo = 57136.4-15867.862+787.14

# RKR potential energy curve from spectroscopic constants Gv, Bv
R, PEC, vdv, RTP, PTP = RKR(iso, vv, Gv, Bv, Rgrid, Voo)

# save curve
fn = 'potentials/rkrX3S-1.dat'
np.savetxt(fn, np.column_stack((R.T, PEC.T)), fmt='%8.5f %15.8e')

# evaluate spectroscopic constanst from the potential energy curve
Gcse, Bcse, Dcse = levels(iso, fn) 
vint, dG, dB = compare(vv, Gv, Bv, Gcse, Bcse, Dcse)

# Repeat RKR calculation correcting spectroscopic constants by first order
# difference: P Pajunen J Chem Phys 92(12) 7484-7487 (1990) 
Gc = Gv.copy()
Bc = Bv.copy()
Gc[1:] += dG
Bc[1:] += dB

print('RKR PEC from corrected input Gv, Bv values')
R, PEC, vdv, RTP, PTP = RKR(iso, vv, Gc, Bc, Rgrid, Voo)

# save curve - final
fn = 'potentials/rkrX3S-1.dat'
np.savetxt(fn, np.column_stack((R.T, PEC.T)), fmt='%8.5f %15.8e')

Gcse, Bcse, Dcse = levels(iso, fn) 
vint, dG, dB = compare(vv, Gv, Bv, Gcse, Bcse, Dcse)

plot(R, PEC, vdv, RTP, PTP, vint, dG, dB)