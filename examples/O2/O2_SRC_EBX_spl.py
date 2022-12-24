import numpy as np
import cse
import matplotlib.pylab as plt
from cse.tools import analytical
from scipy.optimize import least_squares
from scipy.interpolate import splrep, splev
import time

#########################################################################
#
# O2_SRC_EBX_spl.py 
#
#  Simple ³Σ⁻ Rydberg - ³Σ⁻ valence interaction 
#
#  Potential energy curves optimised around crossing region
#  to fit experimental photodissociation cross section yielding O(¹D₂)
#
# Stephen.Gibson@anu.edu.au - December 2022
#
#########################################################################

def cse_model(coup=4000, Rx=1.18):
    # O₂ ground X³Σ⁻ state
    X = cse.Cse('O2', VT=['potentials/X3S-1.dat'], en=800)

    # O₂ excited E-B³Σ⁻ coupled-state
    EB = cse.Cse('O2', dirpath='potentials', suffix='.dat',
                 VT=['B3S-1', 'E3S-1'], coup=[coup*np.exp(-(X.R-Rx)**2)])
    return EB, X

def residual(pars, wavenumber, xsO1D, R, X, Borig, Eorig, knots, residual=True):
    # scale PECs by spline evaluated at the parameter knots
    B, E = Borig.copy(), Eorig.copy()

    coupl, edtm, yshft = pars[-3:]
    rk = np.logical_and(R >= knots[0], R <= knots[-1])
    parsB, parsE = np.split(pars[:-3], 2)

    indices = np.searchsorted(R[rk], knots)
    splB = splrep(R[indices], parsB)
    splE = splrep(R[indices], parsE)

    B[rk] *= splev(R[rk], splB)
    E[rk] *= splev(R[rk], splE)
    E[rk] -= (yshft - 1)

    # Rydberg-valence potential energy curve crossing point
    ri = R < R[rk][-1]
    Rx = R[np.abs(B[ri] - E[ri]).argmin()]

    # ³Σ⁻-valence state coupled to ³Σ⁻-Rydberg
    EB = cse.Cse('O2', VT=[(R, B), (R, E)], coup=[coupl*np.exp(-(R-Rx)**2)])

    EBX = cse.Transition(EB, X, dipolemoment=[edtm, 0],
                         transition_energy=wavenumber)

    if residual:
        return EBX.xs[:, 0]*1e19 - xsO1D
    else:
        return EBX.xs[:, 0]*1e19, B, E

# main ============================================================
# experimental data - O₂ photodissociation cross section yielding O(¹D₂)
wavelength, xsO1D, xsO1Dstderr = np.loadtxt('data/ANU/xsf.dat',  unpack=True)

wavenumber = 1e8/wavelength  # wavelength convert Å to cm⁻¹
wavenumber, xsO1D = wavenumber[::-1], xsO1D[::-1]  # reverse order

# limit experimental data for fit, to regions of significant cross section
limit = xsO1D >  xsO1D[wavenumber < 73000].max()*1/2

EB, X = cse_model()
R, (B, E) = EB.R, np.diagonal(EB.VT).T*EB._evcm  # PECs in cm⁻¹
Borig, Eorig = B.copy(), E.copy()

# R-values for scaling factors
knots = np.arange(0.9, 1.4, 0.1)
pars = np.ones(len(knots)*2, dtype=float)
pars = np.concatenate((pars, [1500, 0.9, 1]))  # B, E, coupling, EDTM, Eshift

# least squares fit - adjusting PECs, EDTM, and coupling
t0 = time.time()
result = least_squares(residual, pars, method='lm', x_scale='jac',
                       args=(wavenumber, xsO1D, R, X, Borig, Eorig, knots))
t1 = time.time()

result.stderr = analytical.fiterrors(result)

dt = t1 - t0
print(f'fit in {int(dt // 60):d} mins  {int(dt % 60):d} sec')
print(result.message)
print(result.cost)
print(result.x)
print(result.stderr)

xsfit, B, E = residual(result.x, wavenumber, xsO1D, R, X, Borig, Eorig, knots,
                       False)

rk = np.logical_and(R >= knots[0], R <= knots[-1])
indices = np.searchsorted(R[rk], knots)

# plot ---------------------------------
plt.rcParams['ytick.right'] = plt.rcParams['ytick.labelright'] = True

fig, (ap, ax) = plt.subplots(1, 2, sharey=True)

# cross section
ax.semilogx(xsfit*1e-19, wavenumber, label='PyDiatomic')
ax.semilogx(xsO1D*1e-19, wavenumber, label='ANU expt.')
ax.set_xlim(1e-20, 5e-16)
ax.legend(fontsize='small', labelspacing=0.3)
ax.yaxis.set_label_position('right')
ax.yaxis.tick_right()
ax.ticklabel_format(axis='y', style='sci', scilimits=(4, 4))
ax.set_xlabel(r'cross section (cm$^2$)')
ax.set_ylabel(r'wavenumber (cm$^{-1}$)')

# potential energy curves
ap.plot(R, Borig-X.cm, '--', label=r'valence $^3\Sigma^-_1$')
ap.plot(R, Eorig-X.cm, '--', label=r'Rygberg $^3\Sigma^-_1$')
ap.plot(R, B-X.cm, '-', label='valence fit to $\sigma$')
ap.plot(R, E-X.cm, '-', label='Rydberg fit to $\sigma$xs.')
ap.legend(fontsize='small', labelspacing=0.3)
ap.axis([0.8, 2.5, wavenumber[0]-10000, wavenumber[-1]+5000])
ap.set_xlabel(r'internuclear distance ($\AA$)')
ap.set_ylabel(r'potential energy above $X_{v=0, J=0}$ (cm$^{-1}$)')

plt.suptitle(r'O$_2$ photodissociation yielding O($^1D_2$)')
plt.tight_layout(w_pad=0)
plt.savefig('figures/O2_SRC_EBX_fit.svg')
plt.show()
