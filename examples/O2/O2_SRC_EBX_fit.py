import numpy as np
import cse
import matplotlib.pylab as plt
from cse.tools import analytical
from scipy.optimize import least_squares

#########################################################################
#
# O2_SRC_EBX_fit.py 
#  A simple ³Σ⁻ Rydberg - ³Σ⁻ valence interaction 
#
#  Adjust potential curves around crossing region
#  to fit the experimental photodissociation cross section
#
# Stephen.Gibson@anu.edu.au - December 2022
#
#########################################################################

def cse_model(coup=4000, Rx=1.18):
    # O₂ ground X³Σ⁻ state
    X = cse.Cse('O2', VT=['potentials/X3S-1.dat'], en=800)
    evcm = X._evcm  # conversion eV to cm⁻¹

    # O₂ excited (B)³Σ⁻ state
    EB = cse.Cse('O2', dirpath='potentials', suffix='.dat',
                 VT=['B3S-1', 'E3S-1'], coup=[coup*np.exp(-(X.R-Rx)**2)])
    return EB, X

def initial_PEC_parameters(R, B, E):
    # determine Wei analytical curve parametes for PECs near crossing

    # ³Σ⁻-valence state inner limb (adiabatic "B-state")
    rB = np.logical_and(R > 0.95, R < 1.25)
    re = R[B.argmin()]
    voo = B[-1]
    De = voo - B.min()
    Bfit = analytical.Wei_fit(R[rB], B[rB], re=re, voo=voo, De=De)

    # ³Σ⁻-Rydberg state potential well (adiabatic "E-state")
    rE = np.logical_and(R > 0.87, R < 1.30)
    Efit = analytical.Wei_fit(R[rE], E[rE], voo=E[-1])

    pars = np.append(list(Bfit.paramdict.values()),
                     list(Efit.paramdict.values()))
    return pars, Bfit, Efit, rB, rE

def residual(pars, wavenumber, xsO1D, R, B, E, Bfit, Efit, rB, rE):
    coupl = pars[-1]
    edm = pars[-2]

    Bpar, Epar = np.split(pars[:10], 2)
    # insert Wei curves into B, E PECs
    B[rB] = analytical.Wei(R[rB], *Bpar)
    E[rE] = analytical.Wei(R[rE], *Epar)
    ri = R < R[rB][-1]
    Rx = R[np.abs(B[ri] - E[ri]).argmin()]

    # ³Σ⁻-valence state coupled to ³Σ⁻-Rydberg
    EB = cse.Cse('O2', VT=[(R, B), (R, E)], coup=[coupl*np.exp(-(R-Rx)**2)])

    EBX = cse.Transition(EB, X, dipolemoment=[edm, 0],
                         transition_energy=wavenumber)

    res = EBX.xs[:, 0]*1e19 - xsO1D
    return res

# main ============================================================
# experimental data - O₂ photodissociation cross section yielding O(¹D₂)
wavelength, xsO1D, xsO1Dstderr = np.loadtxt('data/ANU/xsf.dat',  unpack=True)
wavenumber = 1e8/wavelength  # wavelength convert Å to cm⁻¹
wavenumber = wavenumber[::-1]  # reverse order
xsO1D = xsO1D[::-1]

# limit experimental data for fit, to regions of strong cross section
limit = xsO1D >  xsO1D[wavenumber < 75000].max()*3/4

# initial parameters Wei analytical fit to B and E PECs (states)
EB, X = cse_model()
R, (B, E) = EB.R, np.diagonal(EB.VT).T*EB._evcm  # PECs in cm⁻¹
pars, Bfit, Efit, rB, rE = initial_PEC_parameters(R, B, E)

pars = np.concatenate((pars, [0.85, 1500]))

result = least_squares(residual, pars, 
                       args=(wavenumber, xsO1D, R, B, E, Bfit, Efit, rB, rE))

# least shares fit - adjusting PECs, EDTM, and coupling
result.stderr = analytical.fiterrors(result)

print(result)

Bpar, Epar = np.split(result.x[:10], 2)

xsfit = residual(result.x, wavenumber, xsO1D, R, B, E, Bfit, Efit, rB, rE)\
        + xsO1D

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
ap.plot(R, B-X.cm, '--', label=r'valence $^3\Sigma^-_1$')
ap.plot(R, E-X.cm, '--', label=r'Rygberg $^3\Sigma^-_1$')
ap.plot(R[rB], analytical.Wei(R[rB], *Bpar)-X.cm,
        label='valence fit to $\sigma$')
ap.plot(R[rE], analytical.Wei(R[rE], *Epar)-X.cm,
        label='Rydberg fit to $\sigma$xs.')
ap.legend(fontsize='small', labelspacing=0.3)
ap.axis([0.8, 2.5, wavenumber[0]-10000, wavenumber[-1]+5000])
ap.set_xlabel(r'internuclear distance ($\AA$)')
ap.set_ylabel(r'potential energy above $X_{v=0, J=0}$ (cm$^{-1}$)')

plt.suptitle(r'O$_2$ photodissociation yielding O($^1D_2$)')
plt.tight_layout(w_pad=0)
plt.savefig('figures/O2_SRC_EBX_fit.svg')
plt.show()
