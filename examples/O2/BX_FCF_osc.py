import numpy as np
import cse
import matplotlib.pyplot as plt

##########################################################################
#
# O₂ B-X (v', v"=0) Franck-Condon factors and oscillator strengths
#
# Stephen.Gibson@anu.edu.au
#   6 September 2022
##########################################################################

# single channel model initial X-state and final B-state
# potential energy curve instances
O2X = cse.Cse('O2', VT=['potentials/X3S-1.dat'], en=800)
O2B = cse.Cse('O2', VT=['potentials/B3S-1.dat'])

# transition instance, electric-dipole-transition-moment (ETDM) =1 a.u.
O2 = cse.Transition(O2B, O2X, dipolemoment=[1])

# B-state energy levels:  O2B.results = {v:(Gv, Bv, Dv, J=0)}
print('Evaluating B-state energy levels')
O2B.levels(ntrial=20)

vib = np.array(list(O2B.results.keys()), dtype=int)
bands = [G-O2X.cm for (G, B, D, _) in O2B.results.values()]

# |<v'|v"=0>|²  for each B-band (v', 0) 
print('Evaluating |<B v\'|X v"=0>|²')
O2.calculate_xs(transition_energy=bands)

osc = O2.xs.flatten()  # oscillator strengths
fcf = osc*1e6/bands/3  # Franck-Condon factors
osc *= 0.84**2  # ETDM should be ~0.84

# Comparison data  ------------------------------
# FCFs from Krupenie review
Kvd, Kvdd, KdE, Kfcf = np.loadtxt("data/O2BX-FCF-Krupenie.dat", unpack=True)
Kvd = np.array(Kvd, dtype=int)
Kv0 = Kvd[0]  # first vibrational level of data

# Oscillator strengths from Yoshino Harvard data
Yv, Yosc, Yerr = np.loadtxt('data/Harvard/O2osc-Yoshino.dat', unpack=True)
Yv = np.array(Yv, dtype=int)
Yv0 = Yv[0]

print(r' v\'   FCF_cse    FCF_Krupenie       osc_cse     osc_Yoshino')
for v, nv, f, o in zip(vib, O2.wavenumber, fcf, osc):
    if v in Kvd:
        print(f'{v:2d} {f:12.3e} {Kfcf[v-Kv0]:12.3e}', end='')
    else:
        print(f'{v:2d} {f:12.3e} {"-":^12s}', end='')

    print('     ', end='')
    if v in Yv:
        print(f' {o:12.3e}   {Yosc[v-Yv0]:12.3e}')
    else:
        print(f' {o:12.3e}   {"-":^12s}')

# plot -----------------------------
fig, (axf, axo) = plt.subplots(1, 2)

# FCFs
axf.plot(vib, fcf, 'o', label=r'PyDiatomic FCF')
axf.plot(Kvd, Kfcf, 'o', mfc='w', label=r'Krupenie')
axf.set_ylabel(r'Franck-Condon factor')
axf.set_xlabel(r'$v^\prime$')
axf.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
axf.legend(fontsize='small', labelspacing=0.3)
axf.set_yscale('log')

# oscs
axo.plot(vib, osc, 'o', label=r'PyDiatomic osc')
axo.errorbar(Yv, Yosc, Yerr, fmt='o', color='C3', mfc='w', label=r'Yoshino')
axo.set_xlabel(r'$v^\prime$')
axo.set_ylabel(r'oscillator strength')
axo.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
axo.legend(fontsize='small', labelspacing=0.3)
axo.set_yscale('log')

plt.suptitle(r'O$_2$ $B\, ^3\Sigma_{u}^{-} - X\, ^3\Sigma_{g}^{-}$ $(v^\prime,'
             r' v^{\prime\prime}=0)$')

plt.tight_layout()
plt.savefig('figures/BX_FCF_osc.svg')
plt.show()
