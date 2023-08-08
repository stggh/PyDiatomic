import numpy as np
import cse
import matplotlib.pylab as plt
import time

# Model ------------------------------
# O₂ ground state X instance
X = cse.Cse('O2', VT=['potentials/X3S-1.dat'], en=800)
evcm = X._evcm  # conversion factor eV to cm⁻¹

# O₂ upper coupled B-state instance
B = cse.Cse('O2', dirpath='potentials', suffix='.dat',
            VT=['B3S-1', 'E3S-1', '3P1', '3PR1'],
            coup=[4000, 40, 0, 0, 0, 7000])
# fine-tune vibrational energy level positions
B.VT[1, 1] -= 1000/evcm
B.VT[-1, -1] -= 50/evcm

print('Initial state ----------------------')
print(X)
print('Coupled excited state ----------------------')
print(B)

wn = np.arange(57500, 90000, 10)
print('Photodissociation cross section')
print('  speeded by Python multiprocessing Pool.map')
print(f'   {wn[0]:.0f} to {wn[-1]:.0f} in {wn[1]-wn[0]:.2g} cm⁻¹ steps')

# transition
t0 = time.time()
BX = cse.Transition(B, X, dipolemoment=[1, 0, 0, 0.3], transition_energy=wn)
t1 = time.time()

print(f' ...  in {time.time()-t0:.2f} seconds')
print('finished - see plot')

# experimental data - ANU - J Elect Spectrosc and Rel Phenom 80. 9-12 (1996)
xsO1D = np.loadtxt('data/ANU/xsf.dat', unpack=True)
xsO3P = np.loadtxt('data/ANU/xsp.dat', unpack=True)
xsexpt = np.loadtxt('data/ANU/xst.dat', unpack=True)

# graphics ---------------------------------------
fig, (axx, axp) = plt.subplots(1, 2, sharey=True)

xst = BX.xs.sum(axis=1)  # total cross section
axx.plot(xst, BX.wavenumber, 'k:', label='total')

for V, xs, lbl in zip(np.diagonal(B.VT).T, BX.xs.T, B.statelabel):
    i = int('Σ' in lbl)
    col = f'C{i}'
    ls = '-' if i==1 else '--'
    axp.plot(B.R, V*evcm, col, label=lbl, ls=ls)

    if np.all(xs <= 0):  # only plot non-zero cross sections
        continue
    axx.plot(xs, wn, col, label=lbl, ls=ls)

axx.plot(xsexpt[1]*1e-19, 1e8/xsexpt[0], 'C9', label='expt. total')
axx.plot(xsO1D[1]*1e-19, 1e8/xsO1D[0], 'C8', label=r'expt. O($^1D_2$)')
axx.plot(xsO3P[1]*1e-19, 1e8/xsO3P[0], 'C7', label=r'expt. O($^3P_J$)')

axx.set_ylabel(r'wavenumber above $X$min. ($10^4$cm$^{-1}$)')
axx.set_xlabel(r'cross section (cm$^{2}$)')
axx.axis(xmin=1.5e-17, xmax=-0.1e-17, ymin=4, ymax=10)
axx.set_title("photodissociation", fontsize=12)
axx.legend(fontsize='small', labelspacing=0.3)
axx.ticklabel_format(axis='y', style='sci', scilimits=(4, 4))

axp.plot(X.R, X.VT[0, 0]*evcm, color='k', label=X.statelabel[0])

axp.set_title("diabatic PECs", fontsize=12)
axp.axis(xmin=0.8, xmax=2.8, ymin=40000+X.cm, ymax=100000+X.cm)
axp.set_xlabel("internuclear distance ($\AA$)")
axp.axes.get_yaxis().set_visible(False)
axp.legend(fontsize='small', labelspacing=0.3)

plt.tight_layout(h_pad=0, w_pad=0)
plt.savefig('figures/O2_EBX.svg')
plt.show()
