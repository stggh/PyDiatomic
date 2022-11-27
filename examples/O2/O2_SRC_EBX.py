import numpy as np
import cse
import matplotlib.pylab as plt
import time

# Model ------------------------------
# O2 ground state X
X = cse.Cse('O2', VT=['potentials/X3S-1.dat'], en=800)
evcm = X._evcm  # conversion eV to cm⁻¹

# O2 upper coupled B-state
B = cse.Cse('O2', dirpath='potentials', suffix='.dat',
            VT=['B3S-1', 'E3S-1'], coup=[4000])
B.VT[1, 1] -= 1000/evcm
B.VT[-1, -1] -= 50/evcm

wn = np.arange(57500, 90000, 10)
print('O₂ photodissociation cross section yielding O(¹D₂)')
print('  speeded by Python multiprocessing Pool.map')
print(f'   {wn[0]:.0f} to {wn[-1]:.0f} in {wn[1]-wn[0]:.2g} cm⁻¹ steps')

# transition
t0 = time.time()
BX = cse.Transition(B, X, dipolemoment=[1, 0], transition_energy=wn)
t1 = time.time()

print(f' ...  in {time.time()-t0:.2g} seconds')
print('finished - see plot')

# experimental data - ANU - J Elect Spectrosc and Rel Phenom 80. 9-12 (1996)
xsO1D = np.loadtxt('data/ANU/xsf.dat', unpack=True)

# graphics ---------------------------------------
fig, (axx, axp) = plt.subplots(1, 2, sharey=True)

for V, xs, lbl in zip(np.diagonal(B.VT).T, BX.xs.T, B.statelabel):
    axp.plot(B.R, V*evcm, label=lbl)

    if np.all(xs <= 0):  # only plot non-zero cross sections
        continue
    axx.plot(xs, wn, label=lbl)

axx.plot(xsO1D[1]*1e-19, 1e8/xsO1D[0], 'C2', label=r'expt. O($^1D_2$)')

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
plt.savefig('figures/O2_SRC_EBX.svg')
plt.show()
