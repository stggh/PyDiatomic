import numpy as np
import cse
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths

# CSE model ---------------------------------------
# ground state
Xf = cse.Cse('O2', VT=['potentials/X3S-1.dat'], en=800)

# excited state
B = cse.Cse('O2', dirpath='potentials', suffix='.dat', VT=['B3S-1'])
B.levels()
print(B)

# coupled excited state
Bf = cse.Cse('O2', dirpath='potentials', suffix='.dat',
             VT=['B3S-1', '5P1', '3S+1', '3P1', '1P1'],
             coup=[74, 38, 29, 28, *([0]*6)])

band = [(v, w[0]-Xf.cm) for v, w in B.results.items()]

# variable transition energy grid, 0.1 cm⁻¹ near band, 10 cm⁻ between bands
wn = np.zeros(1)
dw = 0.05
for v, w in band[:-1]:
    wm = int(w/dw)*dw
    wx = int(band[v+1][1]/dw)*dw
    wn = np.append(wn, np.arange(wm-5, wm+5, dw))
    wn = np.append(wn, np.arange(wm+5, wx-5, 10))

wn = wn[1:]

# transition instance
BXf = cse.Transition(Bf, Xf, transition_energy=wn, dirpath='transitionmoments',
                     dipolemoment=['dipole_b_valence.dat', *([0]*4)])

xstotal = BXf.xs.sum(axis=1)

# analysis -----------------------------------------
peaks, _ = find_peaks(xstotal)
FWHM, *_ = peak_widths(xstotal, peaks, rel_height=1/2)
FWHM *= dw   # peak grid increment

shift = []
print(' v   position   FWHM   shift')
for vb, p, f in zip(band, peaks, FWHM):
    shift.append(wn[p]-vb[1])
    print(f'{vb[0]:2d} {wn[p]:,.2f} {f:5.2f} {shift[-1]:.2f}')

'''
partfwhm = []
for x in BXf.xs.T:  # each partial cross section
    pks, _ = find_peaks(x)
    fwhm, *_ = peak_widths(x, pks, rel_height=1/2)
    partfwhm.append(fwhm*dw)
'''

anu = np.genfromtxt('SRBwidths.dat', unpack=True, dtype=[int, float, float])

# plot -------------------------------------------
fig, axes = plt.subplot_mosaic('''
                               pxxww
                               pxxss
                               ''', figsize=(12, 8))
ap, ax, aw, ash = axes['p'], axes['x'], axes['w'], axes['s']


ap.plot(Xf.R, Xf.VT[0, 0]*Xf._evcm-Xf.cm, label=Xf.statelabel[0])
ap.text(*(3, 41000), Xf.statelabel[0], fontsize='large', color='C0')
coord = [(2.3, 50000), (2.3, 45000), (2.3, 48000), (2.3, 52000), (2.3, 54000)]
for i in range(5):
    ap.plot(Bf.R, Bf.VT[i, i]*Bf._evcm-Xf.cm)
    ap.text(*coord[i], Bf.statelabel[i], fontsize='large', color=f'C{i+1}')

ap.set_ylabel(r'transition energy (cm$^{-1}$)')
ap.set_xlabel(r'internculear distance (Å)')
ap.set_title('PECs')
ap.ticklabel_format(axis='y', style='sci', scilimits=(4, 4))
ap.axis([1.2, 4, 40000, 60000])
ap.legend()

ax.semilogy(BXf.wavenumber, BXf.xs.sum(axis=1),label='total')
for v, (p, f) in enumerate(zip(peaks, FWHM)):
    ax.text(BXf.wavenumber[p], xstotal[p]*5, f'{v}', ha='center', va='bottom')
    ax.text(BXf.wavenumber[p], xstotal[p]*10, f'{f:.1f}', ha='center',
            va='bottom')

for xs, lbl in zip(BXf.xs.T, Bf.statelabel):
    if np.any(xs):
        ax.semilogy(BXf.wavenumber, xs, '--', label=f'channel {lbl}')

ax.legend(fontsize='small', labelspacing=0.3)
ax.set_xlabel(r'wavenumber (cm$^{-1}$)')
ax.set_ylabel(r'cross section (cm$^2$)')
ax.set_title('photodissociation cross section')

aw.plot(FWHM, 'o-', label='PyD.')
aw.plot(*anu[:2], 'o', mfc='w', label='ANU') 

aw.legend(fontsize='small', labelspacing=0.3)
aw.set_xlabel('vibrational quantum number') 
aw.set_ylabel(r'FWHM (cm$^{-1}$)')
aw.set_title('line width')

ash.plot(shift, 'o-')
ash.set_title('perturbation shift')
ash.set_xlabel('vibrational quantum number')
ash.set_ylabel(r'level shift (cm$^{-1}$)')
ash.axis([0, 22, -1, 1])

plt.suptitle(r'O$_2$ Schumann-Runge band predissociation $f$-level interaction')
plt.tight_layout(h_pad=-0.5)
plt.savefig('figures/O2_SRB_predissociation.svg')
plt.show()
