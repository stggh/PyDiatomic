import numpy as np
import cse
import matplotlib.pyplot as plt
import scipy.constants as C
import time
from scipy.interpolate import splrep, splev

t0 = time.time()

# cse ---------------------
X = cse.Cse('S2', VT=['potentials/X/X3S-1.dat'], en=400)
B = cse.Cse('S2', dirpath='potentials',
             VT=['B/B3S-1.dat', 'P/Bpp3P1.dat', 'abinitio/Wheeler/5P1.dat',
                 'abinitio/Wheeler/1P1.dat'], coup=[57, 205, 66, *([0]*3)])
                  # JCP 148, 244303 
BX = cse.Transition(B, X, dipolemoment=[0.934, 0.12, 0, 0])

wavenumber = np.arange(35820, 42000, 1)

pecfsheader = " ".join(B.pecfs)

Q = 0.0
G0, B0, D0, _ = X.results[0]
xst = np.zeros_like(wavenumber, dtype=float)
for Ndd in range(1, 55, 2):
    for branch in ('P', 'R'):
        edd = G0 + B0*Ndd*(Ndd+1)
        Nd = Ndd + ord(branch) - ord('Q')

        BX.calculate_xs(transition_energy=wavenumber, eni=edd, roti=Ndd,
                        rotf=Nd, honl=True) 

        xs = BX.xs
        np.savetxt(f'data/xsN/xs_0_{Nd}_{Ndd}.dat.gz',
                np.column_stack((wavenumber , *xs.T)), fmt='%8.3f'+'%15.8e'*4,
                header=f'{BX.gs.cm} {pecfsheader}')

        Boltz = (2*Ndd+1)*np.exp(-edd*C.h*C.c*100/C.k/370) 
        Q += Boltz

        xsN = xs.sum(axis=1)  # f-levels
        # fine-structure shift e-levels
        spl = splrep(wavenumber, xsN)
        xsJ = splev(wavenumber-2*11.78/3, spl)  # λv = 11.78 cm⁻¹
        xst += (2*xsN + xsJ)*Boltz

        # finsh calculation if intensity is weak
        if Ndd > 5 and Boltz < 5e-3 and branch == 'R':
            break

t1 = time.time()

xst /= Q

# experimental data - Glenn Stark
xs370Kpre = np.loadtxt('data/Wellesley/xs_370K_prediss.dat', unpack=True)

sf = xs370Kpre[1].max()/xst.max()
print(f'scalefactor = {sf:5.3g}')
sf *= 1.1

dt = int(t1 - t0)
mins = dt % 60
secs = dt // 60
print(f'  in {mins} minutes and {secs} seconds\n')

# plot --------------------
plt.semilogy(BX.wavenumber, xst*sf, label=f'PyDiatomic x{sf:5.2g}')
plt.semilogy(*xs370Kpre, label='Stark')
plt.xlabel(r'wavenumber (cm$^{-1})')
plt.ylabel(r'photodissocation cross section (cm$^2$)')
plt.legend(fontsize='small', labelspacing=0.3)

plt.savefig('figures/BX.svg')
plt.show()
