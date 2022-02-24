import numpy as np
import cse
import matplotlib.pyplot as plt

# instance
O2X = cse.Cse('O2', VT=['potentials/X3S-1.dat'], en=800)
O2B = cse.Cse('O2', VT=['potentials/B3S-1.dat'])

O2 = cse.Transition(O2B, O2X, dipolemoment=[1])

# B-X transition energy guesses
bands = [49357, 50045, 50710, 51352, 51968, 52560, 53123, 53655,
         54157, 54622, 55051, 55439, 55785, 56086, 56341, 56551,
         56720, 56853, 56955, 57032, 57087, 57121]

O2.calculate_xs(transition_energy=bands)
osc = O2.xs

Kvd, Kvdd, KdE, Kfcf = np.loadtxt("data/O2BX-FCF-Krupenie.dat", unpack=True)
Kv0 = int(Kvd[0])

fcf = []
print(r" v'     FCF_cse      FCF_Krupenie")
for v, f in enumerate(osc):
    f = f[0]*1.0e6/bands[v]/3
    fcf.append(f)
    if v in Kvd:
        print(f'{v:2d}   {f:10.3e}   {Kfcf[v-Kv0]:10.3e}')
    else:
        print(f'{v:2d}   {f:10.3e}   {"-":>5s}')

plt.plot(fcf, 'o', label=r'PyDiatomic')
plt.plot(Kvd, Kfcf, 'C1+', ms=8, label=r'Krupenie')

plt.title(r'O$_2$ $B\, ^3\Sigma_{u}^{-} - X\, ^3\Sigma_{g}^{-}$ $(v^\prime,'
          r' v^{\prime\prime}=0)$')
plt.ylabel(r'Franck-Condon factor')
plt.xlabel(r'$v^\prime$')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.legend()
plt.yscale('log')

plt.tight_layout()
plt.savefig('output/O2_fcf.svg')
plt.show()
