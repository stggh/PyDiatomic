import numpy as np
import cse

###########################################################################
#
# PyDiatomic fine-structure energy levels of O2 X
#   relative to the exact analytical values of Rouille JMS 154, 372-382 (1992)
#
# Stephen.Gibson@anu.edu.au - November 2017
#
###########################################################################

# f-levels
O2f = cse.Cse('O2', VT=['potentials/X3S-1.dat'])

# e-levels
O2e = cse.Cse('O2', dirpath='potentials',
               VT=['X3S-1.dat', 'X3S0.dat', 'b1S0.dat'], coup=[-2.005, 0, 229])

O2f.solve(en=800)
E0 = O2f.cm

print("PyDiatomic O2 X-state fine-structure levels")
print("  energy diffences (cm-1): Rouille - PyDiatomic")
print(" N        F1          F2          F3")
F = np.zeros(4)
f = np.zeros(4)
for N in range(1, 101, 2):
    F[2] = cse.rouille(0, N, N)
    O2f.solve(en=F[2]+E0, rot=N)
    f[2] = O2f.cm
    for fi in (1, 3):
        J = N - fi + 2
        F[fi] = cse.rouille(0, N, J)  
        O2e.solve(en=F[fi]+E0, rot=J)
        f[fi] = O2e.cm

    print(f'{N:2d}  {F[1]+E0-f[1]:10.3f}  {F[2]+E0-f[2]:10.3f}  '
          f'{F[3]+E0-f[3]:10.3f}')
