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
O2e = cse.Cse('O2', dirpath='potentials', suffix='.dat',
               VT=['X3S-1', 'X3S0', 'b1S0'], coup=[-2.005, 0, 229])

O2f.solve(en=800)
E0 = O2f.cm  # reference energy J"=0, N"=0 virtual state
print(f'E0 = {E0:8.5f}')

print("PyDiatomic O2 X-state fine-structure levels")
print("  energy diffences (cm-1): Rouille - PyDiatomic")
print(" N        F1          F2          F3")
F = np.zeros(4)
f = np.zeros(4)
for N in range(1, 33, 2):
    F[2] = cse.rouille(0, N, N) + E0

    O2f.solve(en=F[2], rot=N)
    f[2] = O2f.cm

    for fi in (1, 3):
        J = N - fi + 2
        F[fi] = cse.rouille(0, N, J)  + E0

        O2e.solve(en=F[fi], rot=J)

        f[fi] = O2e.cm

    print(f'{N:2d}  {F[1]-f[1]:10.3f}  {F[2]-f[2]:10.3f} {F[3]-f[3]:10.3f}')
