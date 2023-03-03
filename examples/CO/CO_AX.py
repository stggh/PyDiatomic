import numpy as np
import cse

X = cse.Cse('12C18O', dirpath='potentials', suffix='.dat', VT=['X1S0'])
X.levels(5)
X0 = X.results[0][0]
B0 = X.results[0][1]

A = cse.Cse('12C18O', dirpath='potentials', suffix='.dat', VT=['A1P1'], rot=1)
A.levels(5)
offset = 67618.07 - A.results[2][0]
print(offset)

AG2 = A.results[2][0]
AB2 = A.results[2][1]

print('¹²C¹⁸O A¹Π-X¹Σ⁺ (2,0) VUV-FT: Table 1 JQSRT 273 107837 (2021)')
print(' J"   R(J")    Q(J")    P(J")')
for Jdd in range(46):
    eni = X0 + B0*Jdd*(Jdd+1)
    X.solve(eni, rot=Jdd)

    print(f'{Jdd:2} ', end='')
    for Jd in (Jdd+1, Jdd, Jdd-1):
        if Jd < 1:
            continue

        enf = AG2 + AB2*Jd*(Jd+1)
        A.solve(enf, rot=Jd)

        wn = A.cm - X.cm + offset + X0
        print(f'{wn:9.2f} ', end='')
    print()
