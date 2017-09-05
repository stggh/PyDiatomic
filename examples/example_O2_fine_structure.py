import numpy as np
import cse

# intial guess energies for rotational levels J" = N"-F"+2
en1 = np.loadtxt("data/X1-energies.dat")
en2 = np.loadtxt("data/X2-energies.dat")
en3 = np.loadtxt("data/X3-energies.dat")

# e-levels
O2e = cse.Cse('O2', VT=['potentials/X3S-1.dat', 'potentials/X3S0.dat',
                        'potentials/b1S0.dat'], coup=[-2, 0, 229])

print("F1 levels")
print("  cm-1      J      diff.")
for en, J, F in en1[:10]:
    O2e.solve(en, rot=J)
    print("{:8.3f}   {:2.0f}   {:8.3f}".format(O2e.cm, J, O2e.cm-en))

print("\nF3 levels")
print("  cm-1      J      diff.")
for en, J, F in en3[:10]:
    O2e.solve(en, rot=J)
    print("{:8.3f}   {:2.0f}   {:8.3f}".format(O2e.cm, J, O2e.cm-en))


# f-levels
O2f = cse.Cse('O2', VT=['potentials/X3S-1.dat'])
print("\nF2 levels")
print("  cm-1      J      diff.")
for en, J, F in en2[:10]:
    O2f.solve(en, rot=J)
    print("{:8.3f}   {:2.0f}   {:8.3f}".format(O2f.cm, J, O2f.cm-en))
