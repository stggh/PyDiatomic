import numpy as np
import cse
import matplotlib.pyplot as plt

# potential curve data   O2 X-state
R, V = np.loadtxt('potentials/X3S-1.dat', unpack=True)

# trim
subR = np.logical_and(R >= 0.8, R <= 6)
R = R[subR]
V = V[subR]

morse = cse.tools.LeRoy.Morsefit(R, V, Rref=2.5, Nbeta=5, q=3,
                                 fitpar=['beta', 'Rref', 'Re'])

print(morse.est.x)  # betas from linear fit
print(morse.fit.message)
print(morse.fit.x)  # fitpar from full EMO fit = beta, Rref, Re

plt.plot(R, V, label='PEC')
plt.plot(morse.R, morse.VEMO, label='EMO fit')
plt.legend()
plt.xlabel(r'internuclear distance ($\AA$)')
plt.ylabel(r'potential energy (cm$^{-1}$)')
plt.subplots_adjust(left=0.15, right=0.95)

plt.savefig('output/example_extended_Morse_oscillator.png', dpi=75)
plt.show()
