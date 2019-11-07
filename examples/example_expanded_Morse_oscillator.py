import numpy as np
import cse
import matplotlib.pyplot as plt

# potential curve data   O2 X-state
R, V = np.loadtxt('potentials/X3S-1.dat', unpack=True)

subR = np.logical_and(R >= 0.8, R <= 6)
R = R[subR]
V = V[subR]

morse = cse.tools.LeRoy.Morsefit(R, V, Rref=2.0, Nbeta=5, q=4,
                                 fitpar=['beta', 'Rref', 'Re'])

print('linear fit betas:\n',morse.est.x)  # betas from linear fit
print('full EMO fit message:\n', morse.fit.message)
print('full EMO fit parameters betas, Rref, Re:\n',morse.fit.x) 

# compare eignevalues ---------------------------------------
# eigenvalues - RKR potential curve
Xrkr = cse.Cse('O2', VT=[(R, V)])
Xrkr.levels(20)

# expanded Morse oscillator fit
Xemo = cse.Cse('O2', VT=[(morse.R, morse.VEMO)])

dG = []
for v, (Gv, Bv, Dv, J) in Xrkr.results.items():
        if v > 20: break
        Xemo.solve(Gv)
        dG.append(Gv - Xemo.results[v][0])

# plot --------------------------------------
fig, (ax0, ax1) = plt.subplots(1, 2)

ax0.plot(R, V*8065.541, label='PEC')
ax0.plot(morse.R, morse.VEMO*8065.541, label='EMO fit')
ax0.legend()
ax0.set_xlabel(r'internuclear distance ($\AA$)')
ax0.set_ylabel(r'potential energy (cm$^{-1}$)')
ax0.axis(xmax=4, ymin=-1000, ymax=60000)

ax1.plot(dG)
ax1.set_ylabel(r'$G_v$ (cm$^{-1}$)')
ax1.set_xlabel(r'$v$')
ax1.set_title('diffence RKR - EMO')

plt.subplots_adjust(wspace=0.4)

plt.savefig('output/example_expanded_Morse_oscillator.png', dpi=75)
plt.show()


