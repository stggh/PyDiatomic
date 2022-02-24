import numpy as np
import cse
import matplotlib.pyplot as plt

# potential curve data   O2 X-state
R, V = np.loadtxt('potentials/X3S-1.dat', unpack=True)

subR = np.logical_and(R >= 0.8, R <= 6)
R = R[subR]
V = V[subR]

morse = cse.tools.LeRoy.Morsefit(R, V, Re=R[V.argmin()], Rref=2.13, q=6, p=3,
                                 beta=[3., 1., 1., 1.],
                                 fitpar=['beta', 'Re'])

print('linear fit betas:\n',morse.est.x)  # betas from linear fit
print('full EMO fit message:\n', morse.est.message)
print('full EMO fit parameters betas, Re:\n',morse.fit.x) 

# compare eignevalues ---------------------------------------
# eigenvalues - RKR potential curve
Xrkr = cse.Cse('O2', VT=[(R, V)])

v = np.arange(20)
Xrkr.levels(v[-1]+2)
Gvrkr, Bvrkr, Dvrkr, Jvrkr = list(zip(*Xrkr.results.values()))
Gvrkr = np.array(Gvrkr)
subv = v <= v[-1] 

# expanded Morse oscillator fit
Xemo = cse.Cse('O2', VT=[(morse.R, morse.VEMO)])
Xemo.levels(v[-1]+2)
Gv, Bv, Dv, Jv = list(zip(*Xemo.results.values()))
Gv = np.array(Gv)
dG = Gv - Gvrkr

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

plt.tight_layout(w_pad=0.4)

plt.savefig('output/example_expanded_Morse_oscillator.svg')
plt.show()


