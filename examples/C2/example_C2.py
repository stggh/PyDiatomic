import numpy as np
import cse
import matplotlib.pyplot as plt

####################################################################
#
# C2 D-X Franck-Condon factors
#
# (1) X1S0 - ground state RKR potential energy curve
# (2) D1S0 - upper state RKR potential energy curve
# (3) solve Schroedinger equation to evaluate Franck-Condon factors
#
###################################################################

def G(v, we, wexe):
    return we*(v+1/2) - wexe*(v+1/2)**2

def B(v, Be, alphae):
    return Be - alphae*(v+1/2)

fig, (ax0, ax1) = plt.subplots(1, 2)

evcm = 8065.541
internuclear_distance = np.arange(0.5, 10.001, 0.001)

# RKR potential energy curves
# X1S0 ground state ---------------------
vv = np.arange(10)
Gv = G(vv, we=1854.71, wexe=13.34)
Bv = B(vv, Be=1.8189, alphae=0.0176)

# C2 reduced mass
μ = 6.00535

# De - dissociation energy
De = 6.21*8065.541
Voo = De

# outer limb extension
limb = 'L'

# ground state RKR
print("X1S0 - ground state RKR ------------")
RX, X1S0, RTP, PTP = cse.tools.RKR.rkr(μ, vv, Gv, Bv, De, Voo, limb, dv=0.1,
                                       Rgrid=internuclear_distance)
ax0.plot(RX, X1S0, label=r'$X ^1\Sigma_g^+$')
ax0.plot(RTP[::10], PTP[::10], 'oC9')
ax0.set_xlabel(r'internuclear distance ($\AA$)')
ax0.set_ylabel(r'potential energy (cm$^{-1}$)')

# D1S0 upper state ---------------------
Te = 43239.4
Gv = G(vv, we=1829.57, wexe=13.94) + Te
Bv = B(vv, Be=1.8332, alphae=0.0196)

# uppder state RKR
print("\nD1S0 - upper state RKR -------------------")
RD, D1S0, RTP, PTP = cse.tools.RKR.rkr(μ, vv, Gv, Bv, De, Voo, limb, dv=0.1,
                                       Rgrid=internuclear_distance)

# Te displacement
ax0.plot(RD, D1S0, 'C2', label=r'$D ^1\Sigma_g^+$')
ax0.plot(RTP[::10], PTP[::10], 'oC8')
ax0.axis(xmin=0.5, xmax=3, ymin=-10000, ymax=80000)

# C2 D-X FCF calculation 
#  published values Sorkhabi et al. J. Molec. Spectrosc. 188, 200-208 (1998)
#  DXFCF[v", v']
Sorkhabi_FCF = np.array([[0.9972, 2.72e-3, 8.807e-5, 3.676e-7, 5.04e-9],
                         [2.642e-3, 0.9922, 4.842e-3, 2.692e-4, 1.929e-6],
                         [1.644e-4, 4.547e-3, 0.9883, 6.447e-3, 5.494e-4],
                         [5.995e-6, 4.596e-4, 5.846e-3, 0.9847, 8.00e-3],
                         [2.59e-7, 1.76e-8, 0.001, 0.017, 0.9802]])

# CSE calculation ---------------------
print("\nCSE FCF calculation -------------------")
C2X = cse.Cse('12C12C', VT=[(RX, X1S0)])
C2D = cse.Cse('12C12C', VT=[(RD, D1S0)])
C2 = cse.Transition(C2D, C2X, dipolemoment=[1])

C2.gs.solve(1000)
print(f'X(v"={C2.gs.vib:d}) = {C2.gs.cm:10.5f} cm-1')
ax0.plot(C2.gs.R, C2.gs.wavefunction.T[0, 0]*4000+C2.gs.energy)

# ground state eigenvalues - calculate every vibrational level = slow
# C2.gs.levels(4)
# enX = C2.gs.results[0][0]

# upper state eigenvalues - all levels = slow, but need the accurate 
# print('calculating upper state eigenvalues, will take a little while ...')
# C2.us.levels(4)
# vibD = C2.us.results.keys()
# enD = np.asarray(list(zip(*C2.us.results.values()))[0])

C2.us.solve(44000)
print(f'D(v\'={C2.us.vib:d}) = {C2.us.cm:10.5f} cm-1\n')
C2.us.solve(50000)
ax0.plot(C2.us.R, C2.us.wavefunction.T[0, 0]*4000+C2.us.energy)

# or roughly known
enD = np.array([44200, 45960, 47730, 49500, 51300, 55820, 63300, 80880])

print('Franck-Condon calculation')
print('v"    v\'    E\'-E"      FCF')
for vdd in range(2):
    C2.gs.solve(1890*(vdd+1/2))
    enX = C2.gs.cm
    print(f'{C2.gs.vib:1d}', end=None)

    enDX = enD - enX
    C2.calculate_xs(transition_energy=enDX, eni=enX)

    fcf = C2.xs[:, 0]*1.0e6/C2.wavenumber/3

    for vib, (en, osc) in enumerate(zip(C2.wavenumber, fcf)):
        print(f'     {vib:2d}   {en:8.3f}   {osc:7.2e}')
        if vib == 0:
            lbl = rf'v"={vdd}'
        else:
            lbl = ''
        p1 = ax1.semilogy(vib, osc, f'oC{vdd}', label=lbl)

    p2 = ax1.plot(Sorkhabi_FCF[vdd], f'+C{9-vdd}', ms=10,
                  label=f'Sorkhabi {vdd}')

plt.legend(labelspacing=0.1)
plt.tight_layout()
plt.savefig('figures/example_C2.svg')
plt.show()
