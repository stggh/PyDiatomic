import numpy as np
import cse
import matplotlib.pyplot as plt

####################################################################
#
# C2 D-X Franck-Condon factors
#
# (1) X1S0 - ground state RKR potential energy curve
# (2) D1S0 - upper state RKR potential energy curve
# (3) solve Schroedinger equation
#
###################################################################

# RKR potential energy curves
# X1S0 ground state -----------
we = 1854.71
wexe = 13.34

Be = 1.8189
alphae = 0.0176

Gv = []
Bv = []
vv = np.arange(10)
for v in vv:
    Gv.append(we*(v+1/2) - wexe*(v+1/2)**2)
    Bv.append(Be - alphae*(v+1/2))

Gv = np.asarray(Gv)
Bv = np.asarray(Bv)

# C2 reduced mass
mu = 6

# De - dissociation energy
De = 6.21*8065.541

# outer limb extension
limb = 'L'

# ground state RKR
print("X1S0 - ground state RKR ------------")
RX, X1S0, RTP, PTP = cse.tools.RKR.rkr(mu, vv, Gv, Bv, De, limb, dv=0.1,
                                       Rgrid=np.arange(0.5, 10.001, 0.001))


# D1S0 upper state -----------

Te = 43239.4/8065.541

we = 1829.57
wexe = 13.94

Be = 1.8332
alphae = 0.0196

Gv = []
Bv = []
vv = np.arange(10)
for v in vv:
    Gv.append(we*(v+1/2) - wexe*(v+1/2)**2)
    Bv.append(Be - alphae*(v+1/2))

Gv = np.asarray(Gv)
Bv = np.asarray(Bv)

# uppder state RKR
print("\nD1S0 - upper state RKR -------------------")
RD, D1S0, RTP, PTP = cse.tools.RKR.rkr(mu, vv, Gv, Bv, De, limb, dv=0.1,
                                       Rgrid=np.arange(0.5, 10.001, 0.001))

# Te displacement
D1S0 += Te

# CSE calculation

print("\nCSE FCF calculation -------------------")
C2 = cse.Xs('C2', VTi=[(RX, X1S0)], eni=900, VTf=[(RD, D1S0)],
            dipolemoment=[1])

C2.gs.solve(1000)
print('E(v"={:d}) = {:8.5f} cm-1'.format(C2.gs.vib, C2.gs.cm))

C2.us.solve(43500)
print('E(v\'={:d}) = {:8.5f} cm-1\n'.format(C2.us.vib, C2.us.cm))

# ground state eigenvalues - calculate every vibrational level = slow
# C2.gs.levels(4)
# enX = C2.gs.calc[0][0]

# upper state eigenvalues - all levels = slow
# C2.us.levels(4)
# vibD = C2.us.calc.keys()
# enD = np.asarray(list(zip(*C2.us.calc.values()))[0])

# or roughly known
enD = np.array([44200, 45960, 47730, 49500, 51300, 55820, 63300, 80880])

print('v"    v\'    E\'-E"      FCF')
for vdd in range(2):
    C2.gs.solve(1890*(vdd+1/2))
    enX = C2.gs.cm
    print('{:1d}'.format(C2.gs.vib), end=None)

    enDX = enD - enX
    C2.calculate_xs(transition_energy=enDX, eni=enX)

    fcf = C2.xs[:, 0]*1.0e6/enDX/3

    for vib, (en, osc) in enumerate(zip(enDX, fcf)):
        print('   {:2d}   {:8.3f}   {:7.2e}'.format(vib, en, osc))
