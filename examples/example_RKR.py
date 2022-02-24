# -*- coding: utf-8 -*-
#################################################################
# Rydberg-Klein-Rees evaluation of a potential energy curve
# from spectroscopic constants
#
# see also example_Morse.py
#
# Stephen.Gibson@anu.edu.au
# 2016
#################################################################


import numpy as np
import cse

import scipy.constants as const
import matplotlib.pyplot as plt

print("example_RKR.py - for this example accept the default inputs\n")
fn = input("RKR: Spectroscopic constants filename [data/GB.dat]: ")
fn = 'data/GB.dat' if fn == '' else fn

try:
    vv, Gv, Bv = np.loadtxt(fn, unpack=True)
except FileNotFoundError:
    print(f"RKR: file '{fn:s}' not found")

# reduced mass in atomic mass units- see Huber+Herzberg - default is O2 = 7.9975
mol = input("RKR: diatomic molecule [O2]: ")
if mol == '':
    mol = 'O2'
μ = cse.cse_setup.reduced_mass(mol)[0]/const.m_u

# De - dissociation energy
De = input("RKR: De [42021.47 cm-1]: ")
De = 42021.47 if De == '' else float(De)

# outer limb extension
limb = input("RKR: Outer-limb LeRoy(L) or Morse(M) [L]: ")
if limb == '':
    limb = 'L'

R, PEC, RTP, PTP = cse.tools.RKR.rkr(μ, vv, Gv, Bv, De, Voo=0, limb=limb, 
                                     dv=0.1, ineV=True,
                                     Rgrid=np.arange(0.005, 10.004, 0.005))

data = np.column_stack((R.T, PEC.T))
np.savetxt("data/RKR.dat", data)
print("RKR: potential curve written to 'data/RKR.dat'")

plt.plot(R, PEC, label='RKR potential curve')
plt.plot(RTP[::10], PTP[::10], 'o', label='turning points')
plt.legend()
plt.axis(xmin=0.8, xmax=4, ymin=-0.1, ymax=6)
plt.title("example_RKR.py")

plt.xlabel(r"R($\AA$)")
plt.ylabel("E(eV)")

plt.savefig("output/example_RKR.svg")
plt.show()
