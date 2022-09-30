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


import cse
import numpy as np
import scipy.constants as const
from scipy.interpolate import splrep, splev
import matplotlib.pyplot as plt
import sys

fn = 'vGB-B-Partridge.dat'

vv, Gv, Bv = np.loadtxt(fn, unpack=True)

# reduced mass - see Huber+Herzberg - default is O2
mu = 15.9860364

Te = 3.946*8065.541
# De - dissociation energy
De = 44837.379 - Te

# outer limb extension
limb = input("RKR: Outer-limb LeRoy(L) or Morse(M) [L]: ")
if limb == '':
    limb = 'L'

R, PEC, RTP, PTP = cse.rkr(mu, vv, Gv, Bv, De, limb, dv=0.1,
                           Rgrid=np.arange(0.005, 10.004, 0.005))

data = np.column_stack((R.T, PEC.T+Te/8065.541))
np.savetxt("B3S-1rkr.dat", data)
print("RKR: potential curve written to 'B3S-1rkr.dat'")

plt.plot(R, PEC, label='RKR potential curve')
plt.plot(RTP[::10], PTP[::10], 'o', label='turning points')
plt.legend()
plt.axis(xmin=0.8, xmax=4, ymin=-0.1, ymax=6)
plt.title(r"S$_2$ $X {}^{3}\Sigma_g^-$")

plt.xlabel(r"R($\AA$)")
plt.ylabel("E(eV)")

plt.savefig("RKR-B.png", dpi=75)
plt.show()
