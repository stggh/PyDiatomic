# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

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

print("example_RKR.py - for this example accept the default inputs\n")
fn = input("RKR: Spectroscopic constants filename [data/GB.dat]: ")
fn = 'data/GB.dat' if fn == '' else fn

try:
    vv, Gv, Bv = np.loadtxt(fn, unpack=True)
except FileNotFoundError:
    print("RKR: file '{:s}' not found".format(fn))

# reduced mass - see Huber+Herzberg - default is O2
mu = input("RKR: Molecule reduced mass [7.99745751]: ")
mu = 7.99745751 if mu == '' else float(mu)

# De - dissociation energy
De = input("RKR: De [42021.47 cm-1]: ")
De = 42021.47 if De == '' else float(De)

# outer limb extension
limb = input("RKR: Outer-limb LeRoy(L) or Morse(M) [L]: ")
if limb == '':
    limb = 'L'

R, PEC, RTP, PTP = cse.rkr(mu, vv, Gv, Bv, De, limb, dv=0.1,
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

plt.savefig("output/example_RKR.png", dpi=75)
plt.show()
