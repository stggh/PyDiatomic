#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

u"""
  Rydberg-Klein-Rees evaluation of a potential energy curve from spectroscopic constants

  Stephen.Gibson@anu.edu.au
  2016
"""

import cse
import numpy as np
import scipy.constants as const
from scipy.interpolate import splrep, splev
import matplotlib.pyplot as plt
import sys

fn = input(u"RKR: Spectroscopic constants filename [data/GB.dat]: ")
fn = u'data/GB.dat' if fn=='' else fn

try:
    vv, Gv, Bv = np.loadtxt(fn, unpack=True)
except IOError as xxx_todo_changeme:
    errno, strerror = xxx_todo_changeme.args
    print(u"RKR: I/O error({}): {}".format(errno,strerror))

# reduced mass - see Huber+Herzberg - default is O2
mu = input(u"RKR: Molecule reduced mass [7.99745751]: ")
mu = 7.99745751 if mu=='' else float(mu)

# De
De = input(u"RKR: De [42021.47 cm-1]: ")
De = 42021.47 if De=='' else float(De)

# outer limb extension
limb = input(u"RKR: Outer-limb LeRoy(L) or Morse(M) [L]: ")
if limb=='': limb='L'

R, PEC, RTP, PTP = cse.rkr(mu, vv, Gv, Bv, De, limb, dv=0.1,
                           Rgrid=np.arange(0.005, 10.004, 0.005))

data = np.column_stack((R.T, PEC.T))
np.savetxt(u"data/RKR.dat",data)
print(u"RKR: potential curve written to 'data/RKR.dat'")

plt.plot(R, PEC, RTP[::10], PTP[::10], 'o')
plt.axis(xmin=0.5, xmax=5, ymin=-0.1, ymax=10)
plt.xlabel(u"R($\AA$)")
plt.ylabel(u"E(eV)")
plt.savefig(u"data/example_rkr.png", dpi=100)
plt.show()
