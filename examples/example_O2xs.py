# -*- coding: utf-8 -*-
"""
  CSE  - solve the coupled-channel time-independent Schrödinger equation
         using recipe of B.R. Johnson J Chem Phys 69, 4678 (1977).

  Stephen.Gibson@anu.edu.au
  2016
"""

import numpy as np
import matplotlib.pylab as plt
import time

import cse

evcm = 8065.541   # conversion factor eV -> cm-1

d = 'potentials/'  # directory name

wavelength = np.arange(110, 174.1, 0.1)  # nm

