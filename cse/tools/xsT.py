import numpy as np
import os
import glob
import gzip
import cse
from cse.tools.intensity import Boltzmann

def total_cross_section(temperature, E0=None):
    xstotal = None
    Q = 0
    for xsf in glob.glob('xs_*.gz'):
        with gzip.open(xsf, 'r') as f:
            header = f.readline().decode('utf8').strip()
        
        wav, xs = np.loadtxt(xsf, unpack=True)
        if np.any(np.isnan(xs)):
            continue
        if np.any(np.isinf(xs)):
            continue

        if xstotal is None:
            xstotal = np.zeros_like(xs, dtype=float)

        enJd = header.split(' ')
        en = float(enJd[-1])
        if E0 is not None:
            en -= E0
        Jdd = int(enJd[2].split('=')[1][:-1])

        bltz = Boltzmann(en, Jdd, temperature)
        # partition function
        Q += bltz
        xstotal += xs*bltz

    np.savetxt(f'{temperature:d}K', np.column_stack((wav, xstotal/Q)))
    print(f'{os.getcwd().split("/")[-1]:s} partition function Q = {Q:g}')
