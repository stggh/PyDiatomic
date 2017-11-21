import numpy as np
import scipy.constants as const
from sympy.physics.wigner import wigner_3j

def Boltzmann(en, J, T):
    return (2*J + 1)*np.exp(-en*const.h*const.c*100/const.k/T)

def honl(Jd, Jdd, Od, Odd):
    return (2*Jdd + 1)*np.float(wigner_3j(Jd, 1, Jdd, -Od, 0, Odd))**2
