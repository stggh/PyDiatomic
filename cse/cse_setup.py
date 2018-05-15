# -*- coding: utf-8 -*-
import numpy as np
import scipy.constants as const
import re
from scipy.interpolate import splrep, splev

def reduced_mass(amu=None):
    """ Reduced mass of diatomic molecule.

    Parameters
    ----------
    amu : str or float
        reduced mass, str one of 'H2', 'HCl', 'OH', 'N2', 'N14', 'N14N15', 'N15', 'NO', 'O2', 'O16O17', 'O17O18', 'O16O18', 'O18', 'CO', 'Cl2', 'CS2', 'Cs2', 'ICl', 'I2', 'Br', 'S2', 'S32', 'S32O16', 'S33O16', 'S34O16' or float in amu 

    Returns
    -------
    mu : reduced mass in kg

    """

    amus = {\
        'H2': 0.50391261,
        'HCl': 0.97959272,
        'OH': 0.9480871,
        'N2': 7.0015372,
        'N14': 7.0015372,
        'N14N15': 7.242227222,
        'N15': 7.50005465,
        'NO': 7.46643323,
        'O2': 7.99745751,
        'O16O17': 1.368448e-26/const.u,
        'O17O18': 1.451734e-26/const.u,
        'O16O18': 1.40607e-26/const.u,
        'O18': 1.49417e-26/const.u,
        'C2': 6.0,
        'CO': 6.85620871,
        'Cl2': 17.4844268,
        'CS2': 8.727,
        'Cs2': 66.452718,
        'ICl': 27.4146708,
        'I2': 63.4522378,
        'Br': 39.459166,
        'S2': 15.9860364,
        'S32': 15.9860364,
        'S32O16': 10.6613029,
        'S33O16': 10.77016005,
        'S34O16': 10.78435767,
        'U' : 2.0   # unknown/unimportant
        }
    molecule = 'unknown'
    if amu is None: 
        mus = input ("CSE: reduced mass a.u. [O2=7.99745751]: ")
        if mus in amus.keys():
            molecule = mus
            amu = amus[mus]
        else:
            amu = float(mus) if len(mus) > 0 else 7.99745751
            molecule = 'O2'
    elif amu in amus.keys():
        molecule = amu
        amu = amus[amu] 
    else:
        # atomic mass given
        if amu < 1.0e-20:
            amu /= const.u
        for k, v in amus.items():
            if np.abs(amu-v) < 0.1:
                molecule = k
                break

    return amu*const.u, molecule


def potential_energy_curves(pecfs=None, R=None):
    """ Read potential energy curve file(s) and assemble as diagonals in an nxn array for the n-filenames provided.

    Parameters
    ----------
    pecfs : list of strings
        potential energy curve file name list ['pot1', 'pot2', 'pot3' ... ]
        Each file has 2 column format:  R(Angstroms)  V(eV)

    R : numpy 1d array
        radial grid if not None

    Returns
    -------
    R : numpy 1d array 
        radial coordinates, set to `numpy.arange(Rmin, Rmax+dR/2, dR)`
        where Rmin = highest R[0], Rmax = lowest R[-1] of all the
        potential curves

    VT : numpy 2D array size nxn
        (transposed) potential energy array for the n-filenames given: ::

            VT = [ V1  0   0 ...]                                     
                 [  0 V2   0 ...]                                    
                 [  0  0  V3 ...]                                    
                    :  :   :  
                 [  0  0  ... Vn] 

    pecfs: list of str
        as inputted

    limits : tuple
        (oo, n, Rmin, Rmax, Vm, Voo)
        oo = int, common length of radial and potential arrays
        n  = int, number of potential curves
        Rmin = highest minimum
        Rmax = lowest maximum
        Vm = lowest minimum potential energy (Te)
        Voo = lowest dissociation limit energy
    
    AM : 1d array of tuples
        Angular momenta quantum numbers (Omega, S, Lambda, Sigma) for each electronic state
    """

    if pecfs == None:
        pecfns =  input ("CSE: potential energy curves [X3S-1.dat]: ")
        pecfs = pecfns.replace(',','').split() if len(pecfns) > 1 else ["X3S-1.dat"]

    n = np.shape(pecfs)[0]

    AM = []
    Rin = []
    Vin = []
    for i,fn in enumerate(pecfs):
        if isinstance(fn, (np.str)):
            radialcoord, potential = np.loadtxt(fn, unpack=True)
            fn = fn.split('/')[-1].upper()
            digits = re.findall('\d', fn)
            if len(digits) > 0:
                degen = int(digits[0])
                S = (degen - 1)//2
                Omega = int(digits[1])
                Lambda = 'SPDF'.index(fn[fn.index(digits[0])+1])
                Sigma = Omega - Lambda
                AM.append((Omega, S, Lambda, Sigma))
            else:
                AM.append((0, 0, 0, 0))

        else:
            radialcoord, potential = fn  # VT=[(R, V), ...] as tuples
            AM.append((0, 0, 0, 0))
            pecfs = ['']
            
        Rin.append(radialcoord)
        Vin.append(potential)
        
    # flatten if only 1 PEC
    if n == 1:
       Rin = np.reshape(Rin, (n, -1))
       Vin = np.reshape(Vin, (n, -1))

    # find internuclear distance min/max domain - some files do not cover 
    # R=0 to 10A, dR=0.005
    Rm  = max([Rin[i][0]    for i in range(n)])   # highest minimum
    Rx  = min([Rin[i][-1]   for i in range(n)])   # lowest maximum
    Vm  = min([Vin[i].min() for i in range(n)])   # lowest potential energy
    Vx = min([Vin[i][-1]   for i in range(n)])   # lowest dissociation limit

    # common internuclear distance grid, that requires no potential
    # curve to be extrapolated
    if R is None:
        dR = Rin[0][-1] - Rin[0][-2]
        dR = round(dR, 1-int(np.floor(np.log10(dR)))-1)
        R = np.arange(Rm, Rx+dR/2, dR)

    oo = len(R)

    # create V matrix, as transpose 
    VT = np.zeros((n, n, oo))
    for j in range(n):
       dRx = (Rin[j][1] - Rin[j][0])/4
       subr = np.logical_and(Rin[j] >= Rm-dRx, Rin[j] <= R[-1]+dRx)
       VT[j, j] = Vin[j][subr]

    limits = (oo, n, Rm, Rx, Vm, Vx)
  
    return R, VT, pecfs, limits, AM


def coupling_function(R, VT, mu, pecfs, coup=None):
    """ Fill the off-diagonal coupling elements of VT.

    Parameters
    ----------
    R : numpy 1d array of floats
    VT : numpy 2D array size nxn of floats
    AM : numpy 1d array of tuples
        (Omega, S, Sigma, Lambda) for each electronic state
    mu : float
        reduced mass
    pecfs: list
        list or potential curve names, to enquire the coupling
    coup : list
        list potential curve couplings (in cm-1) 
    
    """
    # hbar^2/2 mu in eV (once /R^2)
    centrifugal_factor = (const.hbar*1.0e20/mu/2/const.e)*const.hbar
    n, m, oo = VT.shape

    coupling_function = np.ones(np.size(R), dtype='float')
    coupling_function[R>5] = np.exp(-(R[R>5]-5)**2)

    # cm-1 couplings between PECs   (at the moment all homogeneous)
    cnt = 0
    for j in range(n):
        for k in range(j+1,n):
            if coup == None:
                couplestr = input("CSE: coupling {:s} <-> {:s} cm-1 [0]? "\
                                  .format(pecfs[j],pecfs[k]))
                couple = float(couplestr) if len(couplestr) > 1 else 0.0
            else:
                couple = coup[cnt]
                cnt += 1

            VT[j, k] = VT[k, j] = coupling_function*couple/8065.541

    return VT


def load_dipolemoment(dipolemoment=None, R=None, pec_gs=None, pec_us=None):
    def is_number(s):
        try:
            complex(s) # for int, long, float and complex
        except ValueError:
            return False

        return True

    ngs = len(pec_gs)  # number of ground-state potential curves
    nus = len(pec_us)  # upper state curves
    oo = len(R)        # size of internuclear distance grid

    dipole = np.zeros((nus, ngs, oo))
    # loop through all possible ground-state -> upper-state transitions
    for g in np.arange(ngs):
        for u in np.arange(nus):
            i = g*nus + u
            if dipolemoment is not None and len(dipolemoment) > i:
                fn = dipolemoment[i]
            else:
                fn = input("CSE: dipolemoment filename or value {} <- {} : ".
                           format(pec_us[u], pec_gs[g])) 

            if is_number(fn):
                dipole[u][g] = float(fn)             
            else:
                # fn a filename, read and load
                RD, D = np.loadtxt(fn, unpack=True)
                # cubic spline interpolation
                spl = splrep(RD, D)
                dipole[u][g] = splev(R, spl, der=0, ext=1)

    return np.transpose(dipole)
