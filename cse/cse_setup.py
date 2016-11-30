# -*- coding: utf-8 -*-
import numpy as np
import scipy.constants as const

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
        }
    if amu is None: 
        mus = input ("CSE: reduced mass a.u. [O2=7.99745751]: ")
        if mus in amus.keys():
            amu = amus[mus]
        else:
            amu = float(mus) if len(mus) > 0 else 7.99745751
    elif amu in amus.keys():
        amu = amus[amu] 
    else:
        # atomic mass given
        if amu < 1.0e-20:
            return amu   # already in kg

    return amu*const.m_u


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
    
    """

    if pecfs == None:
        pecfns =  input ("CSE: potential energy curves [X3S-1.dat]: ")
        pecfs = pecfns.replace(',','').split() if len(pecfns) > 1 else ["X3S-1.dat"]

    n = np.shape(pecfs)[0]

    Rin = []
    Vin = []
    for i,fn in enumerate(pecfs):
        if isinstance(fn, (np.str)):
            radialcoord, potential = np.loadtxt(fn,unpack=True)
        else:
            radialcoord, potential = fn
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
        R = np.arange(Rm, Rx+dR/2, dR)

    oo = len(R)

    # create V matrix, as transpose 
    VT = np.array(np.zeros((n, n, oo)))
    for j in range(n):
       subr = np.logical_and(Rin[j] >= Rm, Rin[j] <= R[-1])
       VT[j][j] = Vin[j][subr]

    limits = (oo, n, Rm, Rx, Vm, Vx)
  
    return R, VT, pecfs, limits


def coupling_function(R, VT, pecfs, coup=None):
    """ Fill the off-diagonal coupling elements of VT.

    Parameters
    ----------
    R : numpy 1d array of floats
    VT : numpy 2D array size nxn of floats
    pecfs: list
        list or potential curve names, to enquire the coupling
    coup : list
        list potential curve couplings (in cm-1) 
    
    """
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

            VT[j][k] = VT[k][j] = coupling_function*couple/8065.541

    return VT


def load_dipolemoment(dipolemoment=None, R=None, pec_gs=None, pec_us=None):
    def is_number(s):
        try:
            complex(s) # for int, long, float and complex
        except ValueError:
            return False

        return True

    ngs = len(pec_gs)
    nus = len(pec_us)
    oo = len(R)

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
                # fn is a number
                dipole[u][g] = float(fn)             
            else:
                # fn a filename
                RD, D = np.loadtxt(fn, unpack=True)
                mn = np.abs(RD[0] - R[0]).argmin()
                mx = np.abs(RD[-1] - R[-1]).argmin() + 1
                dipole[u][g][mn:mx] = D

    return np.transpose(dipole)
