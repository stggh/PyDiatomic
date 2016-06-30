# -*- coding: utf-8 -*-
import numpy as np
import scipy.constants as const

def reduced_mass(amu=None):
    u""" Reduced mass of diatomic molecule.

    Parameters
    ----------
    amu : str or float
        reduced mass, str one of 'H2', 'HCl', 'OH', 'N2', 'N14', 'N14N15', 'N15', 'NO', 'O2', 'O16O17', 'O17O18', 'O16O18', 'O18', 'CO', 'Cl2', 'CS2', 'Cs2', 'ICl', 'I2', 'Br', 'S2', 'S32', 'S32O16', 'S33O16', 'S34O16' or float in amu 

    Returns
    -------
    mu : reduced mass in kg

    """

    amus = {\
        u'H2': 0.50391261,
        u'HCl': 0.97959272,
        u'OH': 0.9480871,
        u'N2': 7.0015372,
        u'N14': 7.0015372,
        u'N14N15': 7.242227222,
        u'N15': 7.50005465,
        u'NO': 7.46643323,
        u'O2': 7.99745751,
        u'O16O17': 1.368448e-26/const.u,
        u'O17O18': 1.451734e-26/const.u,
        u'O16O18': 1.40607e-26/const.u,
        u'O18': 1.49417e-26/const.u,
        u'CO': 6.85620871,
        u'Cl2': 17.4844268,
        u'CS2': 8.727,
        u'Cs2': 66.452718,
        u'ICl': 27.4146708,
        u'I2': 63.4522378,
        u'Br': 39.459166,
        u'S2': 15.9860364,
        u'S32': 15.9860364,
        u'S32O16': 10.6613029,
        u'S33O16': 10.77016005,
        u'S34O16': 10.78435767,
        }
    if amu is None: 
        mus = input (u"CSE: reduced mass a.u. [O2=7.99745751]: ")
        if mus in amus.keys():
            amu = amus[mus]
        else:
            amu = float(mus) if len(mus) > 0 else 7.99745751
    elif amu in amus.keys():
        amu = amus[amu] 
    else:
        # atomic mass given
        amu = amu

    return amu*const.m_u


def potential_energy_curves(pecfs=None):
    u""" Read potential energy curve file(s) and assemble as diagonals in an nxn array for the n-filenames provided.

    Parameters
    ----------
    pecfs : list of strings
        potential energy curve file name list ['pot1', 'pot2', 'pot3' ... ]
        Each file has 2 column format:  R(Angstroms)  V(eV)

    Returns
    -------
    R : numpy 1d array 
        radial coordinates

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
        (oo, n, Rm, Rx, Vm, Voo)
        oo = int, common length of radial and potential arrays
        n  = int, number of potential curves
        Rm = highest minimum
        Rx = lowest maximum
        Vm = lowest minimum (Te)
        Voo = lowest dissociation limit energy
    
    """

    if pecfs == None:
        pecfns =  input (u"CSE: potential energy curves [X3S-1.dat]: ")
        pecfs = pecfns.replace(',','').split() if len(pecfns) > 1 else [u"X3S-1.dat"]

    n = np.size(pecfs)

    Rin = []
    Vin = []
    for i,fn in enumerate(pecfs):
        radialcoord, potential = np.loadtxt(fn,unpack=True)
        Rin.append(radialcoord)
        Vin.append(potential)

    # flatten if only 1 PEC
    if n == 1:
       Rin = np.reshape(Rin, (n, -1))
       Vin = np.reshape(Vin, (n, -1))

    # find min/max domain and range - some files do not cover R=0 to 10A
    Rm  = max([Rin[i][0]    for i in range(n)])   # highest minimum
    Rx  = min([Rin[i][-1]   for i in range(n)])   # lowest maximum
    Vm  = min([Vin[i].min() for i in range(n)])   # lowest minimum
    Voo = min([Vin[i][-1]   for i in range(n)])   # lowest dissoc

    oo = list(Rin[0]).index(Rx)

    #print("CSE: {:d} PECs read, Rmin={:5.2f} A, Rmax={:5.2f} A," 
    #      "oo={:d} Te={:5.2f}, De={:5.2f}".format(n,Rm,Rx,oo,Vm,Voo))
    
    # create V matrix, as transpose 
    VT = np.array(np.zeros((n,n,oo)))
    for j in range(n):
       VT[j][j] = Vin[j][:oo]

    # internucler distance matrix
    R = Rin[0][:oo]
    if R[0] < 1.0e-16:
        R[0] = 1.0e-16   # hide 1/0.0

    limits = (oo, n, Rm, Rx, Vm, Voo)
  
    return R, VT, pecfs, limits


def coupling_function(R, VT, pecfs, coup=None):
    u""" Fill the off-diagonal coupling elements of VT.

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

    cf = np.ones(np.size(R), dtype='float')
    cf[R>5] = np.exp(-(R[R>5]-5)**2)

    # cm-1 couplings between PECs   (at the moment all homogeneous)
    cnt = 0
    for j in range(n):
        for k in range(j+1,n):
            if coup == None:
                couplestr = input(u"CSE: coupling {:s} <-> {:s} cm-1 [0]? "\
                                  .format(pecfs[j],pecfs[k]))
                couple = float(couplestr) if len(couplestr) > 1 else 0.0
            else:
                couple = coup[cnt]
                cnt += 1

            VT[j][k] = VT[k][j] = cf*couple/8065.541

    return VT


def load_dipolemoment(dipolemoment=None, R=None, pec_gs=None, pec_us=None):
    def is_number(s):
        try:
            complex(s) # for int, long, float and complex
        except ValueError:
            return False

        return True

    dipole = []
    if dipolemoment is not None:
        for d in dipolemoment:
            if is_number(d):
                dipole.append(np.ones_like(R)*d)
            else:
                RD, D = np.loadtxt(d, unpack=True)
                subr = np.logical_and(RD>=R[0], RD<=R[-1])
                dipole.append(D[subr])
   
    else:
        # query for dipolemoment filename/values
        for g in pec_gs:
            for u in pec_us:
                 fn = input(u"CSE: dipolemoment filename or value {} <- {} : ".
                            format(u, g)) 
                 if is_number(fn):
                     dipole.append(np.ones_like(R)*float(fn))             
                 else:
                     RD, D = np.loadtxt(fn, unpack=True)
                     subr = np.logical_and(RD>=R[0], RD<=R[-1])
                     dipole.append(D[subr])

    return np.transpose(np.array(dipole))
