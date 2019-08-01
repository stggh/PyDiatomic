# -*- coding: utf-8 -*-
import numpy as np
import scipy.constants as const
from periodictable import elements
import re
from scipy.interpolate import splrep, splev

def atomic_mass(atom_symbol, isotope=''):
   """atomic mass for atom_symbol, return most abundant atomic mass
      if isotope not given.
   """ 
   elem = elements.isotope(atom_symbol)
   if isotope != '': 
        mass = elem[int(isotope)].mass   # have isotopic mass
   else:
       # most abundant isotopic mass
       elem_abund = []
       for iso in elem.isotopes:
           elem_abund.append(elem[iso].abundance)
       elem_abund = np.asarray(elem_abund)
       mass = elem[elem.isotopes[elem_abund.argmax()]].mass
   return mass


def reduced_mass(molecule):
    """ Reduced mass of diatomic molecule.

    Parameters
    ----------
    molecule : formula str or float value for reduced mass amu or kg
        e.g. O2' or '16O16O' or '16O2' or 'O2', '32S18O' etc. 
        For '{atom}2' the most abundance isotope determines the mass

    Returns
    -------
    μ : diatomic reduced mass in kg

    molecule : str
        molecule formula as input

    """

    if isinstance(molecule, float) or isinstance(molecule, int):
        μ = molecule
        if μ < 1:
            μ /= const.u
        molecule = 'unknown'
    else:
        # from https://stackoverflow.com/questions/41818916
        # /calculate-molecular-weight-based-on-chemical-formula-using-python
        # array of tuples [('34', 'S'), ('16', 'O')]
        atoms = re.findall('([0-9]*)([A-Z][a-z]?)', molecule)

        m1 = atomic_mass(atoms[0][1], atoms[0][0])

        if len(atoms) == 1:
            m2 = m1
        else:
            m2 = atomic_mass(atoms[1][1], atoms[1][0])

        μ = m1*m2/(m1 + m2)

    return μ*const.u, molecule


def potential_energy_curves(pecfs=None, R=None):
    """ Read potential energy curve file(s) and assemble as diagonals in an nxn array for the n-filenames provided.

    Parameters
    ----------
    pecfs : list of strings
        potential energy curve file name list ['pot1', 'pot2', 'pot3' ... ]
        Each file has 2 column format:  R(Angstroms)  V(eV or cm-1)

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
        Angular momenta quantum numbers (Ω, S, Λ, Σ) for each electronic state
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
                Ω = int(digits[1])
                Λ = 'SPDF'.index(fn[fn.index(digits[0])+1])
                Σ = Ω - Λ
                AM.append((Ω, S, Λ, Σ))
            else:
                AM.append((0, 0, 0, 0))

        else:
            radialcoord, potential = fn
            AM.append((0, 0, 0, 0))

        if potential[-1] > 100:
            potential /= 8065.541   # convert cm-1 to eV

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
    # curve is extrapolated
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


def coupling_function(R, VT, μ, pecfs, coup=None):
    """ Fill the off-diagonal coupling elements of VT.

    Parameters
    ----------
    R : numpy 1d array of floats
    VT : numpy 2D array size nxn of floats
    AM : numpy 1d array of tuples
        (Ω, S, Σ, Λ) for each electronic state
    μ : float
        reduced mass
    pecfs: list
        list or potential curve names, to enquire the coupling
    coup : list
        list potential curve couplings (in cm-1) 
    
    """
    # hbar^2/2μ in eV (once /R^2)
    centrifugal_factor = (const.hbar*1.0e20/μ/2/const.e)*const.hbar
    n, m, oo = VT.shape

    coupling_function = np.ones(np.size(R), dtype='float')
    coupling_function[R>5] = np.exp(-(R[R>5]-5)**2)

    # cm-1 couplings between PECs   (at the moment all homogeneous)
    cnt = 0
    for j in range(n):
        for k in range(j+1,n):
            if coup == None:
                couplestr = input(
                     f'CSE: coupling {pecfs[j]:s} <-> {pecfs[k]:s} cm-1 [0]? ')
                couple = float(couplestr) if len(couplestr) > 1 else 0.0
            elif isinstance(coup[cnt], tuple):
                Rcouple, couple = coup[cnt]
                spl = splrep(Rcouple, couple)
                couple = splev(R, spl, ext=3) 
                cnt += 1
            elif isinstance(coup[cnt], str):  # dipolemoment file
                Rcouple, couple = np.loadtxt(coup[cnt], unpack=True)
                spl = splrep(Rcouple, couple)
                couple = splev(R, spl, ext=3) 
                cnt += 1
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
                fn = input("CSE: dipolemoment filename or value "
                           "{pec_us[u]} <- {pec_gs[g]} : ")

            if is_number(fn):
                dipole[u][g] = float(fn)             
            else:
                # fn a filename, read and load
                RD, D = np.loadtxt(fn, unpack=True)
                # cubic spline interpolation
                spl = splrep(RD, D)
                dipole[u][g] = splev(R, spl, der=0, ext=1)

    return np.transpose(dipole)
