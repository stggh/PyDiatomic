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


def potential_energy_curves(pecfs=None, R=None, dirpath='./', suffix='',
                            frac_Omega=False):
    """ Read potential energy curve file(s) and assemble as diagonals in
        a nxn array for the n-filenames provided.

    Parameters
    ----------
    pecfs : list of strings
        potential energy curve file name list ['pot1', 'pot2', 'pot3' ... ]
        Each file has 2 column format:  R(Angstroms)  V(eV or cm-1)

    R : numpy 1d array
        radial grid if not None

    dirpath : str
        dirpath to directory of potential energy curve files

    frac_Omega: bool
        Ω specified as 2Ω+1 rather than Ω in the potential energy curve
        filename:
         (2S+1)[S,P,D,F]Ω  vs (2S+1)[S,P,D,F](2Ω+1)  e.g. X2S2.dat Ω=½

    ignore_grid: bool
        

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
        potential curve file names, as inputted

    statelabel : list of str
        symbolic state label for potential curves, (2S+1)ΛΩ e.g. ³Σ₁

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

    if pecfs is None:
        pecfns = input("CSE: potential energy curve(s) [X3S-1.dat]: ")
        if len(pecfns) > 1:
            pecfs = pecfns.replace(',', '').split()
        else:
            pecfs = ['X3S-1.dat']

    n = len(pecfs)
    AM = []
    statelabel = []  # unicode state label 
    Rin = []
    Vin = []
    for i, fn in enumerate(pecfs):
        if isinstance(fn, (str)):
            radialcoord, potential = np.loadtxt(dirpath+'/'+fn+suffix,
                                                unpack=True)
            fn = fn.split('/')[-1].upper()
            
            code = re.findall('\d{1}[SPDF]\S*[+-]*\d{1}', fn)
            if len(code) > 0:
                first_digit = re.search(r'\d', fn)
                if first_digit:
                    prefix = fn[:first_digit.start()]
                    label = prefix
                else:
                    label = ''
                    
                code = code[0]
                degen = int(code[0])
                S = (degen - 1)//2

                Λ = ['S', 'P', 'D', 'F'].index(code[1])
                pm = int(Λ == 0 and '-' in code)

                Ω = int(code[-1])
                if frac_Omega:
                    # (2Ω+1) breaks old notation, allows fractional values
                    Ω = (Ω - 1)/2

                Σ = Ω - Λ
                AM.append((Ω, S, Λ, Σ, pm))
                
                # state label
                label += ['', '¹', '²', '³'][degen] + ['Σ', 'Π', 'Δ', 'Φ'][Λ] +\
                         ['₀', '₁', '₂'][Ω]
                if Λ == 0:
                    label += ['⁺', '⁻'][pm]
                statelabel.append(label)
            else:
                AM.append((0, 0, 0, 0, 0))
                statelabel.append('')

        else:
            radialcoord, potential = fn
            AM.append((0, 0, 0, 0))
            statelabel.append('')

        if potential[-1] > 100:
            potential = potential.copy()  # leave original untouched
            potential /= 8065.541   # convert cm⁻¹ to eV

        Rin.append(radialcoord)
        Vin.append(potential)

    # flatten if only 1 PEC
    if n == 1:
        Rin = np.reshape(Rin, (n, -1))
        Vin = np.reshape(Vin, (n, -1))

    # find internuclear distance min/max domain - some files do not cover
    # R=0 to 10A, dR=0.005
    Rm = max([Rin[i][0] for i in range(n)])  # highest minimum
    Rx = min([Rin[i][-1] for i in range(n)])  # lowest maximum
    Vm = min([Vin[i].min() for i in range(n)])  # lowest potential energy
    Vx = min([Vin[i][-1] for i in range(n)])  # lowest dissociation limit

    # common internuclear distance grid - maps first PEC input, limited by
    # highest Rmin and lowest Rmax of the range
    if R is None:
        R = Rin[0]
        fracdR = (R[1] - R[0])/4  # fractional separation
        # select input internuclear distance between Rm .. Rx
        subR = np.logical_and(R > Rm-fracdR, R < Rx+fracdR)
        R = R[subR]

    # create V matrix, as transpose
    oo = len(R)
    VT = np.zeros((n, n, oo))
    for j in range(n):
        spl = splrep(Rin[j], Vin[j])  # spline representation
        VT[j, j] = splev(R, spl)  # interpolate to common grid - R

    limits = (oo, n, Rm, Rx, Vm, Vx)

    return R, VT, pecfs, limits, AM, statelabel


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
    n, m, oo = VT.shape

    coupling_function = np.ones(np.size(R), dtype=float)
    coupling_function[R > 5] = np.exp(-(R[R > 5] - 5)**2)

    # cm-1 couplings between PECs
    # See Table 3.2 (page 97) Lefebrvre-Brion and Fieldi: Spectra and Dynam
    cnt = 0
    for j in range(n):
        for k in range(j+1, n):
            if coup is None:
                couplestr = input(
                     f'CSE: coupling {pecfs[j]:s} <-> {pecfs[k]:s} cm-1 [0]? ')
                couple = float(couplestr) if len(couplestr) > 1 else 0.0
            elif isinstance(coup[cnt], tuple):
                Rcouple, couple = coup[cnt]
                spl = splrep(Rcouple, couple)
                couple = splev(R, spl, ext=3)
                cnt += 1
            elif isinstance(coup[cnt], str):  # R-dependent file
                Rcouple, couple = np.loadtxt(coup[cnt], unpack=True)
                spl = splrep(Rcouple, couple)
                couple = splev(R, spl, ext=3)
                cnt += 1
            else:
                couple = coup[cnt]
                cnt += 1

            VT[j, k] = VT[k, j] = coupling_function*couple/8065.541

    return VT


def load_dipolemoment(dipolemoment=None, R=None, pec_gs=None, pec_us=None,
                      dirpath='./', suffix=''):
    def is_number(s):
        try:
            complex(s)  # for int, long, float and complex
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
                           f"{pec_us[u]} <- {pec_gs[g]} : ")

            if is_number(fn):
                dipole[u][g] = float(fn)
            else:
                # fn a filename, read and load
                RD, D = np.loadtxt(dirpath+'/'+fn+suffix, unpack=True)
                # cubic spline interpolation
                spl = splrep(RD, D)
                dipole[u][g] = splev(R, spl, der=0, ext=1)

    return np.transpose(dipole)
