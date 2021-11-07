# -*- coding: utf-8 -*-
import numpy as np
import scipy.linalg as scla
import scipy.constants as const
from scipy import interpolate
from collections import OrderedDict

from . import johnson
from . import expectation
from . import cse_setup
from . import tools


class Cse():
    """ Class to setup and solve the TISE via the Johnson renormalized
        Numerov method i.e. drive johnson.py for a single
        set of coupled states.

    The following attributes may be available subject to the calculation.
    "print(classvariable)" for its representation.

    Attributes
    ----------
    R  : float array
        internuclear distance grid
        If `R=None` set to `numpy.arange(Rmin, Rmax+dR/2, dR)`
        where Rmin = highest minimum, Rmax = lowest maximum
        of all the potential curves

    dirpath : str
        path to directory conatining the potential energy curve files

    suffix : str
        suffix appended to potential energy curve file name

    VT : numpy 3d array
        transpose of the potential curve and couplings array
        Note: potential curves spline interpolated to grid `R`

    Bv : float
        evaluated rotational constant (if single wavefunction)

    cm : float
        eigen energy in cm-1 (from method solve)

    energy : float
        eigenvalue energy in eV

    results : dict
        single state calculation results  {vib: (energy, Bv, Dv, Jrot)} in cm-1
        (see also class representation)

    molecule: str
        chemical formula str

    AM : 1d array of tuples
        Angular momenta quantum numbers (Ω, S, Λ, Σ) for each electronic state

    limits : tuple
        array sizes: (oo, n, Rmin, Rmax, Vmin, Te)

    μ : str or float
        molecule formula str or reduced mass in amu or kg 

    rot : int
        total angular momentum quantum number

    vib : int
        vibrational quantum number (if available)

    wavefunction : numpy array
        wavefunction(s) for all channels

    wavenumber: numpy float
        wavenumber eigenvalue of the solution

    """


    def __init__(self, μ=None, R=None, VT=None, coup=None, eigenbound=None,
                 en=None, rot=0, dirpath='./', suffix=''):

        self._evcm = 8065.541
        self.set_μ(μ=μ)
        self.rot = rot

        if eigenbound is None:
            self.eigenbound = 500/self._evcm  # eV

        if R is not None:
            # PEC array provided directly
            self.R = R
            self.VT = VT
            n, m, oo = VT.shape
            self.pecfs = [' ']
            self.limits = (oo, n, R[0], R[-1], VT[0][0].min(), VT[0][0][-1])
        else:
            # list of file names provided in VT
            self.R, self.VT, self.pecfs, self.limits, self.AM =\
                    cse_setup.potential_energy_curves(VT, dirpath=dirpath,
                                                      suffix=suffix)
            self.set_coupling(coup=coup)

        # fudge to eliminate 1/0 error for 1/R^2
        zeros = np.abs(self.R) < 1.0e-16
        if np.any(zeros):
            self.R[zeros] = 1.0e-16

        self.results = OrderedDict()  # store results for single bound channel
        if en is not None:
            self.solve(en, self.rot)

    def set_μ(self, μ):
        self.μ, self.molecule = cse_setup.reduced_mass(μ)

    def set_coupling(self, coup):
        self.VT = cse_setup.coupling_function(self.R, self.VT, self.μ,
                                              self.pecfs, coup=coup)

    def solve(self, en, rot=None, eigenbound=None):
        """ solve the Schrodinger equation for the (coupled) potential(s).

        Parameters
        ----------
        en : float
            (initial) solution energy
        rot : int
            total angular momentum (excluding nuclear)
        eigenbound : float
            bound-eigenvalue limits en+-eigenbound - same units as en
            This helps scipy.optimize.least_squares() from straying 
        """

        if en > 20:
            en /= self._evcm   # convert to eV energy unit
            if eigenbound is not None:
                eigenbound /= self._evcm

        if rot is not None:
            self.rot = rot   # in case called separately

        if eigenbound is not None:
            self.eigenbound = eigenbound

        johnson.solveCSE(self, en)

        self.cm = self.energy*self._evcm

        if self.limits[1] == 1:  # single channel save more detail
            if self.energy < self.VT[0][0][-1]:
                self.node_count()  # could use det(R) of matching point instead
                self.Bv = expectation.Bv(self)
                self.Dv = expectation.Dv(self)
                # keep results
                self.results[self.vib] = (self.cm, self.Bv, self.Dv, self.rot)
            else:
                self.vib = None 

    def node_count(self):
        V = self.VT[0][0][:np.shape(self.wavefunction)[0]]

        wfx = self.wavefunction[self.energy > V]  # inside well

        vib = (wfx[1:]*wfx[:-1] < 0).sum()

        self.vib = vib
        return vib

    def levels(self, vmax=None, ntrial=5, exact=True):
        """ Evaluate the vibrational energies of a potential energy curve.

        method spline interpolate v = 0, ..., vmax from the eigenvalue
        solutions for the initial guess energies (Vdissoc - Vmin)/ntrial

        Parameters
        ----------
        vmax : int
            maximum vibrational quantum number at which to evaluate the spline

        ntrial : int
            number of trial initial energies

        exact : boolean
            solve TISE for v = 0, ..., vmax, using the interpolated guesses

        Returns
        -------
        results : dict of solutions
            attribute .results = {vib: (energy, Bv, Dv, rot)}

        """
        V = np.transpose(self.VT[0][0])

        for en in np.linspace(V.min()*self._evcm+100,
                              V[-1]*self._evcm-10, ntrial):
            self.solve(en)
            if vmax is not None and self.vib > vmax and len(self.results) > 3:
                break  # don't waste time

        maxv = max(list(self.results.keys()))
        if vmax is None:
            vmax = maxv
        else:
            vmax = min(maxv, vmax)  # no extrapolation

        # interpolate calculation
        v = np.arange(vmax+1)

        vib = sorted(self.results.keys())
        actual = [self.results[vi][0] for vi in vib]
        Bv = [self.results[vi][1] for vi in vib]
        Dv = [self.results[vi][2] for vi in vib]

        spl = interpolate.interp1d(vib, actual, kind='cubic')
        splB = interpolate.interp1d(vib, Bv, kind='cubic')
        splD = interpolate.interp1d(vib, Dv, kind='cubic')

        for vi in v:
            self.results[vi] = (float(spl(vi)), float(splB(vi)),
                                float(splD(vi)), self.rot)

        if exact:
            for v, (en, Bv, Dv, rot) in list(self.results.items()):
                self.solve(en, rot)

        # sort in order of energy
        self.results = OrderedDict(sorted(self.results.items(),
                                   key=lambda t: t[1]))

    def diabatic2adiabatic(self):
        """ Convert diabatic interaction matrix to adiabatic (diagonal)
            A = UT V U     unitary transformation

        """
        V = np.transpose(self.VT)
        A = np.zeros_like(V)
        diag = np.diag_indices(A.shape[1])
        for i, Vi in enumerate(V):
            w, U = scla.eigh(Vi)
            A[i][diag] = w

        self.AT = np.transpose(A)

    def __repr__(self):
        n = self.limits[1]
        about = '\n' + f'Molecule: {self.molecule}'
        about += f'  mass: {self.μ:g} kg, {self.μ/const.u:g} amu\n'
        about += f"Electronic state{'s' if n > 1 else '':s}:"
        if isinstance(self.pecfs[0], str):
            for fn in self.pecfs:
                about += f' {fn:s}'
            about += '\n'
        if n > 1:
            about += f'Coupling at R = {self.R[240]:g} Angstroms (cm-1):'
            for i in range(n):
                for j in range(i+1, n):
                    about += f' {self.VT[i, j, 240]*8065.541:g}'

        if len(self.results) > 0:
            about += "eigenvalues (that have been evaluated for this state):\n"
            about += " v  rot   energy(cm-1)    Bv(cm-1)     Dv(cm-1)\n"
            vib = sorted(list(self.results.keys()))
            for v in vib:
                about += f'{v:2d}  {self.results[v][3]:2d}   '
                about += f'{self.results[v][0]:10.3f}     '
                about += f'{self.results[v][1]:9.5f}'
                about += f'{self.results[v][2]:15.3e}\n'
        return about


class Xs():
    """ Class to evaluate photodissociation cross sections, i.e. solve the
    coupled-channel problems, for initial and final coupled-channels.

    *Note*: `cse.Transition()` to set up the transition states first.

    The following attributes may be available subject to the calculation.

    Attributes
    ----------
    wavenumber : numpy float array
        wavenumber range of calculation
    xs : numpy float array
        photodissociation cross section for each open channel


    :note: coupled-channel attributes for initial (gs) and final (us)
           coupled-states, attributes as listed for :class:`Cse()`.

    """

    def __init__(self, μ=None, dirpath='./', suffix='',
                 Ri=None, VTi=None, coupi=None, eni=0, roti=0,
                 Rf=None, VTf=None, coupf=None, rotf=0,
                 dipolemoment=None, transition_energy=None):

        self._evcm = 8065.541

        # ground state
        self.gs = Cse(μ=μ, R=Ri, VT=VTi, coup=coupi, rot=roti, en=eni)

        # upper state
        self.us = Cse(μ=self.gs.μ, R=Rf, VT=VTf, coup=coupf,
                      rot=rotf, en=0)

        self.align_grids()

        # electronic transition moment
        self.dipolemoment = cse_setup.load_dipolemoment(
                                dipolemoment=dipolemoment,
                                R=self.us.R, pec_gs=self.gs.pecfs,
                                pec_us=self.us.pecfs, dirpath=dirpath,
                                suffix=suffix)

        if transition_energy is not None:
            self.calculate_xs(transition_energy)


    def set_param(self, μ=None, eni=None, coupi=None, roti=None,
                                          coupf=None, rotf=None):

        if μ is not None or coupi is not None or roti is not None:
            # recalculate initial (coupled) state(s)
            if μ is not None:
                self.gs.set_μ(μ)
            if coupi is not None:
                self.gs.set_coupling(coupi)

            if eni is None:
                eni = self.gs.energy

            self.gs.solve(eni, roti)

        if μ is not None or coupf is not None or rotf is not None:
            # recalculate final couples states
            if μ is not None:
                self.us.set_μ(μ)
            if coupi is not None:
                self.us.set_coupling(coupf)

    def calculate_xs(self, transition_energy, eni=None, roti=None, rotf=None,
                     honl=False):
        transition_energy = np.array(transition_energy)
        emax = np.abs(transition_energy).max()
        if emax < 50:
            # energy unit is eV
            self.wavenumber = transition_energy*self._evcm
        elif emax < 500:
            # energy unit is nm wavelength
            self.wavenumber = 1.0e7/transition_energy
        else:
            # energy unit is wavenumber
            self.wavenumber = transition_energy

        if eni is not None or roti is not None:
            if eni is None:
                eni = self.gs.cm
            self.gs.solve(eni, roti)

        if rotf is not None:
            self.us.rot = rotf

        if honl:
            # Hönl-London factor J' J" Ω' Ω" for the main allowed transition
            # fix me - this should be within expectation calculation
            #          at the rotational matrix level
            self.honl = tools.intensity.honl(self.us.rot, self.gs.rot,
                                             self.us.AM[0][0], self.gs.AM[0][0])
            # self.eta = 
        else:
            self.honl = 1

        if self.honl > 0:
            xswav = expectation.xs_vs_wav(self)
            self.xs, self.wavenumber = zip(*xswav)
            self.xs = np.array(self.xs)*self.honl
            self.wavenumber = np.array(self.wavenumber)
            self.oci = np.any(self.xs > 0, axis=0)  # open channels xs > 0
            self.nopen = self.oci.sum()
        else: # don't waste time on the calculation
            self.xs = np.zeros((len(transition_energy), self.us.VT.shape[0]))


    def align_grids(self):
        """ ensure the same internuclear grid for each block
            of coupled-channels, ground-states vs upper-states.
            NB assumes common dR for each grid.

        """

        # limits = (oo, n, Rm, Rx, Vm, Vx)
        if self.us.limits[0] != self.gs.limits[0]:
            Rm = max(self.us.limits[2], self.gs.limits[2])  # Rm
            Rx = min(self.us.limits[3], self.gs.limits[3])  # Rx
            for state in [self.gs, self.us]:
                _, n, _, _, Vm, Vx = state.limits
                subr = np.logical_and(state.R >= Rm, state.R <= Rx)
                state.R = state.R[subr]
                V = np.transpose(state.VT)
                V = V[subr]
                state.VT = np.transpose(V)
                oo = len(state.R)
                state.limits = (oo, n, Rm, Rx, Vm, Vx)

    def __repr__(self):
        return self.gs.__repr__() + '\n' + self.us.__repr__()


class Transition(Xs):
    """A simpler interface to evaluate transitions between
       initial and final coupled states, replacing class:Xs().

    """

    def __init__(self, final_coupled_states, initial_coupled_states,
                 dipolemoment=None, transition_energy=None, eni=None,
                 roti=None, rotf=None, dirpath='./', suffix=''):
        """
        Parameters
        ----------
        initial_coupled_states: `cse.Cse` class
            initial (coupled)electronic state(s)

        final_coupled_states: `cse.Cse` class
            final (coupled)electronic state(s)

        dipolemoment: float array
            electric dipole transition moment (in a.u.)

        dirpath : str
            path to the directory containing the electric dipole transition
            moment file

        suffix : str
            suffix appended to electric dipole transition moment file name

        """

        self.gs = initial_coupled_states
        self.us = final_coupled_states

        self.align_grids()

        # electronic transition moment
        self.dipolemoment = cse_setup.load_dipolemoment(
                                dipolemoment=dipolemoment,
                                R=self.us.R, pec_gs=self.gs.pecfs,
                                pec_us=self.us.pecfs, dirpath=dirpath,
                                suffix=suffix)

        if transition_energy is not None:
            self.calculate_xs(transition_energy, roti=roti, rotf=rotf, eni=eni)
