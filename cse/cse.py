# -*- coding: utf-8 -*-
import numpy as np
import scipy.linalg as scla
from scipy import interpolate

from . import johnson
from . import expectation
from . import cse_setup


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
    VT : numpy 3d array
        transpose of the potential curve and couplings array
        Note: potential curves spline interpolated to grid `R`
    Bv : float
        evaluated rotational constant (if single wavefunction)
    cm : float
        eigen energy in cm-1 (from method solve)
    energy : float
        eigenvalue energy in eV
    calc : dict
        single state calculation results  {vib: (energy, Bv)} in cm-1
        (see also class representation)
    limits : tuple
        array sizes: (oo, n, Rmin, Rmax, Vmin, Te)
    mu : float
        reduced mass in kg
    rot : int
        total angular momentum quantum number
    vib : int
        vibrational quantum number (if available)
    wavefunction : numpy array
        wavefunction(s) for all channels
    wavenumber: numpy float
        wavenumber eigenvalue of the solution

    """

    _evcm = 8065.541

    def __init__(self, mu=None, R=None, VT=None, coup=None, rot=0, en=0):

        self.set_mu(mu=mu)
        self.rot = rot

        if R is not None:
            # PEC array provided directly
            self.R = R
            self.VT = VT
            n, m, oo = VT.shape
            self.limits = (oo, n, R[0], R[-1], VT[0][0].min(), VT[0][0][-1])
        else:
            # list of file names provided in VT
            self.R, self.VT, self.pecfs, self.limits, self.AM =\
                    cse_setup.potential_energy_curves(VT)
            self.set_coupling(coup=coup)

        # fudge to eliminate 1/0 error for 1/R^2
        zeros = np.abs(self.R) < 1.0e-16
        if np.any(zeros):
            self.R[zeros] = 1.0e-16

        self.calc = {}  # store results
        if en > 0:
            self.solve(en, self.rot)

    def set_mu(self, mu):
        self.mu, self.molecule = cse_setup.reduced_mass(mu)

    def set_coupling(self, coup):
        self.VT = cse_setup.coupling_function(self.R, self.VT, self.mu,
                                              self.pecfs, coup=coup)

    def solve(self, en, rot=None):
        if en > 20:
            en /= self._evcm   # convert to eV energy units

        if rot is not None:
            self.rot = rot   # in case called separately

        self.wavefunction, self.energy, self.openchann = \
            johnson.solveCSE(self, en)

        self.cm = self.energy*self._evcm

        if self.limits[1] == 1:
            if self.energy < self.VT[0][0][-1]:
                self.node_count()
                self.Bv = expectation.Bv(self)
                self.calc[self.vib] = (self.cm, self.Bv)
            else:
                self.vib = -1


    def node_count(self):
        V = self.VT[0][0][:np.shape(self.wavefunction)[0]]

        wfx = self.wavefunction[self.energy > V]  # inside well

        vib = 0
        for i, wfxi in enumerate(wfx[:-1]):
            if (wfxi > 0 and wfx[i+1] < 0) or (wfxi < 0 and wfx[i+1] > 0):
                vib += 1

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
        energies : numpy 1D array
           eigenvalues (cm-1) for v = 0, ..., vmax

        Bvs : numpy 1D array
           rotational constants (cm-1)

        """
        V = np.transpose(self.VT[0][0])

        for en in np.linspace(V.min()*self._evcm+100,
                              V[-1]*self._evcm-10, ntrial):
            self.solve(en)
            if vmax is not None and self.vib > vmax and len(self.calc) > 3:
               break  # don't waste time

        maxv = max(list(self.calc.keys()))
        if vmax is None:
            vmax = maxv
        else:
            vmax = min(maxv, vmax)  # no extrapolation

        # interpolate calculation
        v = np.arange(vmax+1)

        vib = sorted(self.calc.keys())
        actual = [self.calc[vi][0] for vi in vib]
        Bv = [self.calc[vi][1] for vi in vib]

        spl = interpolate.interp1d(vib, actual, kind='cubic')
        splB = interpolate.interp1d(vib, Bv, kind='cubic')

        for vi in v:
            self.calc[vi] = (float(spl(vi)), float(splB(vi)))

        if exact:
            for en, Bv in list(self.calc.items()):
                self.solve(en)


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
        about = '\n' + "Molecule: {}".format(self.molecule)
        about += "  mass: {:g} kg\n".format(self.mu)
        about += "Electronic state{:s}:"\
                 .format('s' if n > 1 else '')
        for fn in self.pecfs:
            about += " {:s}".format(fn)
        about += '\n'
        if n > 1:
            about += "Coupling at R = {:g} Angstroms (cm-1):"\
                     .format(self.R[240])
            for i in range(n):
                for j in range(i+1, n):
                    about += " {:g}".format(self.VT[i, j, 240]*8065.541)

        if len(self.calc) > 0:
            about += "eigenvalues (that have been evaluated for this state):\n"
            about += " v    energy(cm-1)    Bv(cm-1)\n"
            vib = sorted(list(self.calc.keys()))
            for v in vib: 
                about += "{:2d}    {:10.3f}     {:8.5f}\n"\
                         .format(v, self.calc[v][0], self.calc[v][1])

        return about


class Xs():
    """ Class to evaluate photodissociation cross sections, i.e. solve the
    coupled-channel problems, for initial and final coupled-channels.

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

    def __init__(self, mu=None, Ri=None, VTi=None, coupi=None, eni=0, roti=0,
                                Rf=None, VTf=None, coupf=None, rotf=0,
                                dipolemoment=None, transition_energy=None,
                                honl=False):

        # ground state
        self.gs = Cse(mu=mu, R=Ri, VT=VTi, coup=coupi, rot=roti, en=eni)

        # upper state
        self.us = Cse(mu=self.gs.mu, R=Rf, VT=VTf, coup=coupf,
                      rot=rotf, en=0)

        self.align_grids()

        # electronic transition moment
        self.dipolemoment = cse_setup.load_dipolemoment(
                                dipolemoment=dipolemoment,
                                R=self.us.R, pec_gs=self.gs.pecfs,
                                pec_us=self.us.pecfs)

        if transition_energy is not None:
            self.calculate_xs(transition_energy)

        self.honl = honl

    def set_param(self, mu=None, eni=None, coupi=None, roti=None,
                                           coupf=None, rotf=None):

        if mu is not None or coupi is not None or roti is not None:
            # recalculate initial (coupled) state(s)
            if mu is not None:
                self.gs.set_mu(mu)
            if coupi is not None:
                self.gs.set_coupling(coupi)

            if eni is None:
                eni = self.gs.energy

            self.gs.solve(eni, roti)
            # print('E(v"={}, J"={}) = {:.2f} cm-1 '.
            #       format(self.gs.vib, roti, self.gs.cm))

        if mu is not None or coupf is not None or rotf is not None:
            # recalculate final couples states
            if mu is not None:
                self.us.set_mu(mu)
            if coupi is not None:
                self.us.set_coupling(coupf)

    def calculate_xs(self, transition_energy, eni=None, roti=None, rotf=None):
        transition_energy = np.array(transition_energy)
        emax = transition_energy.max()
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
            self.gs.solve(eni, roti)

        if rotf is not None:
            self.us.rot = rotf

        self.xs = expectation.xs_vs_wav(self)
        self.nopen = self.xs.shape[-1]

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
