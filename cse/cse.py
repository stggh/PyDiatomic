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
    Bv : float
        evaluated rotational constant (if single wavefunction)
    Bvs : float array
        rotational constans v = 0, ..., vmax (from self.levels(vmax))
    R  : float array
        internuclear distance grid
        If `R=None` set to `numpy.arange(Rmin, Rmax+dR/2, dR)`
        where Rmin = highest minimum, Rmax = lowest maximum
        of all the potential curves
    VT : numpy 3d array
        transpose of the potential curve and couplings array
        Note: potential curves spline interpolated to grid `R`
    cm : float
        eigen energy in cm-1 (from method solve)
    energy : float
        eigenvalue energy in eV
    energies : numpy float array
        eigenvalues v = 0, ..., vmax (from self.levels())
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

        if self.limits[1] == 1:
            if self.energy < self.VT[0][0][-1]:
                self.node_count()
                self.Bv = expectation.Bv(self)
            else:
                self.vib = -1

        self.cm = self.energy*self._evcm

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
        Te = V.min()
        Too = V[-1]
        trial = np.linspace(Te*self._evcm+100, Too*self._evcm-10, ntrial)

        actual = []
        vib = []
        Bv = []
        for en in trial:
            self.solve(en)
            actual.append(self.cm)
            vib.append(self.vib)
            Bv.append(self.Bv)

        if vmax is None:
            vmax = vib[-1]
        v = np.arange(vmax+1)

        spl = interpolate.interp1d(vib, actual, kind='cubic')
        self.energies = spl(v)

        splB = interpolate.interp1d(vib, Bv, kind='cubic')
        self.Bvs = splB(v)

        if exact:
            for level, en in enumerate(self.energies):
                self.solve(en)
                self.energies[level] = self.cm
                self.Bvs[level] = self.Bv

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

        done = False
        try:
            e0 = self.energies[0]
            about += "Eigenvalues:  v    energy(cm-1)    Bv(cm-1)\n"
            for v, en in enumerate(self.energies):
                about += "             {:2d}    {:10.3f}     {:8.5f}\n"\
                         .format(v, en, self.Bvs[v])
            done = True
        except AttributeError:
            pass

        if not done:
            try:
                about += "Eigenvalue: {:g} cm-1,  v = {:d}, Bv = {:8.5f} cm-1"\
                         .format(self.cm, self.vib, self.Bv)
            except AttributeError:
                pass

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
