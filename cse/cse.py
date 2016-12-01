# -*- coding: utf-8 -*-
import numpy as np

from . import johnson
from . import expectation
from . import cse_setup

class Cse():
    """ Class to setup and solve the TISE via the Johnson renormalized
        Numerov method i.e. drive johnson.py for a single
        set of coupled states.

    The following attributes may be available subject to the calculation.

    Attributes
    ----------
    Bv : float
        evaluated rotational constant (if single wavefunction)
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
        eigen energy in eV
    limits : tuple
        array sizes: (oo, n, Rmin, Rmax, Vmin, Te)
    mu : float
        reduced mass in kg
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
        else:
            # list of file names provided in VT
            self.R, self.VT, self.pecfs, self.limits =\
                    cse_setup.potential_energy_curves(VT)

        self.set_coupling(coup=coup)

        # fudge to eliminate 1/0 error for 1/R^2
        if self.R[0] < 1.0e-16:
            self.R[0] = 1.0e-16

        if en > 0:
            self.solve(en, self.rot)

    def set_mu(self, mu):
        self.mu = cse_setup.reduced_mass(mu)

    def set_coupling(self, coup):
        self.VT = cse_setup.coupling_function(self.R, self.VT, self.pecfs,
                                              coup=coup)
    def solve(self, en, rot=None):
        if en > 20:
            en /= self._evcm   # convert to eV energy units

        if rot is not None:
            self.rot = rot   # in case called separately

        self.wavefunction, self.energy = johnson.solveCSE(en, self.rot,
                                                 self.mu, self.R, self.VT)

        if self.limits[1] == 1:
            if self.energy < self.VT[0][0][-1]:
                self.node_count()
                self.rotational_constant()
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

    def rotational_constant(self):
        wf = self.wavefunction[:, 0, 0]
        self.Bv = expectation.Bv(self.R, wf, self.mu)


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
                                dipolemoment=None, transition_energy=None):

        # ground state
        self.gs = Cse(mu=mu, R=Ri, VT=VTi, coup=coupi, rot=roti, en=eni)

        # upper state
        self.us = Cse(mu=self.gs.mu, R=Rf, VT=VTf, coup=coupf,
                      rot=rotf, en=0)

        # electronic transition moment
        self.dipolemoment = cse_setup.load_dipolemoment(
                                dipolemoment=dipolemoment,
                                R=self.us.R, pec_gs=self.gs.pecfs,
                                pec_us=self.us.pecfs)

        if transition_energy is not None:
            self.calculate_xs(transition_energy)

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
            print('E(v"={}) = {:.2f} cm-1 '.format(self.gs.vib, self.gs.cm))

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

        self.xs = expectation.xs_vs_wav(self.wavenumber, self.dipolemoment,
                                        self.gs.energy, self.us.rot,
                                        self.gs.mu,
                                        self.us.R, self.us.VT,
                                        self.gs.wavefunction)
        self.nopen = self.xs.shape[-1]
