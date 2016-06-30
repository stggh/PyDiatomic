PyCSE README
============


Introduction
------------

PyCSE solves the time-independent coupled-channel Schroedinger equation
using the Johnson renormalized Numerov method [1]. This is very compact and stable algorithm.

The PyCSE code is oriented toward the calculation of diatomic molecular photodissociation cross sections. Coupling of electronic states results in broadening and line-shape asymmetry. A coupled-channel calculation is the only correct method compute the cross-section.



Installation
------------

PyCSE requires Python 3, numpy and scipy. If you don't already have Python, we recommend an "all in one" Python package such as the `Anaconda Python Distribution <https://www.continuum.io/downloads>`_, which is available for free.

Download the latest version from github ::

    git clone https://github.com/PyCSE/PyCSE.git

`cd`  to the PyCSE directory, and use ::

    python3 setup.py install --user

Or, if you wish to edit the PyAbel source code without re-installing each time ::

    python3 setup.py develop --user


Example of use
--------------

PyCSE has a wrapper classes :class:`cse.Cse()` and :class:`cse.Xs()` 

:class:`cse.Cse()` set ups the CSE problem 
(interaction matrix of potential energy curves, and couplings) and solves 
the coupled channel Schr\"odinger equation for an initial guess energy.

Input parameters may be specified in the class instance, or they will be 
requested if required.

.. code-block:: python

   import cse
   X = cse.Cse()   # class instance
   # CSE: reduced mass a.u. [O2=7.99745751]:    # requested parameters
   # CSE: potential energy curves [X3S-1.dat]:
   X.solve(800)    # solves TISE for energy ~ 800 cm-1
   # attributes
   # X.Bv                   X.mu                   X.set_mu
   # X.R                    X.node_count           X.solve
   # X.VT                   X.pecfs                X.vib
   # X.cm                   X.rot                  X.wavefunction
   # X.energy               X.rotational_constant  
   # X.limits               X.set_coupling       
   X.cm
   # 787.3978354211097
   X.vib
   # 0


:class:`cse.Xs()` evaluates two couple channel problems, for an intitial 
and final set of coupled channels, to calculate the photodissociation 
cross section.

.. code-block:: python

   import numpy as np
   import cse
   Y = cse.Xs()
   # CSE: reduced mass a.u. [O2=7.99745751]: 
   # CSE: potential energy curves [X3S-1.dat]: 
   # CSE: potential energy curves [X3S-1.dat]: B3S-1.dat, E3S-1.dat
   # CSE: coupling B3S-1.dat <-> E3S-1.dat cm-1 [0]? 4000
   # CSE: dipolemoment filename or value B3S-1.dat <- X3S-1.dat : 1
   # CSE: dipolemoment filename or value E3S-1.dat <- X3S-1.dat : 0
   Y.calculate_xs(transition_energy=np.arange(110, 174, 0.1), eni=800)
   # attributes
   # Y.calculate_xs  Y.gs            Y.set_param     Y.xs
   # Y.dipolemoment  Y.nopen         Y.us            Y.wavenumber  
   # and those associated with the initial and final states
   # 
   # Y.gs.Bv                   Y.gs.mu                   Y.gs.set_mu
   # Y.gs.R                    Y.gs.node_count           Y.gs.solve
   # Y.gs.VT                   Y.gs.pecfs                Y.gs.vib
   # Y.gs.cm                   Y.gs.rot                  Y.gs.wavefunction
   # Y.gs.energy               Y.gs.rotational_constant  
   # Y.gs.limits               Y.gs.set_coupling      
   # 
   # Y.us.R                    Y.us.node_count           Y.us.set_coupling
   # Y.us.VT                   Y.us.pecfs                Y.us.set_mu
   # Y.us.limits               Y.us.rot                  Y.us.solve
   # Y.us.mu                   Y.us.rotational_constant  

A simple :math:`^3\Sigma_{u}^{-} \leftrightarrow {}^3\Sigma^{-}_{u}` Rydberg-valence coupling in O\ :sub:`2`

.. code-block:: python

   import numpy as np
   import cse
   import matplotlib.pyplot as plt

   Z = cse.Xs('O2', VTi=['X3S-1.dat'], VTf=['B3S-1.dat', 'E3S-1.dat'],
              coupf=[4000], dipolemoment=[1, 0],
              transition_energy=np.arange(110, 174, 0.1), eni=800)
   
   plt.plot(Z.wavenumber, Z.xs)
   plt.show()

.. image:: 

   `file://examples/Rydberg-valence.png`


Documentation
-------------

PyCSE documentation is available at `readthedocs <http://pycse.readthedocs.org>`_.


Historical
----------

PyCSE is a Python implementation of the Johnson renormalized Numerov method. 
It provides a simple introduction to the profound effects of channel-coupling
in the calculation of diatomic photodissociation spectra.

More sophisticated C and Fortran implementations have been in use for a number 
of years, see references below. These were developed by Stephen Gibson (ANU),
Brenton Lewis (ANU), and Alan Heays (ANU and Leiden). 


Reference
---------

[1] `B.R. Johnson "The renormalized Numerov method applied to calculating the bound states of the coupled-channel Schroedinger equation" J. Chem. Phys. **69**, 4678 (1978) <http://dx.doi.org/10.1063/1.436421>`_

[2] `B.R. Lewis, S.T. Gibson, F. T. Hawes, and L. W. Torop "A new model for
the Schumann-Runge bands of O2" Phys. Chem. Earth(C) **26** 519 (2001) <http://dx.doi.org/10.1016/S1464-1917(01)00040-X>`_

[3] `A. N. Heays "Photoabsorption and photodissociation in molecular nitrogen <https://digitalcollections.anu.edu.au/handle/1885/7360>`_
