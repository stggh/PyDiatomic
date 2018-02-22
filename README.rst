PyDiatomic README
=================

.. image:: https://travis-ci.org/stggh/PyDiatomic.svg?branch=master
    :target: https://travis-ci.org/stggh/PyDiatomic


Introduction
------------

PyDiatomic solves the time-independent coupled-channel Schroedinger equation
using the Johnson renormalized Numerov method [1]. This is very compact and stable algorithm.

The code is directed to the computation of photodissociation cross sections for diatomic molecules. The coupling of electronic states results in transition profile broadening, line-shape asymmetry, and intensity sharing. A coupled-channel calculation is the only correct method compute the photodissociation cross-section.



Installation
------------

PyDiatomic requires Python 3.5 (*), numpy and scipy. If you don't already have Python, we recommend an "all in one" Python package such as the `Anaconda Python Distribution <https://www.continuum.io/downloads>`_, which is available for free.

Download the latest version from github ::

    git clone https://github.com/stggh/PyDiatomic.git

`cd`  to the PyDiatomic directory, and use ::

    python3 setup.py install --user

Or, if you wish to edit the PyAbel source code without re-installing each time ::

    python3 setup.py develop --user



(*) due to the use of infix matrix multiplication ``@``. To run with python < 3.5, replace ``A @ B`` with ``np.dot(A, B)`` in ``cse.py`` and ``expectation.py``.


Example of use
--------------

PyDiatomic has a wrapper classes :class:`cse.Cse()` and  :class:`cse.Xs()` 

:class:`cse.Cse()`  set ups the CSE problem 
(interaction matrix of potential energy curves, and couplings) and solves 
the coupled channel Schroedinger equation for an initial guess energy.

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

.. code-block:: python

   import cse
   X = cse.Cse('O2', VT=['X3S-1.dat'])
   X.levels(vmax=5)  # evaluates energy levels for v=0, .., vmax
                     # attributes .energies, .Bvs
   X  # class representation
   # Molecule: O2  mass: 1.32801e-26 kg
   # Electronic state: X3S-1.dat
   # Eigenvalues:  v    energy(cm-1)    Bv(cm-1)
   #               0       787.398      1.43768
   #               1      2343.763      1.42186
   #               2      3876.577      1.40612
   #               3      5386.169      1.39043
   #               4      6872.504      1.37478
   #               5      8335.767      1.35921

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

A simple :math:`^{3}\Sigma_{u}^{-} \leftrightarrow {}^{3}\Sigma^{-}_{u}` Rydberg-valence coupling in O\ :sub:`2`

.. code-block:: python

    import numpy as np
    import cse
    import matplotlib.pyplot as plt

    Z = cse.Xs('O2', VTi=['X3S-1.dat'], VTf=['B3S-1.dat', 'E3S-1.dat'],
               coupf=[4000], dipolemoment=[1, 0],
               transition_energy=np.arange(110, 174, 0.1), eni=800)

    plt.plot(Z.wavenumber, Z.xs*1.0e16)
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Cross section ($10^{-16}$ cm$^{2}$)")
    plt.axis(ymin=-0.2)
    plt.title("O$_{2}$ $^{3}\Sigma_{u}^{-}$ Rydberg-valence interaction")
    plt.savefig("RVxs.png", dpi=75)
    plt.show()


.. figure:: https://cloud.githubusercontent.com/assets/10932229/21469172/177a519c-ca91-11e6-8251-52efb7aa1a37.png
   :width: 300px
   :alt: calculated cross section
   

`example_O2xs.py`:

.. figure:: https://user-images.githubusercontent.com/10932229/33101884-53a8ab68-cf6e-11e7-86f2-876d28809328.png
   :width: 300px
   :alt: example_O2xs


`example_O2_continuity.py`:

.. figure:: https://user-images.githubusercontent.com/10932229/30096079-b869e486-9319-11e7-8adb-3ae64bff88d4.png
   :width: 300px
   :alt: example_O2_continuity


`example_O2X_fine_structure.py`:

.. code-block:: python

    PyDiatomic O2 X-state fine-structure levels
      energy diffences (cm-1): Rouille - PyDiatomic
     N        F1          F2          F3
     1      -0.000       0.000       0.000
     3      -0.005       0.000       0.009
     5      -0.009       0.000       0.013
     7      -0.013       0.000       0.017
     9      -0.017       0.000       0.022
    11      -0.021       0.000       0.026
    13      -0.025       0.000       0.030
    15      -0.029      -0.000       0.034
    17      -0.033      -0.000       0.039
    19      -0.037      -0.000       0.043
    21      -0.041      -0.000       0.047



`example_O2_SRB4.py`:

.. figure:: https://user-images.githubusercontent.com/10932229/33054465-7094c0f0-cecd-11e7-99c1-4f14c4ffad48.png
   :width: 300px
   :alt: example_O2_SRB4


`example_HO.py`:

.. figure:: https://user-images.githubusercontent.com/10932229/30100890-b3195eee-932d-11e7-9480-fec2af23f6ff.png
   :width: 300px
   :alt: example_HO


`example_rkr.py`:

.. figure:: https://cloud.githubusercontent.com/assets/10932229/21469152/a33fd798-ca90-11e6-8fe3-1f3c3364de26.png
   :width: 300px
   :alt: example_rkr


Rotation
~~~~~~~~

.. code-block:: python

    import cse
    
    X = cse.Cse('O2', VT=['X3S-1.dat'])  # include path to potential curve
    X.solve(900, rot=0)
    X.cm
    # 787.3978354211097
    X.Bv
    # 1.4376793638070153
    X.solve(900, 20)
    X.cm
    # 1390.369249612629
    # (1390.369-787.398)/(20*21) = 1.4356


Timing
------

Each transition energy solution to the coupled-channel Schroedinger
equation is a separate calculation.  PyDiatomic uses :code:`multiprocessing`
to perform these calculations in parallel, resulting in a substantial
reduction in execution time on multiprocessor systems. e.g. for :code:`example_O2xs.py`:


==============     ====     ======     ==========
machine            GHz      CPU(s)     time (sec)
==============     ====     ======     ==========
Xenon E5-2697      2.6      64         6
i7-6700            3.4      8          17
Macbook pro i5     2.4      4          63
raspberry pi 3     1.35     4          127
==============     ====     ======     ==========


Documentation
-------------

PyDiatomic documentation is available at `readthedocs <http://pydiatomic.readthedocs.io/en/latest/>`_.


Historical
----------

PyDiatomic is a Python implementation of the Johnson renormalized Numerov method. 
It provides a simple introduction to the profound effects of channel-coupling
in the calculation of diatomic photodissociation spectra.

More sophisticated C and Fortran implementations have been in use for a number 
of years, see references below. These were developed by Stephen Gibson (ANU),
Brenton Lewis (ANU), and Alan Heays (ANU, Leiden, and ASU). 


References
----------

[1] `B.R. Johnson "The renormalized Numerov method applied to calculating the bound states of the coupled-channel Schroedinger equation" J. Chem. Phys. 69, 4678 (1978) <http://dx.doi.org/10.1063/1.436421>`_

[2] `B.R. Lewis, S.T. Gibson, F. T. Hawes, and L. W. Torop "A new model for
the Schumann-Runge bands of O2" Phys. Chem. Earth(C) 26 519 (2001) <http://dx.doi.org/10.1016/S1464-1917(01)00040-X>`_

[3] `B.R. Lewis, S.T. Gibson, and P.M. Dooley "Fine-structure dependence of predissociation linewidth in the Schumann-Runge bands of molecular oxygen"
" J. Chem. Phys. 100 7012 (1994) <https://doi.org/10.1063/1.466902>`_

[4] `A. N. Heays "Photoabsorption and photodissociation in molecular nitrogen, PhD Thesis (2011) <https://digitalcollections.anu.edu.au/handle/1885/7360>`_


Citation
--------
If you find PyDiatomic useful in your work please consider citing this project.


.. image:: https://zenodo.org/badge/23090/stggh/PyDiatomic.svg
   :target: https://zenodo.org/badge/latestdoi/23090/stggh/PyDiatomic
