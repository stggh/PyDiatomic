PyDiatomic README
=================

Introduction
------------

PyDiatomic solves the time-independent coupled-channel Schroedinger equation
using the Johnson renormalized Numerov method [1]. This is very compact and stable algorithm.

The code is directed to the computation of photodissociation cross sections for diatomic molecules. The coupling of electronic states results in transition profile broadening, line-shape asymmetry, and intensity sharing. A coupled-channel calculation is the only correct method compute the photodissociation cross-section.



Installation
------------

PyDiatomic requires Python 3.6 (*), numpy, scipy and periodictable. If you don't already have Python, we recommend an "all in one" Python package such as the `Anaconda Python Distribution <https://www.continuum.io/downloads>`_, which is available for free.

Download the latest version from github ::

    git clone https://github.com/stggh/PyDiatomic.git

`cd`  to the PyDiatomic directory, and use ::

    pip install .

Or, if you wish to edit the PyDiatomic source code without re-installing each time ::

    pip install -e . 


periodictable ::

    pip install periodictable

execution speed: There are big gains `> x8` in using the intel math kernel librayr

   sudo apt install intel-mkl



(*) due to the use of infix matrix multiplication ``@``. To run with python < 3.5, replace ``A @ B`` with ``np.dot(A, B)`` in ``cse.py`` and ``expectation.py``.


Example of use
--------------

PyDiatomic has a wrapper classes :class:`cse.Cse()` and
:class:`cse.Transition()` 

:class:`cse.Cse()`  set ups the CSE problem 
(interaction matrix of potential energy curves, and couplings) and solves 
the coupled channel Schroedinger equation for an initial guess energy.

Input parameters may be specified in the class instance, or they will be 
requested if required.

.. code-block:: python

   import cse
   X = cse.Cse('O2')   # class instance
   # CSE: potential energy curves [X3S-1.dat]:   # requested parameter
   X.solve(800)    # solves TISE for energy ~ 800 cm-1
   # attributes
   #     AM                   limits               rot                  
   #     Bv                   molecule             set_coupling()       
   #     calc                 mu                   set_mu()             
   #     cm                   node_count()         solve()              
   #     diabatic2adiabatic() openchann            vib                  
   #     energy               pecfs                VT                   
   #     levels()             R                    wavefunction         
   X.cm
   # 787.3978354211097
   X.vib
   # 0
   X.calc
   # {0: (787.3981436364634, 1.4376793143458806)}   {vib: (eigenvalue, Bv}
   X  # class representation
   # Molecule: O2  mass: 1.32801e-26 kg
   # Electronic state: X3S-1.dat
   # eigenvalues (that have been evaluated for this state):
   # v    energy(cm-1)    Bv(cm-1)
   # 0       787.398      1.43768


.. code-block:: python

   import cse
   X = cse.Cse('O2', VT=['X3S-1.dat'])
   X.levels(vmax=5)  # evaluates energy levels for v=0, .., vmax
                     # attribute .calc
   X  # class representation
   # Molecule: O2  mass: 1.32801e-26 kg
   # Electronic state: X3S-1.dat
   # evaluated eigenvalues:
   # v    energy(cm-1)    Bv(cm-1)
   # 0       787.398      1.43768
   # 1      2337.360      1.42051
   # 2      3867.008      1.40407
   # 3      5375.938      1.38823
   # 4      6863.744      1.37288
   # 5      8335.901      1.35919
   # 7     11196.366      1.32867
   # 11     16131.082      1.22378
   # 15     21719.531      1.20443
   # 17     24119.541      1.17186
   # 24     31559.738      0.99627
   # 25     32754.587      1.03787
   # 35     40566.037      0.74300


:class:`cse.Transition()` evaluates two couple channel problems, for an
intitial and final set of coupled channels, to calculate the photodissociation 
cross section.

.. code-block:: python

   import numpy as np
   import cse
   # initial state
   O2X = cse.Cse('O2', VT=['potentials/X3S-1.dat'], en=800)
   # final state
   O2B = cse.Cse('O2', VT=['potentials/B3S-1.dat'])
   # transition 
   BX = cse.Transition(O2B, O2X)
   # methods 
   # BX.calculate_xs()  
   BX.calculate_xs(transition_energy=np.arange(110, 174, 0.1), eni=800)
   # attributes
   # the calculated cross section BX.xs and those of the initial BX.gs and
   # final coupled states BS.us

A simple :math:`^{3}\Sigma_{u}^{-} \leftrightarrow {}^{3}\Sigma^{-}_{u}` Rydberg-valence coupling in O\ :sub:`2`

.. code-block:: python

    import numpy as np
    import cse
    import matplotlib.pyplot as plt

    O2X = cse.Cse('O2', VT=['X3S-1.dat'], en=800)
    O2B = cse.Cse('O2', VT=['B3S-1.dat', 'E3S-1.dat'], coup=[4000])
    O2BX = cse.Transition(B, X, dipolemoment=[1, 0],
               transition_energy=np.arange(110, 174, 0.1))

    plt.plot(O2BX.wavenumber, O2BX.xs*1.0e16)
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
i7-9700            4.6      8          3
Xeon E5-2697       2.6      64         6
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


Applications
------------
The following publications have made use of `PyDiatomic`:

[1] `Z. Xu, N. Luo, S. R. Federman, W. M. Jackson, C-Y. Ng, L-P. Wang, and K. N. Crabtree "Ab Initio Study of Ground-state CS Photodissociation via Highly Excited Electronic States" Astrophy. J. 86, 882 (2019) <http://doi.org/10.3847/1538-4357/ab35ea>`_

[2] `Z. Xu, S. R. Federman, W. M. Jackson, C-Y. Ng, L-P. Wang, and K. N. Crabtree "Multireference configuration interaction study of the predissociation of C₂ via its F¹Πu state" J. Chem. Phys. (2022) <http://doi.org/10.1063/5.0097451>`_


References
----------

[1] `B.R. Johnson "The renormalized Numerov method applied to calculating the bound states of the coupled-channel Schroedinger equation" J. Chem. Phys. 69, 4678 (1978) <http://dx.doi.org/10.1063/1.436421>`_

[2] `B.R. Lewis, S.T. Gibson, F. T. Hawes, and L. W. Torop "A new model for
the Schumann-Runge bands of O₂" Phys. Chem. Earth(C) 26 519 (2001) <http://dx.doi.org/10.1016/S1464-1917(01)00040-X>`_

[3] `B.R. Lewis, S.T. Gibson, and P.M. Dooley "Fine-structure dependence of predissociation linewidth in the Schumann-Runge bands of molecular oxygen"
" J. Chem. Phys. 100 7012 (1994) <https://doi.org/10.1063/1.466902>`_

[4] `A. N. Heays "Photoabsorption and photodissociation in molecular nitrogen, PhD Thesis (2011) <https://digitalcollections.anu.edu.au/handle/1885/7360>`_


Citation
--------
If you find PyDiatomic useful in your work please consider citing this project.


.. image:: https://zenodo.org/badge/23090/stggh/PyDiatomic.svg
   :target: https://zenodo.org/badge/latestdoi/23090/stggh/PyDiatomic
