PyDiatomic README
=================

Introduction
------------

PyDiatomic solves the time-independent coupled-channel Schrödinger equation
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

execution speed: There are big gains `> x8` in using the intel math kernel library

   sudo apt install intel-mkl



(*) due to the use of infix matrix multiplication ``@``. To run with python < 3.5, replace ``A @ B`` with ``np.dot(A, B)`` in ``cse.py`` and ``expectation.py``.


Example of use
--------------

PyDiatomic has a wrapper classes :class:`cse.Cse()` and
:class:`cse.Transition()` 

:class:`cse.Cse()`  set ups the CSE problem 
(interaction matrix of potential energy curves, and couplings) and solves 
the coupled channel Schrödinger equation for an initial guess energy.

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
   # Molecule: O2  mass: 1.32801e-26 kg, 7.99746 amu
   # Electronic state:   X³Σ₁⁻
   # eigenvalues (that have been evaluated for this state):
   # v  rot   energy(cm⁻¹)    Bv(cm⁻¹)     Dv(cm⁻¹)
   # 0   0      787.399       1.43768      4.840e-06
   # 1   0     2343.762       1.42186      4.839e-06
   # 2   0     3876.579       1.40613      4.837e-06
   # 3   0     5386.161       1.39043      4.839e-06
   # 4   0     6872.504       1.37479      4.840e-06
   # 5   0     8335.794       1.35923      4.842e-06
   # 6   0     9776.167       1.34368      4.846e-06


:class:`cse.Transition()` evaluates two couple channel problems, for an
intitial and final set of coupled channels, to calculate the photodissociation 
cross section.

.. code-block:: python

   import numpy as np
   import cse
   # initial state instance
   O2X = cse.Cse('O2', VT=['potentials/X3S-1.dat'], en=800)
   # final state instance
   O2B = cse.Cse('O2', VT=['potentials/B3S-1.dat'])
   # transition instance O2B <- O2X, electric dipole transition 
   BX = cse.Transition(O2B, O2X, dipolemoment=[1])

   # methods
   # evaluate cross section 57,550-90,000 cm⁻¹ step 500 cm⁻¹
   BX.calculate_xs(transition_energy=np.arange(57550, 90000, 500))

   # attributes
   #   BX.wavenumber, BX.xs - the calculated cross section
   #   BX.us - upper state instance, BX.gs - ground state instance


A simple :math:`^{3}\Sigma_{u}^{-} \leftrightarrow {}^{3}\Sigma^{-}_{u}` Rydberg-valence coupling in O\ :sub:`2`, `examples/O2/O2_RVxs.py`:

.. code-block:: python

    import numpy as np
    import cse
    import matplotlib.pyplot as plt

    O2X = cse.Cse('O2', VT=['potentials/X3S-1.dat'], en=800)
    O2B = cse.Cse('O2', dirpath='potentials', VT=['B3S-1.dat', 'E3S-1.dat'],
                  coup=[4000])
    O2BX = cse.Transition(O2B, O2X, dipolemoment=[1, 0],
               transition_energy=np.arange(57550, 90000, 100))  # cm⁻¹

    plt.plot(O2BX.wavenumber, O2BX.xs[:, 0])  # '0' is 'B3S-1.dat' channel
    plt.xlabel('Wavenumber (cm$^{-1}$)')
    plt.ylabel('Cross section (cm$^{2}$)')
    plt.ticklabel_format(axis='y', style='sci', scilimits=(-19, -19))
    plt.title('O$_{2}$ $^{3}\Sigma_{u}^{-}$ Rydberg-valence interaction')

    plt.savefig('figures/O2_RVxs.svg')
    plt.show()


.. figure:: https://github.com/stggh/PyDiatomic/assets/10932229/5e466dee-3017-4414-bd01-6aeef4edab17
   :width: 500px
   :alt: calculated cross section
   :align: center
   

`examples/O2_xs.py`:

.. figure:: https://github.com/stggh/PyDiatomic/assets/10932229/e89cdfdc-7747-425a-bcf1-63c7e30376e1
   :width: 500px
   :alt: O2_xs
   :align: center


`examples/O2_continuity.py`:

.. figure:: https://github.com/stggh/PyDiatomic/assets/10932229/87375946-ddc3-41aa-b715-3e50eed8ab2c
   :width: 500px
   :alt: O2_continuity
   :align: center


`examples/O2_fine_structure_X.py`:

.. code-block:: python

    PyDiatomic O₂ X-state fine-structure levels
      energy diffences (cm⁻¹): Rouille - PyDiatomic
     N        F₁          F₂          F₃
     1      -0.001       0.000      -0.591
     3      -0.005       0.000       0.009
     5      -0.009       0.000       0.013
     7      -0.013       0.000       0.017
     9      -0.016       0.000       0.022
    11      -0.020       0.001       0.026
    13      -0.024       0.001       0.031
    15      -0.027       0.001       0.036
    17      -0.031       0.002       0.041
    19      -0.034       0.002       0.046
    21      -0.037       0.003       0.051
    23      -0.040       0.003       0.056
    25      -0.043       0.004       0.062
    27      -0.046       0.005       0.067
    29      -0.049       0.007       0.073
    31      -0.051       0.008       0.080
    33      -0.053       0.010       0.087
    35      -0.054       0.013       0.094
    37      -0.055       0.016       0.103
    39      -0.055       0.019       0.112
    41      -0.054       0.024       0.122
    43      -0.052       0.030       0.133
    45      -0.049       0.037       0.145
    47      -0.044       0.045       0.160
    49      -0.038       0.056       0.176


`examples/O2_SRB_analyse_xs.py` (`dirpath = 'AX16O2_12'`):

.. figure:: https://github.com/stggh/PyDiatomic/assets/10932229/65175c10-8097-4597-9418-fb1e31edb0f2
   :width: 800px
   :alt: O2_SRB_analyse_xs
   :align: center


`examples/general/harmonic_oscillator.py`:

.. figure:: https://github.com/stggh/PyDiatomic/assets/10932229/cb4b30d7-3fa2-4ff7-8671-a438c7e592c1
   :width: 500px
   :alt: harmonic_oscillator
   :align: center


`examples/O2_RKR_Xstate.py`:

.. figure:: https://github.com/stggh/PyDiatomic/assets/10932229/cf8a9f53-c923-4c11-af6f-aac9dad434af 
   :width: 500px
   :alt: O2_RKR_Xstate
   :align: center


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

Each transition energy solution to the coupled-channel Schrödinger
equation is a separate calculation.  PyDiatomic uses :code:`multiprocessing`
to perform these calculations in parallel, resulting in a substantial
reduction in execution time on multiprocessor systems. e.g. for :code:`examples/O2_continuity.py`, Anaconda python 3.11.4, Linux OS:


==============     ====     ======     ==============  =========
Machine            GHz      CPU(s)     Time(sec) osc   continuum
==============     ====     ======     ==============  =========
i7-9700            4.7      8          0.6             4
Orange Pi 5        1.8      8          2.1             9
i7-6700            3.4      8          1               10
Macbook Pro i5     2.4      4          2.5             24
Raspberry Pi 4     1.8      4          8.0             159
==============     ====     ======     ==============  =========


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

[1] `B.R. Johnson "The renormalized Numerov method applied to calculating the bound states of the coupled-channel Schrödinger equation" J. Chem. Phys. 69, 4678 (1978) <http://dx.doi.org/10.1063/1.436421>`_

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
