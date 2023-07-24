import numpy as np
import cse
import os
import numpy.testing as npt

def test_O2X():
    path = os.path.dirname(__file__)
    X = cse.Cse('O2', dirpath=path+'/../../examples/O2/potentials',
                suffix='.dat', VT=['X3S-1'])

    # (Ω, S, Λ, Σ)
    npt.assert_equal(X.AM, [(1, 1, 0, 1, 1)])
    npt.assert_almost_equal(X.μ, 1.328e-26)

    X.solve(800)

    # X-state Tv, Bv, Dv for v=0
    npt.assert_almost_equal(X.cm, 787.40, decimal=2)
    npt.assert_almost_equal(X.Bv, 1.438, decimal=3)
    npt.assert_almost_equal(X.Dv, 4.84e-6, decimal=6)

def test_transition():
    path = os.path.dirname(__file__)
    X = cse.Cse('O2', dirpath=path+'/../../examples/O2/potentials',
                suffix='.dat', VT=['X3S-1'], en=800)
    B = cse.Cse('O2', dirpath=path+'/../../examples/O2/potentials',
                suffix='.dat', VT=['B3S-1'])

    BX = cse.Transition(B, X, dipolemoment=[1])
    BX.calculate_xs(transition_energy=[51968,])  # bound-bound

    npt.assert_almost_equal(BX.wavenumber, 51967.8, decimal=1)

def test_coupled_channel():
    path = os.path.dirname(__file__)
    X = cse.Cse('O2', dirpath=path+'/../../examples/O2/potentials',
                suffix='.dat', VT=['X3S-1'], en=800)
    BE = cse.Cse('O2', dirpath=path+'/../../examples/O2/potentials',
                 suffix='.dat', VT=['B3S-1', 'E3S-1'], coup=[4000])
    BEX = cse.Transition(BE, X, dipolemoment=[1, 0])

    # O₂ Longest-band
    BEX.calculate_xs(transition_energy=np.arange(80950, 81050, 1))

    npt.assert_almost_equal(BEX.wavenumber[BEX.xs[:, 0].argmax()], 81006,
                            decimal=0)

if __name__ == '__main__':
    test_O2X()
    test_transition()
    test_coupled_channel()
