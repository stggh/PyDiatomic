import numpy as np
import numpy.testing as npt

import cse

def test_Morse():
    r = np.arange(0, 10, 0.05)

    # simulates O2 X-state
    VM = cse.tools.analytical.Morse(r, 1.21, 5.21, 0.0, 2.65)

    morse = cse.Cse('O2', VT=[(r, VM)], en=800)

    npt.assert_almost_equal(morse.cm, 778.3672, decimal=4)

    morse.solve(en=2000)
    npt.assert_almost_equal(morse.vib, 1)

if __name__ == '__main__':
    test_Morse()
