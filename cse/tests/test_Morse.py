import numpy as np
import numpy.testing as npt

import cse

r = np.arange(0, 10, 0.05)

# simulates O2 X-state
VM = cse.analytical.Morse(r, 1.21, 5.21, 0.0, 2.65)

morse = cse.Cse('O2', VT=[(r, VM)], en=800)

npt.assert_almost_equal(morse.cm, 778.47775, decimal=5)

morse.solve(en=2000)
npt.assert_almost_equal(morse.vib, 1)
