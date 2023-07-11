import numpy as np
import cse
import numpy.testing as npt

def test_ΔV():
    iso = 'CO'

    # CSE instances, ground and excited electronic states
    X = cse.Cse(iso, VT=['../../examples/CO/potentials/X1S0.dat'])
    B = cse.Cse(iso, VT=['../../examples/CO/potentials/B1S0.dat'])

    # transition instance B <- X
    BX = cse.Transition(B, X, dipolemoment=[1])

    B.levels(5)  # energy levels

    # levels, to fit, energy offset
    offset = 20
    v = np.arange(5)
    Gv = []
    for vv, (Tv, *_) in B.results.items():
        if vv in v:
            Gv.append(Tv + offset)  # off-set 'experimental levels'


    initval = 15  # starting offset
    lbl = B.statelabel[0] 

    fit = cse.tools.model_fit.Model_fit(BX, method='lm',
                    data2fit={lbl:{'position':(v, Gv)}},  # fit data
                    VT_adj={lbl:{'ΔV': initval}},         # PEC shift
                    verbose=False)

    scaling = fit.result.x[0]
    npt.assert_almost_equal(offset, scaling*initval, decimal=1)

if __name__ == '__main__':
    test_ΔV()
