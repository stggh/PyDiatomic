import numpy as np
import cse
import numpy.testing as npt
from scipy.interpolate import splrep, splev

def CSEmodel(iso='CO'):
    R = np.arange(0.9, 3, 0.01)
    VX = cse.tools.analytical.Wei(R, Re=1.131, De=9.077e+04, Voo=90674, b=2.33,
                                  h=-0.01314)
    VB = cse.tools.analytical.Wei(R, Re=1.114, De=6.845e+04, Voo=155262.65,
                                  b=3.1, h=-0.12)
    X = cse.Cse(iso, VT=[(R, VX)])
    B = cse.Cse(iso, VT=[(R, VB)])
    BX = cse.Transition(B, X, dipolemoment=[1])

    BX.us.levels(5)

    return BX

def test_ΔV():
    BX = CSEmodel()
    B = BX.us

    offset = 20
    v = np.arange(5)
    Tv = np.array([x for (x, *_) in B.results.values()])[v] + offset

    initval = 15
    lbl = B.statelabel[0] 

    fit = cse.tools.model_fit.Model_fit(BX, method='lm',
                    data2fit={lbl:{'position':(v, Tv)}},  # fit data
                    VT_adj={lbl:{'ΔV': initval}},         # PEC shift
                    verbose=False)

    scaling = fit.result.x[0]
    npt.assert_almost_equal(offset, scaling*initval, decimal=1)

def test_Rstr(iso='CO'):
    BX = CSEmodel()
    X = BX.gs
    B = BX.us
    # B - unperturbed energy levels
    v = np.arange(5)
    Tv = np.array([x for (x, *_) in B.results.values()])[v]

    # perturbation - widen the PEC around Re ----------------------
    BV = B.VT[0, 0]
    BR = B.R.copy()
    Re = BR[BV.argmin()]

    inner_perturb = 1.15
    outer_perturb = 1.1
    BR[BR < Re] = (BR[BR < Re] - Re)*inner_perturb + Re
    BR[BR > Re] = (BR[BR > Re] - Re)*outer_perturb + Re
    spl = splrep(BR, BV)
    Bpert = splev(B.R, spl)  # a widened PEC

    # new excited state instance, with widened PEC
    Bp = cse.Cse(iso, VT=[(B.R, Bpert)])
    Bp.levels(5)

    lbl = B.statelabel[0]
    Bp.statelabel[0] = lbl
    # transition instance of perturbed PEC
    BpX = cse.Transition(Bp, X, dipolemoment=[1])

    # model_fit instance ------------------------------------------------
    fit = cse.tools.model_fit.Model_fit(BpX, method='trf', bounds_factor=1,
                          data2fit={lbl:{'position':(v, Tv)}},
                          VT_adj={lbl:{'Rstr': {'inner':1, 'outer':1}}},
                          verbose=False)

    Bf = fit.csemodel.us
    npt.assert_almost_equal(1/inner_perturb, inner_perturb*fit.result.x[0],
                            decimal=1)
    npt.assert_almost_equal(1/outer_perturb, outer_perturb*fit.result.x[1],
                            decimal=1)


if __name__ == '__main__':
    BX = CSEmodel()
    test_ΔV()
    test_Rstr()
