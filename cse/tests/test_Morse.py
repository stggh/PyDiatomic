import numpy as np
import numpy.testing as npt
import scipy.constants as C
import cse


def test_Morse():
    r = np.arange(0, 10, 0.05)

    # simulates O2 X-state
    VM = cse.tools.analytical.Morse(r, 1.21, 5.21, 0.0, 2.65)

    morse = cse.Cse('O2', VT=[(r, VM)], en=800)

    npt.assert_almost_equal(morse.cm, 778.46, decimal=2)

    morse.solve(en=2000)
    npt.assert_almost_equal(morse.vib, 1)


def test_Morse_art():
    '''
    Mizus et al. J. Mol. Spectrosc. 10.1016/j.jms.2022.111621

    Artificial Morse oscillator 1.
    '''

    De = 40000  # cm⁻¹
    m0 = 1*C.m_u
    μ = m0/2  # amu
    Re = 2  # Å
    alpha = 1  # Å⁻¹
    we = 2323.5942  # cm⁻¹
    xe = we/4/De
    A = 68.8885

    r = np.arange(0.8, 8, 0.01)
    mwf = cse.tools.analytical.Morse_wavefunction(r, Re, 0, alpha, A)

    V = cse.tools.analytical.Morse(r, Re=Re, De=De, Te=0, beta=alpha)

    morse = cse.Cse(μ=μ, VT=[(r, V)])
    morse.solve(1153)

    npt.assert_almost_equal(morse.cm, 1152.87, decimal=2)
    npt.assert_almost_equal(morse.wavefunction[:, 0, 0], mwf, decimal=6)


if __name__ == '__main__':
    # test_Morse()
    test_Morse_art()
