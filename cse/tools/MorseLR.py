import numpy as np


class MorseLR():
    def __init__(self, R, Re=4.99, De=287.0, Cm=1.119e7, Cn=2.63e8,
                 m=6, n=8, p=3,
                 phi=[-2.12675, 0.0554, 20.026, -144.25, 371.6, -422.0, 180.0],
                 Ns=1, Nl=3):
        """ Le Roy Morse-long-range potential curve.

            J Chem Phys 126, 194313 (2007)   doi:10.1063/1.2734973
        """

        self.R = R
        self.Re = Re
        self.De = De
        self.Cm = Cm
        self.Cn = Cn
        self.m = m
        self.n = n
        self.p = p
        self.phi = phi
        self.Ns = Ns
        self.Nl = Nl

        self.MLR = self.Morse_long_range(R)

    def Morse_long_range(self, R):
        ULRratio = self.U_LR(R)/self.U_LR(self.Re)
        exponent = self.phiLR(R) * self.y(R)

        return self.De*(1 - ULRratio * np.exp(-exponent))**2  # Eq. (3)

    def U_LR(self, R):
        return self.Cn/R**self.n + self.Cm/R**self.m  # Eq. (4)

    def y(self, R):
        Rp = R**self.p
        Rep = self.Re**self.p

        return (Rp - Rep)/(Rp + Rep)  # Eq. (5)

    def phiLR(self, R):
        # Eq. (8) + (9)
        self.phi_inf = np.log(2*self.De/self.U_LR(self.Re))
        ypR = self.y(R)

        z = np.array([])
        for sub, N in zip([R <= self.Re, R > self.Re], [self.Ns, self.Nl]):
            ypRs = ypR[sub]

            x = 0.0
            for i in range(N):
                x += self.phi[i]*(ypRs**i)

            x *= (1 - ypRs)
            x += ypRs*self.phi_inf

            z = np.append(z, x)

        return z


# main ----------------
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    R = np.arange(4, 10, 0.1)
    V = MorseLR(R)
    plt.plot(V.R, V.MLR)
    plt.show()
