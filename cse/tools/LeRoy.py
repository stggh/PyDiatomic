import numpy as np
from scipy.optimize import least_squares


class Morse():
    def __init__(self, R, Re, Rref, De, Te, beta=[1.0], p=1, q=2, Cm={}):
        """ Le Roy Expanded-Morse-Oscillator (EMO),
            Morse-Long-Range (MLR), and
            spline-pointwise potential curves.

            JQSRT 186, 210-220 (2017) doi:10.1016/j.jqsrt.2016.03.036

            For ordinary Morse set beta to a single value.
        """

        self.R = R
        self.Re = Re
        self.Rref = Rref
        self.beta = beta
        self.De = De
        self.Te = Te
        self.p = p
        self.q = q
        self.Cm = Cm

        self.VEMO = self.EMO()

    def EMO(self):  # Expanded Morse Oscillator, Eq. (3)
        return self.De*(1 - np.exp(-self.betaEMO(self.R, self.q)*\
                       (self.R - self.Re)))**2 + self.Te

    def MLR(self, R):  # Morse long-range, Eq. (6)
        ULRratio = self.ULR(R)/self.ULR(self.Re)
        exponent = self.beta(R, self.q) * self.yref(self.R, self.p)

        return self.De*(1 - ULRratio * np.exp(-exponent))**2

    def betaEMO(self, R, pq):  # EQ. (4)
        by = 0.0
        yrefpq = self.yref(R, pq)
        for i, b in enumerate(self.beta):
            by += b*yrefpq**i
        return by

    def betaMLR(self, R):  # Eq. (10)
        beta_infy = np.log(2*self.De/self.U_LR(self.Re))
        yrefp = self.yref(self.R, self.p)
        return yrefp*beta_infy + (1 - yrefp)*betaEMO(self.R, self.q)

    def yref(self, R, pq):  # Eq. (2)
        Rpq = R**pq
        Rrefpq = self.Rref**pq
        return (Rpq - Rrefpq)/(Rpq + Rrefpq)

    def ULR(self, Rx):
        ulr = 0.0
        for m, Cm in self.Cm.items():
            ulr += Cm/Rx**m
        return ulr


class Morsefit(Morse):
    def __init__(self, R, V, beta=[1.], Rref=None, De=None, q=3,
                 fitpar=[], Cm={}):
        """ fit EMO to supplied potential curve.

        """

        self.fitpar = fitpar

        # supplied potential energy curve
        self.R = R
        self.V = V
        if self.V[-1] < 1000:
            self.V *= 8065.541   # convert eV to cm-1 
        
        # set some easy to determine constants
        self.Re = R[V.argmin()]
        if Rref is None:
            self.Rref = self.Re
        else:
            self.Rref = Rref

        self.Te = V.min()
        if De is None:
            self.De = V[-1] - self.Te
        else:
            self.De = De

        self.q = q
        self.Cm = Cm

        # estimate betas from linear form Eq. (24)
        # self.est_beta()
        self.beta = beta

        super().__init__(R, self.Re, self.Rref, self.De, self.Te,
                         beta=self.beta, q=q)

        # self.fit_EMO()
        # self.VEMO = self.EMO()


    def est_beta(self):   # Eq. (24)
        """ estimate betas from log-linear expression Eq. (24).

        """
        def residual(beta):
            self.beta = beta
            left = self.betaEMO(self.R[inner], self.q)*(self.R[inner]-self.Re)
            right = self.betaEMO(self.R[outer], self.q)*(self.R[outer]-self.Re)
            return np.concatenate((left-inn, right-out))

        inner = self.R < self.Re
        outer = self.R >= self.Re
        outer[-1] = False  # in case value is Nan
        
        inn = -np.log(1.0 + np.sqrt((self.V[inner] - self.Te)/self.De))
        out = -np.log(1.0 - np.sqrt((self.V[outer] - self.Te)/self.De))

        self.est = least_squares(residual, self.beta)

        self.beta = self.est.x


    def fit_EMO(self):
        """ full least-squares fit to EMO.

        """
        def residual(pars):
            for i, p in enumerate(self.fitpar):
                if p == 'beta':
                    self.beta = pars[:len(self.beta)]
                else:
                    self.__dict__[p] = pars[len(self.beta)+i-1]

            return self.EMO() - self.V

        pars = []
        for p in self.fitpar:
            if p == 'beta':
                pars = self.beta
            else:
                pars = np.append(pars, self.__dict__[p])

        self.fit = least_squares(residual, pars)

        for i, p in enumerate(self.fitpar):
            if p == 'beta':
                self.beta = self.fit.x[:len(self.beta)]
            else:
                self.__dict__[p] = self.fit.x[len(self.beta)+i-1]

        self.VEMO = self.EMO()


    def fitCm(self):
        def residual(pars, Rx, Vx):
            for i, (m, Cm) in enumerate(self.Cm.items()):
                self.Cm[m] = pars[i]
            return self.ULR(Rx) - Vx

        # long range part of potential curve
        LR = self.R > 3
        Rx = self.R[LR]
        Vx = self.V[LR]

        pars = np.ones(len(self.Cm.values()))
        
        result = least_squares(residual, pars, args=(Rx, Vx)) 
        self.fitCm = result

        for i, (m, Cm) in enumerate(self.Cm.items()):
            self.C[m] = result.x[i]

