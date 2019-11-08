import numpy as np
from scipy.optimize import least_squares


class Morse():
    def __init__(self, R, Re, Rref, De, Te, beta=[1.0], q=2, Cm={}):
        """ Le Roy Expanded-Morse-Oscillator (EMO),
            Morse-Long-Range (MLR), and
            spline-pointwise potential curves.

            JQSRT 186, 210-220 (2017) doi:10.1016/j.jqsrt.2016.03.036
        """

        self.R = R
        self.Re = Re
        self.Rref = Rref
        self.beta = beta
        self.De = De
        self.Te = Te
        self.q = q

        self.VEMO = self.EMO()

    def EMO(self):  # Eq. (3)
        return self.De*(1 - np.exp(-self.betaEMO(self.R)*\
                                   (self.R - self.Re)))**2 + self.Te

    def yref(self, R):  # Eq. (2)
        Rq = R**self.q
        Rrefq = self.Rref**self.q
        return (Rq - Rrefq)/(Rq + Rrefq)

    def betaEMO(self, R):  # EQ. (4)
        by = 0.0
        yrefq = self.yref(R)
        for i, b in enumerate(self.beta):
            by += b*yrefq**i
        return by

    def ULR(self, Rx):
        ulr = 0.0
        for m, Cm in self.Cm.index():
            ulr += Cm/Rx**m
        return ulr


class Morsefit(Morse):
    def __init__(self, R, V, Rref=None, De=None, Nbeta=3, q=3,
                 fitpar=[]):
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
        self.Nbeta = Nbeta
        self.beta = np.ones(Nbeta)

        # estimate betas from linear form Eq. (24)
        self.est_beta()

        super().__init__(R, self.Re, self.Rref, self.De, self.Te,
                         beta=self.beta, q=q)

        self.fit_EMO()
        self.VEMO = self.EMO()


    def est_beta(self):   # Eq. (24)
        """ estimate betas from log-linear expression Eq. (24).

        """
        def residual(beta):
            self.beta = beta
            left = self.betaEMO(self.R[inner])*(self.R[inner]-self.Re)
            right = self.betaEMO(self.R[outer])*(self.R[outer]-self.Re)
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
                    self.beta = pars[:self.Nbeta]
                else:
                    self.__dict__[p] = pars[-i]

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
                self.beta = self.fit.x[:self.Nbeta]
            else:
                self.__dict__[p] = self.fit.x[-i]

        self.VEMO = self.EMO()
