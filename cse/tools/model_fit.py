import numpy as np
import cse
import re
from cse.tools.analytical import Julienne, Wei, fiterrors
from scipy.interpolate import splrep, splev
from scipy.optimize import least_squares
from scipy.linalg import svd


class Model_fit():
    """ Least-squares fitting CSE model parameters to any experimental
        data.

        CSE model parameter dicts:
          VT_adj : electronic potential energy curves,
          coup_adj : couplings
          etdm_adj : electric transition dipole moment
           
        data2fit: data dict to fit:
        'xs' : {channel statelabel, ([wavenumber], [cross section])}
        'Bv' : {statelabel, ([v], [Bv])}
        'position' : {statelabel, ([v], [wavenumber of peak])}
        'width' : {statelabel, (wn0, wn1, width, J)}

    Attributes
    ----------
    result : dict
        least_squares fit attriubutes, with stderr for each parameter
    """

    def __init__(self, csemodel, data2fit, VT_adj={}, coup_adj={},
                 etdm_adj={}, method='lsq', verbose=True):
        """
        Parameters
        ----------
        data2fit: {'channel':{'position': wavenumber_of_peak,
                              'Bv: [v, Bv] array,
                              'xs': [wavenumber, cross_section] array},
                  {'channel2':   }}

        # potential energy curves -------
        VT_adj: {PEC:
            {'ΔR':float},  # radial shift
            {'ΔV':float},  # energy shift

            # parameters for Wei analytical PEC, applied between Rm...Rn
            {'Wei':{'Re':Re, 'De':De, 'Voo':Voo, 'b':b, 'h':h, 'Rm':Rm,
                    'Rn':Rn}},

            # Julienne (open channel), applied Rm...Rn
            {'Julienne':{'Mx':Mx, 'Rx':Rx, 'Vx':Vx, 'Voo':Voo, 'Rm':Rm,
                         'Rn':Rn}},

            # stretching
            {'Rstr': {'left':left, 'right':right}},  # about Rₑ
            {'Vstr': float},  # about V∞

            # scaling
            {'spline':[R₀, R₁, ..., R₋₁]},  # radial positions of knots
            PEC is scaled by spline between R₀..R₋₁
            )

        # coupling scale factor -------
        coup_adj: {'state0<->state1':scaling, 'state0<->state2':scaling}  # etc.

        # electric dipole transition moment -------
        etdm_adj: {'fsl0<-isl0':scaling_factor, 'fsl1<-isl0':scaling_factor} 

        Each parameter is a scaled relative to 1.0
        """

        self.verbose = verbose
        self.method = method

        # CSE model -------------------------------------------
        self.csemodel = csemodel  # CSE Transition instance

        # handy variables
        self.R = csemodel.us.R  # internuclear distance
        self._evcm = csemodel.us._evcm  # conversion eV to cm⁻¹

        # CSE model parameters to adjust ---------------------
        self.VT_adj = VT_adj  # PEC adjustments
        self.coup_adj = coup_adj  # couplings
        self.etdm_adj = etdm_adj  # electronic transition dipole moment

        # experimental data to fit ---------------------------
        self.data2fit = data2fit  # data sets

        # least-squares fitting ------------------------------
        self.fit()  # least-squares fit
        self.result.stderr = fiterrors(self.result)  # parameter error estimates
        self.print_result()

    def parameter_pack(self):
        """ parameter list for least-squares fitting.

        """
        self.par_count = 0

        # interaction matrix - keep original values
        self.VT_orig = self.csemodel.us.VT.copy()
        # diagonal elements = PECs in units cm⁻¹
        self.VTd_orig = np.diagonal(self.VT_orig).T.copy()*self._evcm

        # PEC parameters ------------------------------------------------
        for statelabel, par_dict in self.VT_adj.items():
            for v in par_dict.values():
                if isinstance(v, (int, float)):
                    self.par_count += 1
                else:
                    self.par_count += len(v)

            if 'Wei' in par_dict or 'Julienne' in par_dict:
                self.par_count -= 2  # 'Rm' and 'Rn' limits not param

        # Coupling ---------------------------------------------
        self.coupling = {}
        for lbl, scaling in self.coup_adj.items():
            state1, state2 = re.split('<->', lbl)
            i = self.csemodel.us.statelabel.index(state1)
            j = self.csemodel.us.statelabel.index(state2)
            # keep original value as tuple
            self.coupling[lbl] = (self.VT_orig[i][j].copy(), i, j)
            self.par_count += 1

        self.etdm = {}
        for lbl, scaling in self.etdm_adj.items():
            lblf, lbli = re.split('<-', lbl)
            indxf = self.csemodel.us.statelabel.index(lblf)
            indxi = self.csemodel.gs.statelabel.index(lbli)
            # keep original transition moment as tuple
            self.etdm[lbl] = \
              (self.csemodel.dipolemoment[:, indxi, indxf].copy(), indxi, indxf)
            self.par_count += 1

        pars = np.ones(self.par_count)
        self.lsqpars = pars
        self.bounds = (pars*0.95, pars*1.05)

    def parameter_unpack(self, pars):
        """ extract new CSE model parameters from values of the least_squares
            parameter list.

        """
        lsqpars = list(pars)

        R = self.R
        VTd = self.VTd_orig.copy()  # original PECs, diagonal interaction array

        # potential energy curve adjustments --------------------
        for state, state_dict in self.VT_adj.items():
            indx = self.csemodel.us.statelabel.index(state)
            for param, value in state_dict.items():
                match param:
                    # single value
                    case 'ΔR':
                        spl = splrep(R-value*lsqpars.pop(0), VTd[indx])
                        VTd[indx] = splev(R, spl)

                    case 'ΔV':
                        scaling = lsqpars.pop(0)
                        VTd[indx] += value*scaling
                        print(f'ΔV: {scaling:.3f}')

                    case 'Vstr':
                        scaling = lsqpars.pop(0)
                        Voo = VTd[indx][-1]
                        VTd[indx] = (VTd[indx] - Voo)*scaling + Voo

                    # multi-value
                    case 'Rstr':
                        left, right = lsqpars[:2]
                        Rscaled = R.copy()
                        Re = R[VTd[indx].argmin()]
                        print(f'left={left:.3f}, right={right:.3f}')

                        rl = R < Re
                        Rscaled[rl] = (R[rl] - Re)*left + Re
                        rr = R >= Re
                        Rscaled[rr] = (R[rr] - Re)*right + Re

                        spl = splrep(Rscaled, VTd[indx])
                        VTd[indx] = splev(R, spl)
                        lsqpars = lsqpars[2:]

                    case 'Wei' | 'Julienne':
                        analyt_dict = state_dict[param].copy()

                        for k, v in state_dict[param].items():
                            if k not in ['Rm', 'Rn']:
                                # reassign analytical PEC parameter in dict
                                analyt_dict[k] = v*lsqpars.pop(0)

                        subR = np.logical_and(R >= value['Rm'],
                                              R <= value['Rn'])

                        VTd[indx][subR] = eval(param)(R[subR], **analyt_dict)

                    case 'spline':
                        spl = splrep(value, lsqpars[:len(value)])

                        subR = np.logical_and(R >= value[0], R <= value[-1])
                        VTd[indx][subR] *= splev(R[subR], spl)

                        lsqpars = lsqpars[len(value):]

            # return modified PECs to csemodel
            self.csemodel.us.VT[indx, indx] = VTd[indx]/self._evcm  # diagonal

        # coupling -----------------------------------------
        for lbl, scaling in self.coup_adj.items():
            scaling = lsqpars.pop(0)
            coupling, i, j = self.coupling[lbl]  # original value tuple
            self.csemodel.us.VT[i][j] = self.csemodel.us.VT[j][i]\
                                      = coupling*scaling
            print(f'coupling {i}{j} = '
                  f'{(coupling*self._evcm*scaling).max():8.2f}')

        # etdm -------------------------------------------------
        for lbl, scaling in self.etdm_adj.items():
            scaling = lsqpars.pop(0)
            etdm, indxi, indxf = self.etdm[lbl]  # original value tuple
            self.csemodel.dipolemoment[:, indxi, indxf] = etdm*scaling


    def print_result(self):
        lsqpars = list(self.result.x)
        stderr = list(self.result.stderr)

        unit = {'Wei': {'Re':'Å', 'De':'cm⁻¹', 'Voo':'cm⁻¹', 'b':'', 'h':''},
                'Julienne': {'Mx':'cm⁻¹/Å', 'Rx':'Å', 'Vx':'cm⁻¹',
                             'Voo':'cm⁻¹'},
                'Rstr':{'left':'', 'right':''}
               }

        print('\n\nModel fitted parameters')

        if self.VT_adj:
            print('Potential energy curves ----------------------------')
            for state, state_dict in self.VT_adj.items():
                print(f'{state:10s} ', end='')
                for param, value in state_dict.items():
                    match param:
                        # single value
                        case 'ΔR' | 'ΔV' | 'Vstr':
                            print(f'{param:15s} '
                                  f'{value*lsqpars.pop(0):8.3f}±'
                                  f'{value*stderr.pop(0):.3f} cm⁻¹') 
                        # multi-value
                        case 'Wei' | 'Julienne' | 'Rstr':
                            print(f'{param:25s} ')
                            for k, v in value.items():
                                if k in ['Rm', 'Rn']:
                                    continue
                                scaling = lsqpars.pop(0)
                                scaling_err = stderr.pop(0)
                                print(f'{" ":27s} ',
                                      f'{scaling:8.3f}±{scaling_err:.3f}'
                                      f'{k:>5s} = {v*scaling:12.3f}±'
                                      f'{v*scaling_err:.3f} '
                                      f'{unit[param][k]}')
                        case 'spline':
                            print(f'{param:6s}   r(Å)         scaling')
                            for r in value:  # here an array
                                print(f'{" ":16s} {r:8.3f}  '
                                      f'{lsqpars.pop(0):12.3f}±'
                                      f'{stderr.pop(0):.3f}')
                print()

        if self.coup_adj:
            print('\nCoupling ------------------------------------')
            for lbl, v in self.coup_adj.items():
                scaling = lsqpars.pop(0)
                scaling_err = stderr.pop(0)

                coupling, i, j = self.coupling[lbl]
                ix = np.abs(self.csemodel.us.VT[i, i] -
                            self.csemodel.us.VT[j, j]).argmin()
                Rx = self.R[ix]

                print(f'{lbl:>10s} {" ":14s} {scaling:8.3f}±{scaling_err:.3f}',
                      end='')
                print(f'  {coupling[ix]*scaling*self._evcm:8.3f}±'
                      f'{coupling[ix]*scaling_err*self._evcm:.3f}'
                      f' cm⁻¹ at {Rx:5.3f} Å')

        if self.etdm_adj:
            print('\nElectronic transition dipole moment ---------')
            ire = self.csemodel.us.VT[0, 0].argmin()
            Re = self.R[ire]
            for lbl, v in self.etdm_adj.items():
                scaling = lsqpars.pop(0)
                scaling_err = stderr.pop(0)
                etdm, i, f = self.etdm[lbl]
                etdm = etdm[ire]
                print(f'{lbl:>10s} {" ":15s} '
                      f'{scaling:8.3f}±{scaling_err:.3f}'
                      f' {etdm*scaling:8.3f}±{etdm*scaling_err:.3f} a.u.',
                      f' at {Re:5.3f} Å')

        print()


    def cross_section(self, data, channel, eni=1100, roti=0, rotf=0):
        if data[0][0] < 100:
            dwn = 1000
            wavenumber = []
            self.peak = np.zeros_like(data[1])
            for p in data[1]:
                wavenumber.append(np.arange(p-dwn, p+dwn, 5))
            wavenumber = np.ravel(wavenumber)
        else:
            wavenumber = data[0]

        self.csemodel.calculate_xs(transition_energy=wavenumber,
                                   eni=eni, rotf=rotf, roti=roti)

        if channel == 'total':
            self.csexs = self.csemodel.xs.sum(axis=1)
        else:
            chnl_indx = self.csemodel.us.statelabel.index(channel)
            self.csexs = self.csemodel.xs[:, chnl_indx]

        if data[0][0] < 100:
            # peak position for each input value
            for i, p in enumerate(data[1]):
                subr = np.logical_and(wavenumber > p-dwn, wavenumber < p+dwn)
                self.peak[i] = wavenumber[subr][self.csexs[subr].argmax()]
            

    # least-squares fit -----------------------------------------------------
    def residual(self, pars):
        self.parameter_unpack(pars)  # resets csemodel with modified parameters

        self.diff = []

        for channel, data_dict in self.data2fit.items():
            for data_type, data in data_dict.items():
                match data_type[:2]:
                    case 'xs':
                        self.cross_section(data, channel)
                        diff = (csexs - data[1])*1e19
                        self.diff.append(diff)

                    case 'po':
                        self.cross_section(data, channel, eni=1100)
                        self.diff.append(self.peak - data[1])
                        print('position: ', self.diff[-1])

                    case 'Bv':
                        self.cross_section(data, channel, eni=1100)
                        self.peak0 = self.peak
                        self.cross_section(data, channel, eni=1200,
                                           roti=10, rotf=10)
                        Bv = np.abs(self.peak0 - self.peak)/10/11
                        self.diff.append(Bv - data[1])
                        print('Bv: ', self.diff[-1])

        self.diff = np.hstack((self.diff))
        self.sum = self.diff.sum()

        if self.verbose:
            print(f'{pars} {self.sum:g}')
        else:
            print('.', end='')

        return self.diff

    def fit(self):

        self.parameter_pack()

        if not self.verbose:
            print('Model_fit: each "." represents an iteration')

        if self.method == 'lm':
            self.result = least_squares(self.residual, self.lsqpars,
                                        method=self.method,
                                        x_scale='jac', diff_step=0.1)
        else:
            self.result = least_squares(self.residual, self.lsqpars,
                                        method=self.method, bounds=self.bounds,
                                        x_scale='jac', diff_step=0.1)
