'''Ocean box models.'''

import numpy as np
import xarray as xr
import constants
import solubility

def rk4(dfdt, dt, t, y, kwargs={}):
    '''4th order Runge-Kutta.'''

    dydt1, diag1 = dfdt(t, y, **kwargs)
    dydt2, diag2 = dfdt(t + dt / 2., y + dt*dydt1 / 2., **kwargs)
    dydt3, diag3 = dfdt(t + dt / 2, y + dt*dydt2 / 2., **kwargs)
    dydt4, diag4 = dfdt(t + dt, y + dt*dydt3, **kwargs)

    y_next_t = y + dt * (dydt1 + 2. * dydt2 + 2. * dydt3 + dydt4) / 6.

    diag_t = diag1
    for key in diag1.keys():
        diag_t[key] = (diag1[key] + 2. * diag2[key] + 2. * diag3[key]
                       + diag4[key]) / 6.

    return y_next_t, diag_t

class two_box_ocean(object):
    '''A two box ocean model.'''

    def __init__(self, **kwargs):
        '''Initialize model.'''

        #-- set-able parameters
        tau_bio_year = kwargs.pop('tau_bio_year', 0.5)
        h_surf_m = kwargs.pop('h_surf_m', 100.)
        PO4_ocn_mean = kwargs.pop('PO4_ocn_mean_mmolm3', 2.1) # mmol/m^3
        S_psu = kwargs.pop('S_psu', 35.)

        self.invtau_bio = 1. / (tau_bio_year * constants.spy)  # 1/s
        self.S = S_psu # salinity
        self.PO4_ocn_mean = PO4_ocn_mean # mean PO4 concentration used for inventory

        #-- tracers and their relationships
        self.tracers = ['PO4', 'O2']
        self.mol_ratio = np.array([1., -150.])
        self.ntracers = len(self.tracers)
        self.ind = {k: self.tracers.index(k) for k in self.tracers}

        #-- geometry
        self.boxes = ['surface', 'deep']
        self.nboxes = len(self.boxes)
        self.V = np.empty((self.nboxes))
        self.V[0] = constants.A_ocn * h_surf_m
        self.V[1] = (constants.V_ocn - self.V[0])

        #-- integration
        self.dt = 365. * 86400. # seconds

        #-- initialize data structures
        self.forcing_t = xr.Dataset()
        self.state = np.empty((self.nboxes, self.ntracers))


    def equilibrium_solution(self, SST, Psi, print_state=True):
        '''Compute steady-state solution.'''

        O2_surf = solubility.O2(self.S, SST)
        bioP = self.invtau_bio * self.V[0]
        bioO2 = self.mol_ratio[self.ind['O2']] * bioP
        P_ocn_inv = self.PO4_ocn_mean * self.V.sum()


        #-- matrix of equations
        # surface box P: Psi * (P_deep - P_surf) - 1/tau_bio * V_surf * P_surf = 0.
        # P conservation: V_surf * P_surf + V_deep * P_deep = P_ocn_inv
        # deep box O2:  1/tau_bio * mol_ratio * P_surf - Psi * O2_deep = -Psi * O2_surf
        A = np.array([[-(Psi + bioP), Psi, 0.],
                      [self.V[0], self.V[1], 0.],
                      [bioO2, 0., -Psi]])

        #-- righthand side
        b = np.array([0., P_ocn_inv, -Psi * O2_surf])

        #-- solve system of equations
        state_init_part = np.linalg.solve(A, b)

        #-- assign result
        state_init = np.empty((self.nboxes, self.ntracers))

        PO4_surf = state_init_part[0]
        PO4_deep = state_init_part[1]
        O2_deep = state_init_part[2]

        state_init[0, self.ind['PO4']] = PO4_surf
        state_init[1, self.ind['PO4']] = PO4_deep
        state_init[0, self.ind['O2']] = O2_surf
        state_init[1, self.ind['O2']] = O2_deep

        export = self.invtau_bio * state_init[0,self.ind['PO4']] * self.V[0]
        export = export * constants.spy * 1e-3 * 106. * 12e-15

        if print_state:
            print(f'Equilibrium state:')
            print(f'Carbon export: {export:0.3f} PgC/yr')
            print(f'PO4 (mmol/m^3)\n  surface: {PO4_surf:0.3f} \n'
                  f'  deep: {PO4_deep:0.3f}')
            print(f'O2 (mmol/m^3)\n  surface: {O2_surf:0.3f}\n'
                  f'  deep: {O2_deep:0.3f}')

        return state_init

    def _define_diags(self):
        '''Define diagnostics.'''
        return {'NCP': {'dims': ('time'),
                        'attrs': {'long_name': 'NCP',
                                  'units': 'PgC/yr'}},
                 'AOU': {'dims': ('time'),
                         'attrs': {'long_name': 'AOU',
                                   'units': 'mmol/m$^3$'}},
                 }

    def _compute_tendencies(self, t, state):
        '''Compute tendencies.'''

        diag_values = {}

        ind = self.ind
        Psi = self.forcing_t['Psi'].values
        PO4 = state[:, ind['PO4']]
        O2 = state[:, ind['O2']]

        # initialize tendency
        dcdt = np.zeros((self.nboxes, self.ntracers))

        # add transport tendency: m^3/s * mmol/m^3 --> mmol/s
        dcdt[0, :] += Psi * (state[1, :] - state[0, :])
        dcdt[1, :] += Psi * (state[0, :] - state[1, :])

        # add biological tendency: 1/s * mmol/m^3 * m^3--> mmol/s
        ncp = self.invtau_bio * PO4[0] * self.V[0] #
        dcdt[0, :] += (-1.0) * self.mol_ratio * ncp
        dcdt[1, :] += self.mol_ratio * ncp

        # divide by volume: mmol/m^3/s
        dcdt = dcdt / self.V

        # compute diagnostics
        diag_values['NCP'] = ncp * constants.spy * 1e-3 * 106. * 12e-15
        diag_values['AOU'] = O2[0] - O2[1]

        return dcdt, diag_values


    def _reset(self):
        '''Reset tracer concentrations after time-step update.'''
        ind = self.ind
        T = self.forcing_t['SST'].values
        self.state[0, ind['O2']] = solubility.O2(self.S, T)


    def run(self, start, stop, forcing, state_init=None):
        '''Integrate the model in time.'''

        #-- pointers for local variable
        ind = self.ind
        dt = self.dt
        func = self._compute_tendencies

        #-- time axis
        time = np.arange(start * constants.spy, stop * constants.spy + dt, dt)
        nt = len(time)

        #-- initialize state
        self._interp_forcing(time[0], forcing)
        if state_init is None:
            state_init = self.equilibrium_solution(SST=self.forcing_t['SST'],
                                                   Psi=self.forcing_t['Psi'],
                                                   print_state=False)
        self.state = state_init

        #-- initialize output dataset
        output = xr.Dataset()
        for tracer in self.tracers:
            output[tracer] = xr.DataArray(np.empty((nt-1, self.nboxes)),
                                          dims=('time', 'box'),
                                          attrs={'units': 'mmol/m$^3$',
                                                 'long_name': tracer})

        for var in self.forcing_t:
            output[var] = xr.DataArray(np.empty((nt-1)), dims=('time'))

        for key, val in self._define_diags().items():
            output[key] = xr.DataArray(np.empty((nt-1)), **val)

        output['time'] = xr.DataArray(time[1:] / constants.spy, dims=('time'),
                                      attrs={'units': 'years'})
        output['box'] = xr.DataArray(self.boxes, dims=('box'))

        #-- begin timestepping
        for l in range(1, nt):

            #-- interpolate forcing
            self._interp_forcing(time[l], forcing)

            #-- integrated
            self.state, diag_t = rk4(func, dt, time[l], self.state)
            self._reset()

            #-- save output
            for tracer in self.tracers:
                output[tracer][l-1, :] = self.state[:, ind[tracer]]

            for var in self.forcing_t:
                output[var][l-1] = self.forcing_t[var]

            for key, val in diag_t.items():
                output[key][l-1] = val


        return output

    def _interp_forcing(self, t, forcing):
        '''Interpolate forcing dataset at time = t.'''

        self.forcing_t = forcing.interp({'time': t / constants.spy})
