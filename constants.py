'''Common constants'''

import numpy as np

# thermodynamics
T0_Kelvin = 273.15
R_gasconst = 8.3144621 # J/mol/K
boltz = 8.6173324E-5 # eV/K
cp = 3985. # J/kg/K


# time conversions
spd = 86400 # seconds per day
spy = spd * 365 # seconds per year

# earth
g = 9.8
R_earth = 6371e3 # m; earth radius
A_earth = 4. * np.pi * R_earth**2 # m^2; earth area
frac_lnd = 0.3 # fraction land area
frac_ocn = 1 - frac_lnd # fraction ocean area
A_ocn = A_earth * frac_ocn # m^2; ocean area
V_ocn = 1.292e18 # m^3; ocean volume
Ps = 1013.5 * (1e2) # Pa; mean surface pressure
rho_ref = 1026.

# numerics
epsTinv = 3.17e-8 # small inverse time scale: 1/year (1/sec)
epsC = 1.00e-8

# isotope standards
PDB_Rstd = 11237.2e-6  #  PDB standard

# molecular weights
mw = {'O2':   32.,
      'N2' :  28.,
      'Ne' :  20.,
      'Ar' :  40.,
      'Kr' :  84.,
      'Xe' : 131.}
