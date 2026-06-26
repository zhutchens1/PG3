import numpy as np
from prob_giantonlyic import *

z1 = 0.023
z2 = 0.022
e1 = 1e-3
e2 = 4e-3
VL = 400/3e5
zgrid = np.arange(0,0.5,1/3e5)

# Galaxy j is gaussian
pz1 = get_pz_group(zgrid, z1, 1/(np.sqrt(2*np.pi)*e1), -0.5/(e1*e1))
pA = dbint_pz_jgauss(zgrid, pz1, z2, e2, VL)
print(pA)

# second method
pz2 = get_pz_group(zgrid, z2, 1/(np.sqrt(2*np.pi)*e2), -0.5/(e2*e2))
pB = dbint_pz_general(zgrid, pz1, pz2, VL)
print(pB)
