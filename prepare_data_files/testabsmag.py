import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.cosmology import LambdaCDM
cosmo = LambdaCDM(70,0.3,0.7)

df = pd.read_csv("xmmservs_w_absrmag_specz.csv")
df = df[(df.mag_R_VOICE>0) & (df.zphot >0) & (df.ngoodband > 3) & (df.redchi2_gal<2)]

Mr = df.mag_R_VOICE + 5 - 5*np.log10(cosmo.luminosity_distance(df.redshift.to_numpy()).to('pc').value)

plt.figure()
sc=plt.scatter(df.redshift, Mr-df.absmag_R_VOICE, c=df.ngoodband, s=1, alpha=0.5, vmin=0, vmax=10)
plt.colorbar(sc)
plt.xlabel('redshift')
plt.ylabel('Mr(dist mod) - Mr(SED)')
plt.ylim(-10,10)
plt.xlim(0,2)
plt.show()

