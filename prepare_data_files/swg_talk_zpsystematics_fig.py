import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import median_abs_deviation

def nmad(*args,**kwargs):
    return 1.4826*median_abs_deviation(*args,**kwargs)


df = pd.read_csv("xmmservs_w_absrmag_specz.csv")
df = df[(df.zphot>0) & (df.zspecbest>0)]

x = (df.zphot - df.zspecbest)/df.zphoterr
med = np.median(x)
nm = nmad(x)


plt.figure()
plt.hist(x, bins=np.arange(-3,3,0.1), density=True)
plt.axvline(med,color='k')
plt.xlabel(r'$(z_{\rm phot} - z_{\rm spec})/\sigma_{phot-z}$', fontsize=14)
plt.annotate(xy=(0.05,0.8), xycoords='axes fraction', text=f'Med={med:0.2f}',fontsize=14)
plt.annotate(xy=(0.05,0.7), xycoords='axes fraction', text=f'NMAD={nm:0.2f}',fontsize=14)
plt.show()

