import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("/afs/cas.unc.edu/users/z/h/zhutchen/Documents/temp_eco_sdss_crossmatch/eco_newczforupload_102120.txt")
df = df[df.newczerr != 100.]

plt.figure()
plt.hist(df.newczerr, bins=np.arange(0,105,5))
#plt.yscale('log')
plt.xlabel('cz error')
plt.show()
