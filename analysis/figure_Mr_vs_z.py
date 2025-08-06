import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.cosmology import LambdaCDM
cosmo = LambdaCDM(70.,0.3,0.7)
from matplotlib.ticker import MaxNLocator
from matplotlib import rcParams
from matplotlib.colors import LogNorm
rcParams['axes.labelsize'] = 9
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['legend.fontsize'] = 9
rcParams['font.family'] = 'sans-serif'
#rcParams['font.sans-serif'] = ['Helvetica']
#rcParams['text.usetex'] = True
rcParams['grid.color'] = 'k'
rcParams['grid.linewidth'] = 0.2
my_locator = MaxNLocator(6)
singlecolsize = (3.3522420091324205, 2.0717995001590714)
doublecolsize = (7.100005949910059, 4.3880449973709)

df = pd.read_hdf("../data/zou/xmmservs_laduma_merged.hdf5")
df = df[(df.absrmag>-990)]

zz = np.linspace(0,2,1000)
limit = 26 + 5 - 5 * np.log10(cosmo.luminosity_distance(zz).to_value()*1e6)
samplebinedges = np.arange(0,1.2,0.2)
box_xy = [(leftedgex, complimit) for leftedgex, complimit in zip(samplebinedges[:-1],np.interp(samplebinedges[1:],zz,limit))]
boxwidth = np.diff(samplebinedges)


plt.figure(figsize=(singlecolsize[0],singlecolsize[1]*1.3))
#plt.scatter(df.bestoverallredshift, df.absrmag, color='cornflowerblue', s=1, alpha=0.05, rasterized=True)
plt.scatter(df.redshift, df.absrmag, color='cornflowerblue', s=1, alpha=0.05, rasterized=True)
plt.plot(zz,limit,color='k', linewidth=1.5, label=r'$m_r = +26$ (VOICE $r$ Limit)', linestyle='dashed')
for ii,box in enumerate(box_xy):
    rect = plt.Rectangle(xy=box, width=boxwidth[ii], height=-20, facecolor='None', edgecolor='k', linewidth=2)
    plt.gca().add_patch(rect)
plt.xlim(0,2)
plt.ylim(-24,-12)
plt.xlim(0,1.0)
plt.legend(loc='best',fontsize=8)
plt.gca().invert_yaxis()
plt.xlabel(r'$z_{\rm best}$')
plt.ylabel(r'$M_r$')
plt.tight_layout()
plt.savefig("../figures/LADUMA_Mr_vs_z.pdf",dpi=300)
plt.show()

