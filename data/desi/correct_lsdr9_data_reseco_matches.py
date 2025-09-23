import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,'/srv/one/zhutchen/paper3/codes/')
from photzcorrection import corrector, nmad
from scipy.stats import norm
from matplotlib.ticker import MaxNLocator
from matplotlib import rcParams
from matplotlib.colors import LogNorm
from scipy.stats import skewtest
rcParams['axes.labelsize'] = 9
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['legend.fontsize'] = 9
rcParams['font.family'] = 'sans-serif'
rcParams['grid.color'] = 'k'
rcParams['grid.linewidth'] = 0.2
my_locator = MaxNLocator(6)
singlecolsize = (3.3522420091324205, 2.0717995001590714)
doublecolsize = (7.100005949910059, 4.3880449973709)

if __name__ == '__main__':
    LS = pd.read_csv("/srv/one/hperk4/decals_eco_resolve_allmatches.csv")
    LS = LS[['name','radeg','dedeg','cz','rmag','ra','dec','ls_id','dec','ra','shape_r','flux_r','mag_r',\
            'type','z_phot_median','z_phot_std','z_spec','absolute_rmag','Separation']]
    LS.loc[:,'z_reseco'] = LS.loc[:,'cz']/2.998e5
    LS = LS[~pd.isna(LS.z_phot_median)]
    assert len(np.unique(LS.name)) == len(LS)
    #plt.figure()
    #plt.scatter(LS.rmag, LS.rmag-LS.mag_r, s=2, color='k', alpha=0.5)
    #plt.xlabel("RESOLVE/ECO r mag")
    #plt.ylabel("RESOLVE/ECO r mag - LS mag_r")
    #plt.show()

    ct = corrector(LS, 'z_phot_median', 'z_phot_std', 'z_reseco', 'mag_r', view_intermediate_plots=False, force_linear_fit2=True, force_linear_fit1=True)
    zphotcorr, sigcorr, flag = ct.run()


    # ---------------------------------------------------------------------------------------------------- #
    # ---------------------------------------------------------------------------------------------------- #
    # Make figure for paper
    # ---------------------------------------------------------------------------------------------------- #
    # ---------------------------------------------------------------------------------------------------- #
    gaussx = np.linspace(-4,4,5000)
    gaussy = norm.pdf(gaussx) 

    fig, axs = plt.subplots(ncols=2,nrows=2, figsize=(doublecolsize[0], doublecolsize[1]*0.8))
    # top left
    errdist = (LS.z_phot_median - LS.z_reseco)/LS.z_phot_std
    med = np.median(errdist)
    scl = nmad(errdist)
    axs[0][0].hist(errdist, bins=np.arange(-4,4,0.1), density=True)
    axs[0][0].plot(gaussx, gaussy, color='orange', alpha=0.8)
    axs[0][0].set_xlabel(r"$(z_{\rm phot} - z_{\rm spec})/\sigma_{\rm phot}$")
    label = f'Med={med:0.2f}\n' + r'NMAD$=$' + f'{scl:0.2f}'
    axs[0][0].annotate(xy=(0.05,0.6), xycoords='axes fraction', text=label, fontsize=9)
    axs[0][0].set_xlim(-4,4)
    axs[0][0].set_ylim(0,0.7)

    # top right
    tt = np.linspace(12,18,100)
    axs[0][1].plot(tt,ct.model1(tt), color='red',alpha=0.7)
    axs[0][1].errorbar(ct.magbinc, ct.mediandz, ct.mediandzerr, fmt='k.', capsize=3)
    axs[0][1].set_xlim(13,18)
    axs[0][1].set_ylim(0,4e-3)
    axs[0][1].invert_xaxis()
    axs[0][1].set_xlabel(r'$m_r$')
    axs[0][1].set_ylabel(r'$z_{\rm phot}-z_{\rm spec}$')

    # bottom left
    axs[1][0].plot(tt,ct.model2(tt),color='red', alpha=0.7)
    axs[1][0].errorbar(ct.magbinc, ct.nmad_dzcorr_over_etab, ct.nmad_dzcorr_over_etab_err,\
                        fmt='k.', capsize=3)
    axs[1][0].set_xlim(13,18)
    axs[1][0].set_ylim(0.4,0.9)
    axs[1][0].invert_xaxis()
    axs[1][0].set_xlabel('$m_r$')
    axs[1][0].set_ylabel(r'NMAD($dz_{\rm corr}/\sigma_{\rm phot}$)')

    # bottom right
    errdist = (zphotcorr - LS.z_reseco)/sigcorr
    print(skewtest(errdist))
    axs[1][1].hist(errdist, bins=np.arange(-4,4,0.1), density=True)
    axs[1][1].plot(gaussx, gaussy, color='orange', alpha=0.8)
    med = np.median(errdist)
    scl = nmad(errdist)
    label = f'Med={med:0.2f}\n' + r'NMAD$=$' + f'{scl:0.2f}'
    axs[1][1].annotate(xy=(0.05,0.6), xycoords='axes fraction', text=label, fontsize=9)
    axs[1][1].set_xlabel(r'$(z_{\rm phot,corr}-z_{\rm spec})/\sigma_{\rm phot,corr}$')
    axs[1][1].set_xlim(-4,4)
    axs[1][1].set_ylim(0,0.45)

    # other
    labels=[['(a)','(b)'],['(c)','(d)']]
    for i in range(0,2):
        for j in range(0,2):
            axs[i][j].annotate(xy=(0.85,0.5), xycoords='axes fraction', text=labels[i][j], fontsize=16)
    plt.tight_layout()
    plt.savefig("/srv/one/zhutchen/paper3/figures/lsdr9_photz_corr_demo.pdf",dpi=300)
    plt.show()

    #plt.figure()
    #plt.plot(LS.z_reseco, errdist, 'k.', alpha=0.7)
    #plt.show()
