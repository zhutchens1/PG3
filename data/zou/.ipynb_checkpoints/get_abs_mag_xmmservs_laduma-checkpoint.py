from astropy.io import fits
import matplotlib.pyplot as plt
import speclite
from speclite import filters
import numpy as np
import pandas as pd
from astropy.io.misc.hdf5 import read_table_hdf5
import h5py
import astropy.units as u
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d
from tqdm import tqdm 
from seaborn import kdeplot
def get_abs_mag(obs_wvln, specific_lum, zz, bandpassfn):
    """
    compute absolute magnitude from XMM-SERVS SED
    
    obs_wvln : observed frame wavelength in nm
    specific_lum: specific luminosity in W/nm
    zz: galaxy redshift
    bandpassfn: callable function - sensitivity as a function of wavelength in given filter
        should be callable function of wavelength [nm]
    """
    rest_wvln = obs_wvln / (1 + zz)
    Lsun = 3.846e26
    W_per_ergss = 1e-7
    cm_per_Mpc = 3.086e24
    dist_cm = 1e-5 * cm_per_Mpc
    nm_per_AA = 0.1 # nm/AA
    rbandzeropoint_flambda = 278e-11 # ergs/s/cm2/AA
    rbandzeropoint_Llambda = rbandzeropoint_flambda * 4 * np.pi * dist_cm * dist_cm # ergs/s/AA
    rbandzeropoint_Llambda = rbandzeropoint_Llambda * W_per_ergss # W/AA
    rbandzeropoint_Llambda = rbandzeropoint_Llambda / nm_per_AA
    zeropointC = 2.5*np.log10(trapezoid(rbandzeropoint_Llambda*bandpassfn(rest_wvln),rest_wvln))
    
    integrand = (specific_lum)*bandpassfn(rest_wvln)
    Lr = trapezoid(integrand,rest_wvln)
    MM = (-2.5*np.log10(Lr)+zeropointC)
    return MM


if __name__=='__main__':
    rband = filters.load_filter('sdss2010-r')
    rbandwavelength=np.array(rband.wavelength)*0.1 # 0.1 nm per AA
    rbandresponse=np.array(rband.response)
    rbandresponseinterp = interp1d(rbandwavelength,rbandresponse,bounds_error=False,fill_value=0)

    xmmservs = pd.read_csv("xmmservs_with_ladumaAAT.csv")
    print([k for k in xmmservs.keys()])
    #xmmservs = xmmservs[(xmmservs.zspec>0)&(xmmservs.zspec<0.06)]
    xmmservs = xmmservs[(np.isnan(xmmservs.zspec.to_numpy()))&(np.isnan(xmmservs.z.to_numpy()))]

    zbest = xmmservs.zphot.to_numpy()
    zspec = xmmservs.zspec.to_numpy()
    zspecaat = xmmservs.z.to_numpy()
    sel = (zspecaat>0)
    zbest[sel] = zspecaat[sel]
    sel = (zspec>0)
    zbest[sel] = zspec[sel]
    xmmservs.loc[:,'zbest']=zbest
    xmmservs = xmmservs[(xmmservs.zbest>0.4)&(xmmservs.zbest<0.6)]

    z_for_Mr = xmmservs.zphot.to_numpy() # XMM-SERVS doesn't have aat redshifts
    sel = (xmmservs.zspec.to_numpy()>0)
    z_for_Mr[sel] = xmmservs.zspec.to_numpy()[sel]

    seds = h5py.File('./seds/wcdfs/models_gal_wcdfs.h5','r')
    absrmag = np.zeros(len(xmmservs))
    TractorIDfixed = np.float64([x[2:-1] for x in xmmservs.Tractor_ID.to_numpy()]).astype(int).astype(str)
    print(TractorIDfixed)
    for ii,TracID in enumerate(tqdm(xmmservs.Tractor_ID.to_numpy())):
        wavelength_arr = np.array(seds[str(TractorIDfixed[ii])]['wavelength'])*1000. # 1000 = nm/micron
        specific_lum_arr = np.array(seds[str(TractorIDfixed[ii])]['L_lambda_total'])
        #absrmag[ii] = get_abs_mag(wavelength_arr, specific_lum_arr, xmmservs[xmmservs.Tractor_ID==TracID].zbest.squeeze(), rbandresponseinterp) # need z best here
        absrmag[ii] = get_abs_mag(wavelength_arr, specific_lum_arr, z_for_Mr[ii], rbandresponseinterp) # need z best here

    eco=pd.read_csv("ECOdata_080822.csv")
    fig,ax=plt.subplots()
    ax.scatter(absrmag,np.log10(xmmservs.Mstar_best.to_numpy()),color='orange',s=2,alpha=0.1, label='XMM-SERVS')
    kdeplot(eco.absrmag,eco.logmstar,color='blue', label='ECO')
    ax.set_xlabel("Absolute r-band Mag from SEDs")
    ax.set_ylabel("log stellar mass")
    ax.invert_xaxis()
    ax.legend(loc='best')
    plt.show()

    xmmservs.loc[:,'absrmag']=absrmag
    xmmservs[xmmservs.absrmag<=-19.5][['RA_1','DEC_1','zphot','absrmag']].to_csv("giants_missing_specz.csv",index=False)
    #plt.figure()
    #c_over_H0 = 2.998e+5/70
    #tx = np.linspace(-30,0)
    #plt.plot(absrmag, xmmservs.mag_R_VOICE - 5*np.log10(c_over_H0*xmmservs.zbest * 1e6 / 10), 'k.', alpha=0.1)
    #plt.plot(tx,tx,color='orange')
    #plt.show()
        
    


"""
    xmmservs = pd.read_hdf("XMMSERVS_all.hdf5")
    specz_sample = xmmservs[(xmmservs.zspec>0.002)&(xmmservs.zspec<0.06)].zspec
    specz_df = xmmservs[(xmmservs.zspec>0.002)&(xmmservs.zspec<0.06)]
    seds = h5py.File('./seds/wcdfs/models_gal_wcdfs.h5', 'r')

    conv_abs_mags=[]
    calc_abs_mags=[]
    c_over_H0 = 2.998e+5/70.
    for ii in specz_sample.index:
        conv_abs_mags.append(get_abs_mag(np.array(seds[str(ii)]['wavelength'])*1000., np.array(seds[str(ii)]['L_lambda_total']), xmmservs.loc[ii,'zspec'], rbandresponseinterp))
        dist_pc = c_over_H0*xmmservs.loc[ii,:].zspec * 1e6
        calc_abs_mags.append(xmmservs.loc[ii,'mag_R_VOICE'] - 5*np.log10(dist_pc/10))
    conv_abs_mags=np.array(conv_abs_mags)
    calc_abs_mags=np.array(calc_abs_mags)

    plt.figure()
    #tx=np.linspace(0,0.06,10)
    tx=np.linspace(-30,0,10)
    plt.plot(tx,tx,label='1:1 Line',color='lightgreen')
    plt.plot(np.array(conv_abs_mags),calc_abs_mags,'k.')
    #pickle.dump([np.array(conv_abs_mags),np.array(calc_abs_mags)],open('magmatch.pkl','wb'))

    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    print(np.nanmedian(np.abs(np.array(conv_abs_mags)-np.array(calc_abs_mags))))
    plt.xlabel("Rest-frame abs mag. from filter convolution",fontsize=12)
    plt.ylabel(r"$m - 5\log_{10}\left(d\, / \, {\rm 10\,pc}\right)$",fontsize=12)
    plt.title(r"XMM SERVS $0 < z_{\rm spec} < 0.06$",fontsize=12)
    plt.legend(loc='best',fontsize=11)
    plt.xlim(-8,-23)
    plt.ylim(-8,-23)
    plt.show()
"""
