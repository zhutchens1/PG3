import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,'../codes/')
from photzcorrection import corrector
from survey_volume import solid_angle, integrate_volume
from copy import deepcopy
from astropy.cosmology import LambdaCDM
from speclite import filters
import astropy.units as u
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d
import h5py
from pathos.multiprocessing import ProcessingPool
from tqdm import tqdm
cosmo = LambdaCDM(70,0.3,0.7)

direc = dict(
    redshift_catalog = './z22_w_laduma_specz.csv',
    subvolpath = '/srv/one/zhutchen/paper3/data/subvolumes/',
    sedfile = '/srv/one/zhutchen/paper3/data/zou/seds/wcdfs/models_gal_wcdfs.h5',
)
def absMag_distmod(z, m):
    return m+5-5*np.log10(cosmo.luminosity_distance(z).to('pc').value)

def get_abs_mag(tractorID, xmmservs_cat, xmmservs_seds, zbest_key):
    try:
        obsv_wavelength = np.array(xmmservs_seds[str(tractorID)]['wavelength'])*1000. # 1000 nm/micron
    except KeyError:
        #print(f"failed on {tractorID} - could not find SED.")
        return -999
    z_sed =xmmservs_cat.loc[tractorID,'redshift']
    z_best=xmmservs_cat.loc[tractorID,zbest_key]
    rest_wavelength = obsv_wavelength / (1+z_best)
    correction = (cosmo.luminosity_distance(z_best)/cosmo.luminosity_distance(z_sed))**2.0
    obsv_specific_lum = np.array(xmmservs_seds[str(tractorID)]['L_lambda_total'])*correction
    rband = filters.load_filter('sdss2010-r')
    rbandwavelength=np.array(rband.wavelength)*0.1 # 0.1 nm per AA
    rbandresponse=np.array(rband.response)
    rbandresponseinterp = interp1d(rbandwavelength,rbandresponse,bounds_error=False,fill_value=0)

    #plt.figure()
    #plt.plot(obsv_wavelength, obsv_specific_lum,'k',label='Observed-Frame Wavelength')
    #plt.plot(rest_wavelength, obsv_specific_lum,'r',label='Rest-Frame wavelength')
    ##plt.xscale('log')
    #plt.axvline(np.min(rbandwavelength))
    #plt.axvline(np.max(rbandwavelength))
    #plt.yscale('log')
    #plt.xlabel(r"$\lambda$ [nm]")
    #plt.ylabel(r"Specific Luminosity $L_\lambda$ [W/nm]")
    ##plt.ylim(1e33,0.2e35)
    #plt.xlim(400,900)
    #plt.legend(loc='best')
    #plt.show()

    Lsun = 3.846e26
    W_per_ergss = 1e-7
    cm_per_Mpc = 3.086e24
    dist_cm = 1e-5 * cm_per_Mpc
    nm_per_AA = 0.1 # nm/AA
    rbandzeropoint_flambda = 278e-11 # ergs/s/cm2/AA
    rbandzeropoint_Llambda = rbandzeropoint_flambda * 4 * np.pi * dist_cm * dist_cm # ergs/s/AA
    rbandzeropoint_Llambda = rbandzeropoint_Llambda * W_per_ergss # W/AA
    rbandzeropoint_Llambda = rbandzeropoint_Llambda / nm_per_AA
    zeropointC = 2.5*np.log10(trapezoid(rbandzeropoint_Llambda*rbandresponseinterp(rest_wavelength),rest_wavelength))

    integrand = (obsv_specific_lum)*rbandresponseinterp(rest_wavelength)
    Lr = trapezoid(integrand,rest_wavelength)
    MM = (-2.5*np.log10(Lr)+zeropointC)
    #except KeyError:
    #    MM = -999
    #    print("KeyError Warning: failed on tractorID {} (SED not found and/or no phometry or redshift)".format(tractorID))
    return MM

if __name__=='__main__':
    print("WARNING -- there is a selection effect here in that redshifts don't look well corrected")
    print("when selected on bestzoverall rather than zphot. The selection of the volume-limited sample")
    print("probably needs to happen iteratively *with* the zphot corrections because they are intricately")
    print("related.")
        
    # ----------------------------------------------------------------- #
    # split redshift catalog into bins
    xmms = pd.read_csv(direc["redshift_catalog"])

    rband_voice_depth = 26
    iband_zp_depth = 24
    rband_zp_depth = np.median(xmms.mag_R_VOICE[(xmms.mag_I_VOICE>iband_zp_depth-0.1)&(xmms.mag_I_VOICE<iband_zp_depth+0.1)])
    xmms = xmms[(xmms.mag_R_VOICE<rband_zp_depth) & (xmms.mag_R_VOICE>0) & (xmms.mag_I_VOICE<iband_zp_depth)]

    #fitting_bins = np.arange(0,1.4,0.4)
    fitting_bins = [0,0.1,0.5,0.8,1.4]
    #fitting_bins = np.arange(0.0,1.5,0.1)
    xmms.loc[:,'zphotcorr_bin'] = np.digitize(xmms.zphot.to_numpy(), bins=fitting_bins)

    # ----------------------------------------------------------------- #
    # get corrected zphot data
    processed=[]
    counts=[]
    groups = xmms.groupby('zphotcorr_bin')
    for _, gg in groups:
        if (gg.zphotcorr_bin==0).all() or (gg.zphotcorr_bin==len(fitting_bins)).all():
            gg.loc[:,'zphotcorr'] = gg.loc[:,'zphot']
            gg.loc[:,'zphoterrcorr'] = gg.loc[:,'zphoterr']
            gg.loc[:,'zphotcorrected'] = 0 #1=corrected 0=not corr.
            processed.append(gg)
            counts.append(len(gg))
        else:
            print(f'Working on bin {gg.iloc[0].zphotcorr_bin}.')
            figpath = './zp_correction_plots_logs/zphot_corr_{zmin:0.2f}_to_{zmax:0.2f}.png'.format(zmin=np.min(gg.zphot), zmax=np.max(gg.zphot))
            logpath = figpath.replace('png','txt')
            ct = corrector(gg, 'zphot', 'zphoterr', 'zspecbest', 'mag_R_VOICE', 'goodzflag', imag_key='mag_I_VOICE', imag_cut=24, \
                    save_final_fig_path=figpath, log_file_path=logpath, bic_diff_thresh=0, force_linear_fit1=True, force_linear_fit2=False,\
                    convg_threshold=0.3)
            zpcorr, etabcorr, flag = ct.run()
            if (etabcorr<0).any():
                print('warning - some corrected uncertainties are negative.')
            gg.loc[:,'zphotcorr'] = zpcorr
            gg.loc[:,'zphoterrcorr'] = etabcorr
            gg.loc[:,'zphotcorrected'] = flag #1=corrected 0=not corr.
            processed.append(gg)
            counts.append(len(gg))

    xmms = pd.concat(processed, axis=0)
    assert len(xmms)==np.sum(counts)

    zbest = deepcopy(xmms.zphotcorr.to_numpy())
    zbesterr = deepcopy(xmms.zphoterrcorr.to_numpy())
    zbestflag = np.zeros_like(zbest)
    spec_avail = (xmms.zspecbest > 0).to_numpy()
    zbest[spec_avail] = xmms.zspecbest.to_numpy()[spec_avail]
    zbesterr[spec_avail] = xmms.zspecbesterr.to_numpy()[spec_avail]
    zbestflag[spec_avail] = 1
    xmms.loc[:,'bestoverallz'] = zbest
    xmms.loc[:,'bestoverallzerr'] = zbesterr
    xmms.loc[:,'bestoverallzflag'] = zbestflag #0=photz, 1=specz

    #sv = xmms[(xmms.bestoverallz>0) & (xmms.bestoverallz<=0.1) & ~pd.isna(xmms.mag_R_VOICE)]
    #plt.figure()
    #sel = (sv.zspecbest>0)
    #bv = np.arange(-4,4,0.1)
    #plt.hist((sv[sel].zphot-sv[sel].zspecbest)/sv[sel].zphoterr, bins=bv)
    #plt.hist((sv[sel].zphotcorr-sv[sel].zspecbest)/sv[sel].zphoterrcorr, bins=bv,alpha=0.5)
    #print(np.median(sv[sel].zphot-sv[sel].zspecbest))
    #print(np.median(sv[sel].zphotcorr-sv[sel].zspecbest))
    #plt.show()
    #exit()
    ### Calculate abs magnitudes
    seds = h5py.File(direc['sedfile'],'r')
    xmms = xmms.set_index('Tractor_ID')
    absrmag = np.zeros(len(xmms))
    worker = lambda Tid: get_abs_mag(Tid, xmms, seds, 'bestoverallz')
    ids_to_process = xmms.index.to_numpy()
    with ProcessingPool(60) as pool:
        absm = list(tqdm(pool.imap(worker,ids_to_process),total=len(ids_to_process)))
    xmms.loc[:,'absmag_R_VOICE'] = absm

    ### Reduce selection and unnecessary columns 
    selection = (xmms.bestoverallz>0) & (xmms.bestoverallz < np.max(fitting_bins)) & (xmms.mag_R_VOICE<rband_zp_depth) & (xmms.mag_R_VOICE>0)
    magcols = [kk for kk in xmms.columns if (kk.startswith('mag') or kk.startswith('absmag'))]
    zcols = [kk for kk in xmms.columns if (kk.endswith('zspec_cat') or kk.startswith('z') or kk.startswith('bestov'))]
    cols_to_keep = ['RA','DEC',"Mstar_gal","Mstar_gal_err","Qz",'goodzflag']+magcols+zcols
    xmms = xmms.loc[selection, cols_to_keep]

    plt.figure()
    sc=plt.scatter(xmms.bestoverallz, xmms.absmag_R_VOICE, c=xmms.zphot, vmin=0, vmax=3, alpha=0.6, s=1)
    z_arr = np.linspace(0,1.4,1000)
    plt.plot(z_arr, absMag_distmod(z_arr, rband_voice_depth), label=r'VOICE Photometric Depth ($m_r = 26$)', color='k')
    plt.plot(z_arr, absMag_distmod(z_arr, rband_zp_depth), label=r'VOICE $z_{\rm phot}$  Depth ($m_r = $' + f'{rband_zp_depth:0.1f}' + r')',\
             linestyle='dashed', color='k')
    plt.colorbar(sc, label='Uncorrected Photo-z')#label=r'$m_r$')
    plt.ylim(-5,-25)
    plt.xlim(0,np.max(fitting_bins))
    plt.xlabel('Best Overall $z$')
    plt.ylabel(r'$M_r$')
    plt.legend(loc='best')
    plt.show()


    # ----------------------------------------------------------------- #
    zwin=0.2
    subvolranges = np.array([np.arange(0,1,zwin), np.arange(0+zwin,1+zwin,zwin)]).T 
    subvolranges_offset = subvolranges+zwin/2
    spws = np.array([[0,0.1], [0.2,0.5], [0.5,0.65]])
    subvolranges = np.vstack([spws, subvolranges, subvolranges_offset])
    print(subvolranges)

    log = open(direc['subvolpath']+"subvolume_metadata.csv",'w+')
    log.write('File\tN\tzmin\tzmax\tSA_str\tVolMpc3\tMrlim\n')
    for (zmin, zmax) in subvolranges:
        fname = "subvolume_{:0.2f}_to_{:0.2f}.csv".format(zmin,zmax)
        subvolume = xmms[(xmms.bestoverallz>zmin) & (xmms.bestoverallz<=zmax) & ~pd.isna(xmms.mag_R_VOICE)]
        ngal = len(subvolume)
        SA = solid_angle(subvolume.RA, subvolume.DEC, bins=700) / 3282.8 # converted from deg2 to steradians

        vol = integrate_volume(np.linspace(zmin,zmax,10000), SA, 70., 0.3, 0.7)
        Mrlim = absMag_distmod(zmax, rband_zp_depth)
        log.write(f"{fname}\t{int(ngal)}\t{zmin:0.2f}\t{zmax:0.2f}\t{SA:0.6E}\t{vol:0.6E}\t{Mrlim:0.2f}\n")
        subvolume.to_csv(direc["subvolpath"]+fname,index=False)
    log.close()
