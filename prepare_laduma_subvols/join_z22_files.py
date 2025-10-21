from astropy.io import fits
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from speclite import filters
import astropy.units as u
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d
import h5py
from pathos.multiprocessing import ProcessingPool
from tqdm import tqdm

if __name__=='__main__':
    home = '/srv/one/zhutchen/paper3/data/zou/' 
    photfile = home+'wcdfs_photcat.v1.fits'
    sedinfofile = home+'wcdfs.v1.fits'
    zfile = home+"zphot_WCDFS_ES1.fits"
    sedfile = home+'seds/wcdfs/models_gal_wcdfs.h5'

    # read-in files, convert to df
    photometry=fits.open(photfile)[1].data
    sedinfo=fits.open(sedinfofile)[1].data
    photz = fits.open(zfile)[1].data
    seds = h5py.File(sedfile, 'r')

    phot = pd.DataFrame().from_records(np.array(photometry).byteswap().newbyteorder())
    sedinfo = pd.DataFrame().from_records(np.array(sedinfo).byteswap().newbyteorder())
    photz = pd.DataFrame().from_records(np.array(photz).byteswap().newbyteorder())

    phot=phot.set_index(phot.Tractor_ID.to_numpy().astype(int)).sort_index()
    sedinfo=sedinfo.set_index(sedinfo.Tractor_ID.to_numpy().astype(int)).sort_index()
    photz=photz.set_index(photz.ID.to_numpy().astype(int)).sort_index()
    xmms = pd.concat([phot,sedinfo,photz],axis=1)
    xmms = xmms.loc[:,~xmms.columns.duplicated()].drop(labels='Tractor_ID',axis=1)
    xmms.index.name = 'Tractor_ID'
    print(len(phot), len(sedinfo), len(photz), len(xmms))
    xmms.to_hdf('xmmservs_joined.hdf5',key='df')

    # get absolute magnitudes in SDSS r band
    #ids_to_process = xmms.index.to_list()
    #worker = lambda i0: process_target(i0, xmms, seds)
    #print(worker(233255))
    #with ProcessingPool(60) as pool:
    #    Mr = list(tqdm(pool.imap(worker,ids_to_process),total=len(ids_to_process)))
    #Mr = np.zeros(len(ids_to_process))
    #for i,ix in tqdm(enumerate(ids_to_process),total=len(Mr)):
    #    Mr[i] = process_target(ix,xmms,seds)
    #xmms.loc[:,'absmag_R_VOICE'] = Mr
    #xmms.to_hdf('xmmservs_absrmag.hdf5',key='df')


#def get_abs_mag(tractorID, xmmservs_cat, xmmservs_seds):
#    try:
#        obsv_wavelength = np.array(xmmservs_seds[str(tractorID)]['wavelength'])*1000. # 1000 nm/micron
#        #zz =xmmservs_cat.loc[tractorID,'redshift']
#        zz =xmmservs_cat.loc[tractorID,'zphot']
#        rest_wavelength = obsv_wavelength / (1+zz)
#        obsv_specific_lum = np.array(xmmservs_seds[str(tractorID)]['L_lambda_total'])
#        rband = filters.load_filter('sdss2010-r')
#        rbandwavelength=np.array(rband.wavelength)*0.1 # 0.1 nm per AA
#        rbandresponse=np.array(rband.response)
#        rbandresponseinterp = interp1d(rbandwavelength,rbandresponse,bounds_error=False,fill_value=0)
#
#        plt.figure()
#        plt.plot(obsv_wavelength, obsv_specific_lum,'k',label='Observed-Frame Wavelength')
#        plt.plot(rest_wavelength, obsv_specific_lum,'r',label='Rest-Frame wavelength')
#        #plt.xscale('log')
#        plt.axvline(np.min(rbandwavelength))
#        plt.axvline(np.max(rbandwavelength))
#        plt.yscale('log')
#        plt.xlabel(r"$\lambda$ [nm]")
#        plt.ylabel(r"Specific Luminosity $L_\lambda$ [W/nm]")
#        #plt.ylim(1e33,0.2e35)
#        plt.xlim(400,900)
#        plt.legend(loc='best')
#        plt.show()
#
#        plt.figure()
#        plt.plot(rbandwavelength, rbandresponse)
#        plt.show()
#
#        Lsun = 3.846e26
#        W_per_ergss = 1e-7
#        cm_per_Mpc = 3.086e24
#        #dist_cm = (3e5*zz)/70. * cm_per_Mpc
#        dist_cm = 1e-5 * cm_per_Mpc
#        nm_per_AA = 0.1 # nm/AA
#        rbandzeropoint_flambda = 278e-11 # ergs/s/cm2/AA
#        rbandzeropoint_Llambda = rbandzeropoint_flambda * 4 * np.pi * dist_cm * dist_cm # ergs/s/AA
#        rbandzeropoint_Llambda = rbandzeropoint_Llambda * W_per_ergss # W/AA
#        rbandzeropoint_Llambda = rbandzeropoint_Llambda / nm_per_AA
#        zeropointC = 2.5*np.log10(trapezoid(rbandzeropoint_Llambda*rbandresponseinterp(rest_wavelength),rest_wavelength))
#
#        integrand = (obsv_specific_lum)*rbandresponseinterp(rest_wavelength)
#        Lr = trapezoid(integrand,rest_wavelength)
#        MM = (-2.5*np.log10(Lr)+zeropointC)
#        print(MM, zz, zeropointC)
#    except KeyError:
#        MM = -999
#        #print("KeyError Warning: failed on tractorID {} (SED not found and/or no phometry or redshift)".format(tractorID))
#    return MM
#
#def process_target(tractorID, xmmservs_cat, xmmservs_seds):
#    try:
#        x = xmmservs_cat.loc[tractorID,'mag_R_VOICE']
#    except ValueError:
#        assert False, "value error for tractorID {}".format(x)
#        
#    if tractorID==np.nan or tractorID=='nan' or pd.isna(tractorID):
#        return -999
#
#    if pd.isna(x) or x==0:
#        return -999
#    else:
#        return get_abs_mag(tractorID, xmmservs_cat, xmmservs_seds)
#
#
