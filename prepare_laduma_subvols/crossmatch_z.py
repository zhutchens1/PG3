import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord, match_coordinates_sky
import astropy.units as uu
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

output_file = './z22_w_laduma_specz.csv'
photpath = '/srv/one/zhutchen/paper3/prepare_laduma_subvols/xmmservs_joined.hdf5'
#zspecpath = '/srv/one/zhutchen/paper3/data/laduma/speczmerged-munira-zbestprior-zerr-columns-17June2025.csv'
laduma_data_path = '/srv/one/zhutchen/paper3/data/laduma/'
zspecpath = laduma_data_path+'LADUMA-specz-cat-allz-lsdr9phot.csv'
lowz_HIcat = laduma_data_path+'LADUMA_lowz_catalog_dr1.csv'
lowz_optcoords = laduma_data_path+'LADUMA_dr1_lowz_optical_coords.csv'
midz_HIcat = laduma_data_path+'LADUMA_midz_catalog_dr1.csv'
highz_HIcat = laduma_data_path+'LADUMA_HighZSPW_SourceList_29April25.csv'
threshold = 6 # arcsec
threshold_HI = 10 # arcsec
nominal_zspec_err = 100/3e5#0.005

if __name__=='__main__':
    # get dataframes
    xmms = pd.read_hdf(photpath)
    xmms.loc[:,'zphoterr'] = (xmms.pz_ulim - xmms.pz_llim)/2.
    speczcat = pd.read_csv(zspecpath).add_suffix('_zspec_cat')
    speczcat = speczcat[(speczcat.Star_zspec_cat==False) & (speczcat.Dud_zspec_cat==False) & (speczcat.Duplicate_zspec_cat==False) & (speczcat.ZERRNEW_zspec_cat>0)]

    # get closest match for each spec-z
    zspeccoords = SkyCoord(ra=speczcat.RA_zspec_cat.to_numpy()*uu.degree, dec=speczcat.DEC_zspec_cat.to_numpy()*uu.degree)
    xmmscoords = SkyCoord(ra=xmms.RA.to_numpy()*uu.degree, dec=xmms.DEC.to_numpy()*uu.degree)
    idx, sep2d, _ = match_coordinates_sky(xmmscoords, zspeccoords)
    matches = speczcat.iloc[idx].set_index(keys=xmms.index)
    assert (matches.index.to_numpy()==xmms.index.to_numpy()).all()

    plt.figure()
    z22pts = np.array([xmmscoords.ra.to('deg').value, xmmscoords.dec.to('deg').value]).T
    hull=ConvexHull(z22pts)
    plt.plot(zspeccoords.ra.to('deg').value, zspeccoords.dec.to('deg').value, 'rx', alpha=0.5, label='zspec cat')
    #plt.plot(z22pts[:,0], z22pts[:,1], 'ko', alpha=0.2, label='phot cat')
    for simplex in hull.simplices:
        plt.plot(z22pts[simplex,0], z22pts[simplex,1], 'k-')
    plt.plot(matches.RA_zspec_cat, matches.DEC_zspec_cat, 'b.')
    plt.xlabel('ra')
    plt.ylabel('dec')
    plt.legend(loc='best')
    plt.show() 

    # filter out bad matches and duplicate matches
    matches.loc[:,'sep'] = sep2d.to('arcsec').value
    matches = matches.sort_values(by='sep').drop_duplicates('ID_zspec_cat')
    matches = matches[matches.sep < threshold]
    assert (not matches.ID_zspec_cat.duplicated().any())
    xmms_w_spec = pd.concat([xmms, matches], axis=1).fillna(-999.)

    n_specz = len(xmms_w_spec[xmms_w_spec.ID_zspec_cat > -999])
    print(f'{n_specz} optical spec-z in crossmatch')

    # assemble zHI data from LADUMA 
    #lowz_spw = pd.read_csv(lowz_HIcat)[['name','RA_NEW','DEC_NEW','redshift', 'z_peak', 'err_z']]
    #midz_spw = pd.read_csv(midz_HIcat)[['name','ra_peak','dec_peak','redshift', 'z_peak', 'err_z']]
    #midz_spw.loc[:,'RA_NEW'] = midz_spw.loc[:,'ra_peak']
    #midz_spw.loc[:,'DEC_NEW'] = midz_spw.loc[:,'dec_peak']
    #ladumacat = pd.concat([lowz_spw, midz_spw], axis=0, ignore_index=True)
    #ladumacat.loc[:,'zHI_err_laduma'] = (ladumacat.err_z / ladumacat.z_peak) * ladumacat.redshift
    #ladumacat.rename({'RA_NEW':'ra_laduma', 'DEC_NEW':'dec_laduma', 'name':'laduma_name', 'redshift':'zHI_laduma'}, inplace=True, axis=1)
    #print("# HI galx: ",len(ladumacat))
    #assert len(ladumacat)==(len(lowz_spw)+len(midz_spw))

    # crossmatch to LADUMA
    #lcoords = SkyCoord(ra=ladumacat.ra_laduma.to_numpy()*uu.degree, dec=ladumacat.dec_laduma.to_numpy()*uu.degree)
    #idx, sep2d, _ = match_coordinates_sky(xmmscoords, lcoords)
    #lmatches = ladumacat.iloc[idx].set_index(keys=xmms.index)
    #assert (lmatches.index.to_numpy()==xmms.index.to_numpy()).all()
    #lmatches.loc[:,'sep'] = sep2d.to('arcsec').value
    #lmatches = lmatches.sort_values(by='sep').drop_duplicates('laduma_name')
    #lmatches = lmatches[lmatches.sep < threshold_HI]
    #assert (not lmatches.laduma_name.duplicated().any())
    #xmms_w_spec = pd.concat([xmms_w_spec, lmatches], axis=1).fillna(-999.)
    #n_zHI = len(xmms_w_spec[xmms_w_spec.zHI_laduma > 0])
    #print(f'{n_zHI} HI spec-z in crossmatch, {len(ladumacat)} in LADUMA source cats')
    #print('========================================================================')
    #print('LADUMA galaxies that did NOT crossmatch:')
    ##nonmatched_names = [ll for ll in ladumacat.laduma_name.to_list() if (ll not in lmatches.laduma_name.to_list())]
    #print(ladumacat.loc[[(ll not in lmatches.laduma_name.to_list()) for ll in ladumacat.laduma_name.to_list()],:].sort_values(by='laduma_name'))
    #print('========================================================================')

    # get zbest, zbesterr
    xmms_w_spec.loc[:,'zspecbest'] = np.zeros(len(xmms_w_spec))-999.
    xmms_w_spec.loc[:,'zspecbesterr'] = np.zeros(len(xmms_w_spec))-999.

    has_hi_spec = (xmms_w_spec.zHI_zspec_cat > 0)
    xmms_w_spec.loc[has_hi_spec,'zspecbest'] = xmms_w_spec.loc[has_hi_spec, 'zHI_zspec_cat']
    #xmms_w_spec.loc[has_hi_spec,'zspecbesterr'] = xmms_w_spec.loc[has_hi_spec, 'zHI_err_zspec_cat'] # col does not exist in Julia's file

    has_optical_spec = (xmms_w_spec.ZBESTNEW_zspec_cat > 0)
    xmms_w_spec.loc[has_optical_spec,'zspecbest'] = xmms_w_spec.loc[has_optical_spec, 'ZBESTNEW_zspec_cat']
    xmms_w_spec.loc[has_optical_spec,'zspecbesterr'] = xmms_w_spec.loc[has_optical_spec, 'ZERRNEW_zspec_cat']

    sel = (xmms_w_spec.zspecbesterr > -99) & (xmms_w_spec.zspecbesterr <= 0)
    xmms_w_spec.loc[sel, 'zspecbesterr'] = nominal_zspec_err

    goodzflag = (xmms_w_spec.Qz < 1) & (xmms_w_spec.zphot>0)
    xmms_w_spec.loc[:,'goodzflag'] = goodzflag
    
    # output
    xmms_w_spec.to_csv(output_file)

