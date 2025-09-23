import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import erf
from astropy.cosmology import LambdaCDM
cosmo = LambdaCDM(70.,0.3,0.7)

sys.path.insert(0,'/srv/one/zhutchen/paper3/codes/')
from photzcorrection import corrector

def gaussian_cdf(x, mu, sig):
    return 0.5*(1+erf((x-mu)/(1.4142135*sig)))

if __name__=='__main__':
    ls = pd.read_csv("/srv/one/hperk4/decals_ecovol.csv")
    matches = pd.read_csv("/srv/one/hperk4/decals_eco_resolve_allmatches.csv")
    ls = ls.drop(labels='absolute_rmag',axis=1) # remove absolute_rmag cols - hannah calculated incorrectly; I recalc below
    matches = matches.drop(labels='absolute_rmag',axis=1)
    h23 = pd.read_csv("/srv/one/zhutchen/g3groupfinder/resolve_and_eco/ECOdata_G3catalog_luminosity.csv").set_index('name')
    
    ecomatches = matches[(matches.name.str.startswith('ECO'))].set_index('name')
    cols_to_keep = ['ls_id','z_phot_median','z_phot_std','z_spec','mag_r']
    ix = [i for i in h23.index if (i in ecomatches.index)]
    ecomatches = ecomatches.loc[ix,cols_to_keep]
    eco_w_lsdata = pd.concat([h23, ecomatches], axis=1).fillna(-99.)
    print(eco_w_lsdata)

    # get data for lsdr9 objs not matched to eco galaxies
    ls = ls.set_index('ls_id')
    not_matched_to_eco = ~ls.index.isin(list(eco_w_lsdata.ls_id))
    ls = ls[not_matched_to_eco]
    ls.loc[:,'absolute_rmag'] = ls.mag_r + 5 - 5*np.log10(cosmo.luminosity_distance(np.array(ls.z_phot_median)).to('pc').value)

    # Ensure objects are within volume
    zmax = 7470/3e5
    zmin = 2530/3e5
    prob_thresh = 50/100 # at least this likely that obj in inside ECO vol
    p_inside_eco = gaussian_cdf(zmax, ls.z_phot_median.to_numpy(), ls.z_phot_std.to_numpy())-gaussian_cdf(zmin,\
         ls.z_phot_median.to_numpy(), ls.z_phot_std.to_numpy())
    sel = ((p_inside_eco > prob_thresh)) & (ls.absolute_rmag < -17.33)
    ls = ls[sel]

    # merge into a larger dataset
    combined = pd.concat([eco_w_lsdata.reset_index().set_index('ls_id'), ls],axis=0)
    combined.loc[:,'name'] = np.where(~pd.isna(combined.name.to_numpy()), combined.name.to_numpy(), np.full(len(combined),'ls_only'))
    combined.loc[:,'radeg'] = np.where(~pd.isna(combined.radeg.to_numpy()), combined.radeg.to_numpy(), combined.ra.to_numpy())
    combined.loc[:,'dedeg'] = np.where(~pd.isna(combined.dedeg.to_numpy()), combined.dedeg.to_numpy(), combined.dec.to_numpy())
    combined.loc[:,'rmag'] = np.where(~pd.isna(combined.rmag.to_numpy()), combined.rmag.to_numpy(), combined.mag_r.to_numpy())
    combined.loc[:,'absrmag'] = np.where(~pd.isna(combined.absrmag.to_numpy()), combined.absrmag.to_numpy(), combined.absolute_rmag.to_numpy())
    cz = np.where(~pd.isna(combined.cz.to_numpy()), combined.cz.to_numpy(), 3e5*combined.z_spec.to_numpy())
    cz[cz<0] = -99.
    combined.loc[:,'cz'] = cz
    combined.loc[:,'z_spec_merged'] = cz/3e5
    print(combined[['name','radeg','dedeg','z_phot_std','z_phot_median']].sort_values(by='radeg'))
    assert len(combined)==len(ls)+len(eco_w_lsdata)    
 
    # correct photo-z's
    ct = corrector(combined, 'z_phot_median', 'z_phot_std', 'z_spec_merged', 'mag_r', view_intermediate_plots=True, force_linear_fit2=True)
    zphotcorr, sigcorr, flag = ct.run() 
    combined.loc[:,'zphotcorr'] = zphotcorr
    combined.loc[:,'zphoterrcorr'] = sigcorr
    if ((combined.zphoterrcorr<0) & (combined.zphoterrcorr>-3)).any():
        print("WARNING: zphoterrcorr has negative non-999 values")
        print('code failed... exiting')
        exit()

    # output 
    combined.to_csv("full_lsdr9_w_eco.csv") 
