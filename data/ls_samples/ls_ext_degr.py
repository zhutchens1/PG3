import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt 
from scipy.special import erf 
from scipy.stats import binned_statistic
from copy import deepcopy
from astropy.io import fits
from astropy.table import Table
from astropy.cosmology import LambdaCDM
from astropy.coordinates import SkyCoord
import astropy.units as uu
from catalog_merge import merge_catalogs_sky
cosmo = LambdaCDM(70.,0.3,0.7)

import sys
sys.path.insert(0,'/srv/one/zhutchen/paper3/codes/')
from photzcorrection import corrector

np.random.seed(1)

def gaussian_cdf(x, mu, sig):
    return 0.5*(1+erf((x-mu)/(1.4142135*sig)))

if __name__=='__main__':
    eco_file = '/srv/one/zhutchen/g3groupfinder/resolve_and_eco/ECOdata_G3catalog_luminosity.csv'
    res_file = '/srv/one/zhutchen/g3groupfinder/resolve_and_eco/RESOLVEdata_G3catalog_luminosity.csv'
    input_file = '/srv/two/zhutchen/paper3/data/lsdr9/ls_ext_degr_base.csv'
    ls_ext_file = '/srv/two/zhutchen/paper3/data/lsdr9/ls_ext.csv'
    ls_deg_file = '/srv/two/zhutchen/paper3/data/lsdr9/ls_deg.csv'

    # read in tables to process
    ls = pd.read_csv(input_file)
    eco = pd.read_csv(eco_file).set_index('name')
    resolve = pd.read_csv(res_file)
    resb = resolve[resolve.f_b==1].set_index('name')
    eco_resb = pd.concat([eco,resb],axis=0)

    c=3e5
    zmax_eco = eco.cz.max() / c
    zmin_eco = eco.cz.min() / c
    zmax_resb = resb.cz.max() / c
    zmin_resb = resb.cz.min() / c
    ramax_eco = eco.radeg.max()
    ramin_eco = eco.radeg.min()
    demax_eco = eco.dedeg.max()
    demin_eco = eco.dedeg.min()
    demax_resb = resb.dedeg.max()
    demin_resb = resb.dedeg.min()

    # crossmatch to RESOLVE/ECO to flag matches and insert RESOLVE/ECO data columns
    # first, need to drop duplicates from LS dataframe (this stems from overlapping tiles/bricks
    #   processed as part of get_ls_extdeg_base.py)
    cds = SkyCoord(ls.RA.values*uu.deg, ls.DEC.values*uu.deg)
    i, j, sep, _ = cds.search_around_sky(cds, 1*uu.arcsec)
    m = i < j # drop self-matches and mirrored pairs
    pairs = pd.DataFrame({'i': i[m], 'j': j[m], 'sep': sep[m].arcsec})
    if True:
        print(len(pairs), 'duplicate pairs')
        print(ls.loc[np.r_[pairs.i, pairs.j]].sort_values(['RA','DEC'])[['RA','DEC','OBJID_1','Z_PHOT_MEDIAN','rmag']])
    ls = ls.drop(index=ls.index[pairs.j.unique()]).reset_index(drop=True)
    ls_w_ecores = merge_catalogs_sky(eco_resb, ls, 'radeg', 'dedeg', 'RA', 'DEC', 5.0).copy()

    # now patch in values to unify columns for name, radeg, etc.
    ls_only_sel = (ls_w_ecores.name == -999)
    ls_w_ecores.loc[ls_only_sel,'name'] = 'ls_only'
    ls_w_ecores.loc[ls_only_sel,'radeg'] = ls_w_ecores.loc[ls_only_sel,'RA']
    ls_w_ecores.loc[ls_only_sel,'dedeg'] = ls_w_ecores.loc[ls_only_sel,'DEC']
    ls_w_ecores.loc[~ls_only_sel,'rmag'] = ls_w_ecores.loc[~ls_only_sel,'rmag_1']
    ls_w_ecores.loc[ls_only_sel,'rmag'] = ls_w_ecores.loc[ls_only_sel,'rmag_2']
    ls_w_ecores.loc[~ls_only_sel,'absrmag'] = ls_w_ecores.loc[~ls_only_sel,'absrmag_1']
    ls_w_ecores.loc[ls_only_sel,'absrmag'] = ls_w_ecores.loc[ls_only_sel,'absrmag_2']
    ls_w_ecores['zspec'] = np.where(ls_w_ecores.cz > 0, ls_w_ecores.cz/3e5, -999.)
    ls_w_ecores.loc[:,'zspecerr'] = np.where(ls_w_ecores.zspec>0, 35/3e5, -999) 
    
    in_eco = (ls_w_ecores.radeg>ramin_eco) & (ls_w_ecores.radeg<ramax_eco) & (ls_w_ecores.dedeg>demin_eco) & \
             (ls_w_ecores.dedeg<demax_eco)
    resb_dec_sel = (ls_w_ecores.dedeg>demin_resb) & (ls_w_ecores.dedeg<demax_resb)
    in_resb1 = (ls_w_ecores.radeg > 0) & (ls_w_ecores.radeg < 45) & resb_dec_sel
    in_resb2 = (ls_w_ecores.radeg > 330) & (ls_w_ecores.radeg < 360) & resb_dec_sel
    in_resb = in_resb1 | in_resb2 
             
    ls_w_ecores.loc[:,'footprint'] = np.where(in_eco, 'eco', np.where(in_resb, 'resb', 'none'))

    print(ls_w_ecores[['name','radeg','dedeg','cz','Z_PHOT_MEDIAN','rmag','absrmag','zspecerr','footprint']])
    n_matches=len(ls_w_ecores[(ls_w_ecores.name!='ls_only') & (ls_w_ecores.Z_PHOT_MEDIAN>0)])
    n_RESECO=len(ls_w_ecores[(ls_w_ecores.name!='ls_only')])
    print(f"{n_matches/n_RESECO} of RESOLVE/ECO have a match in LS DR9")

    no_xmatch_names = ls_w_ecores[ls_w_ecores.Z_PHOT_MEDIAN<0].name
    no_xmatch = eco_resb.loc[no_xmatch_names,:]
    print(no_xmatch[['radeg','dedeg','cz','absrmag','rmag']].describe())
    no_xmatch[['radeg','dedeg']].to_csv("./no_xmatch.csv",index=False)
    #print(eco_resb[eco_resb.index.isin(ls_w_ecores.name)])
    #print(eco_resb)
    #plt.figure()
    #plt.plot(no_xmatch.radeg, no_xmatch.dedeg, 'k.')
    #plt.show()
    #exit()
    # -------------------------------------------------------------- #
    # -------------------------------------------------------------- #
    # Construct the LS-Ext and LS-Degr Samples
    # -------------------------------------------------------------- #
    # -------------------------------------------------------------- #
    ls_ext = deepcopy(ls_w_ecores)
    ls_deg = deepcopy(ls_w_ecores)

    # RES/ECO galaxies without LS matches should be dropped explicitly
    # to simulate photometric incompleteness. Note this is distinct
    # from RES/ECO galaxies with LS matches but whose LS photo-z data
    # fail volume and/or luminosity requirements.
    no_ls = ~ls_deg.name.str.startswith('ls') & (ls_deg.Z_PHOT_MEDIAN < 0)
    print(f'dropping {no_ls.sum()} RESOLVE/ECO galaxies with no LS DR9 counterpart')
    ls_deg = ls_deg[~no_ls].reset_index(drop=True)

    # -------------------------------------------------------------- #
    # <<< construct the LS-extended sample
    
    ct = corrector(ls_ext, 'Z_PHOT_MEDIAN', 'Z_PHOT_STD', 'zspec', 'rmag',\
         view_intermediate_plots=True, force_linear_fit2=True, force_linear_fit1=True)
    zphotcorr, sigcorr, flag = ct.run()
    ls_ext.loc[:,'zphotcorr'] = zphotcorr
    ls_ext.loc[:,'zphoterrcorr'] = sigcorr
    if ((ls_ext.zphoterrcorr<0) & (ls_ext.zphoterrcorr>-3)).any():
        print("WARNING: zphoterrcorr has negative non-999 values")
        print('code failed... exiting')
        exit()

    ls_only = ls_ext.name.str.startswith('ls')
    zbest = np.where(ls_only, ls_ext.zphotcorr.values, ls_ext.cz.values/3e5)
    zbesterr = np.where(ls_only, ls_ext.zphoterrcorr.values, ls_ext.zspecerr.values)
    speczflag = np.where(ls_only, 0, 1)

    sel = (zbest == -99.) | (zbest==-999.) 
    assert not sel.any(), f'{sel.sum()} LS-only galaxies have sentinel zphotcorr'

    ls_ext.loc[:,'zbest'] = zbest
    ls_ext.loc[:,'zbesterr'] = zbesterr
    ls_ext.loc[:,'zbest_specflag'] = speczflag

    # calc Mrcorr
    old_5logd = 5*np.log10(cosmo.luminosity_distance(ls_ext.loc[ls_only,'Z_PHOT_MEDIAN'].to_numpy()).to('pc').value)
    new_5logd = 5*np.log10(cosmo.luminosity_distance(ls_ext.loc[ls_only,'zphotcorr'].to_numpy()).to('pc').value)
    ls_ext.loc[ls_only, 'absrmag'] = ls_ext.loc[ls_only,'absrmag'] + old_5logd - new_5logd
    assert np.isfinite(ls_ext.absrmag).all()

    # construct final sample
    prob_thresh_final = 15/100
    p_inside_eco = gaussian_cdf(zmax_eco, ls_ext.zbest.to_numpy(), ls_ext.zbesterr.to_numpy())-gaussian_cdf(zmin_eco,\
                                 ls_ext.zbest.to_numpy(), ls_ext.zbesterr.to_numpy())
    p_inside_resb = gaussian_cdf(zmax_resb, ls_ext.zbest.to_numpy(), ls_ext.zbesterr.to_numpy())-gaussian_cdf(zmin_resb,\
                                 ls_ext.zbest.to_numpy(), ls_ext.zbesterr.to_numpy())
    in_footprint_resb = (ls_ext.footprint=='resb')
    in_footprint_eco = (ls_ext.footprint=='eco')
    in_cz_resb = (p_inside_resb>prob_thresh_final)
    in_cz_eco = (p_inside_eco>prob_thresh_final)
    sel = (((in_footprint_resb & in_cz_resb) | (in_footprint_eco & in_cz_eco)) & (ls_ext.absrmag <= -17.0))
            #| (ls_ext.name.str.startswith('ECO')) |     (ls_ext.name.str.startswith('rf')) #<-- removed. catalogued galaxies with Mr <= -17.0 should not be in group-finding sample.
    ls_ext = ls_ext[sel]
    ls_ext.to_csv(ls_ext_file,index=False)
    
    # -------------------------------------------------------------- #
    # -------------------------------------------------------------- #
    # <<< construct LSdeg
    laduma = pd.read_csv("/srv/one/zhutchen/paper3/data/subvolumes/subvolume_0.00_to_0.10.csv",low_memory=False)
    Mrbins = np.arange(-30,-10,1)
    Mrbinc = 0.5*(Mrbins[:-1]+Mrbins[1:])
    laduma_specz = laduma[laduma.bestoverallzflag==1]
    count_spec,_,_ = binned_statistic(laduma_specz.absmag_R_VOICE, laduma_specz.absmag_R_VOICE, 'count', bins=Mrbins)
    count_all,_,_ = binned_statistic(laduma.absmag_R_VOICE, laduma.absmag_R_VOICE, 'count', bins=Mrbins)
    speczfrac = count_spec/count_all
    print(f'global LADUMA spec-z fraction: {np.sum(count_spec)/np.sum(count_all)}')
    del laduma, laduma_specz
    plt.figure()
    plt.plot(Mrbinc, speczfrac)
    plt.gca().invert_xaxis()
    plt.show()

    # replace RESOLVE-B/ECO cz with zphot according to this completeness
    ls_only = ls_deg.name.str.startswith('ls')
    ls_deg.loc[:,'zsim'] = np.where(ls_only, ls_deg.Z_PHOT_MEDIAN.values, ls_deg.cz.values/3e5)
    ls_deg.loc[:,'zsimerr']  = np.where(ls_only, ls_deg.Z_PHOT_STD.values, ls_deg.zspecerr.values)
    ls_deg.loc[:,'zsimflag'] = 1-ls_only.astype(int)
    ls_deg.loc[:,'Mrbin_id'] = np.digitize(ls_deg.absrmag.to_numpy(), Mrbins)
    ls_deg.loc[ls_deg.name.str.startswith('ls'),'Mrbin_id'] = -99 # not going to mess with these.
    for idv in ls_deg.Mrbin_id.unique():
        if idv>=0:
            sel = np.logical_not(ls_deg.name.str.startswith('ls')) & (ls_deg.Mrbin_id==idv)
            z_spec_comp = speczfrac[np.argmin(np.abs(Mrbinc-np.median(ls_deg.loc[sel,'absrmag'])))]
            prob = np.random.choice([1,0], p=[z_spec_comp, 1-z_spec_comp], size=np.sum(sel)).astype(bool)
            ls_deg.loc[sel,'zsim'] = np.where(prob, ls_deg.loc[sel,'cz']/3e5, ls_deg.loc[sel,'Z_PHOT_MEDIAN'])
            ls_deg.loc[sel,'zsimerr'] = np.where(prob, ls_deg.loc[sel,'zspecerr'], ls_deg.loc[sel,'Z_PHOT_STD'])
            ls_deg.loc[sel,'zsimflag'] = prob.astype(np.float64)#np.where(prob, np.ones_like(prob), np.zeros_like(prob)).astype(np.float64) # 1 if specz, 0=photz

    zspec_sim = deepcopy(ls_deg.loc[:,'zsim'].to_numpy())
    zspec_sim[(ls_deg.zsimflag.values==0)]=-99.
    ls_deg.loc[:,'zspec_sim'] = zspec_sim


    # zphot corrections
    ct = corrector(ls_deg, 'Z_PHOT_MEDIAN', 'Z_PHOT_STD', 'zspec_sim', 'rmag',\
         view_intermediate_plots=True, force_linear_fit2=True, force_linear_fit1=True)
    zphotcorr, sigcorr, flag = ct.run()
    ls_deg.loc[:,'zphotcorr'] = zphotcorr
    ls_deg.loc[:,'zphoterrcorr'] = sigcorr
    if ((ls_deg.zphoterrcorr<0) & (ls_deg.zphoterrcorr>-3)).any():
        print("WARNING: zphoterrcorr has negative non-999 values")
        print('code failed... exiting')
        exit()

    ls_only = ls_deg.name.str.startswith('ls')
    photsel = (ls_deg.zsimflag==0)
    ls_deg.loc[photsel,'zbest'] = ls_deg.loc[photsel,'zphotcorr']
    ls_deg.loc[photsel,'zbesterr'] = ls_deg.loc[photsel,'zphoterrcorr']
    ls_deg.loc[photsel,'zbest_specflag'] = 0
    ls_deg.loc[~photsel,'zbest'] = ls_deg.loc[~photsel,'cz']/3e5
    ls_deg.loc[~photsel,'zbesterr'] = ls_deg.loc[~photsel,'zspecerr']
    ls_deg.loc[~photsel,'zbest_specflag'] = 1

    sel = (ls_deg.zbest == -99.) | (ls_deg.zbest==-999.) # correct for any objects without measured photo-z
    assert not sel.any(), f'{sel.sum()} LS-deg galaxies have sentinel zphotcorr'
    assert (ls_deg.zbest > 0).all()

    # calc Mrcorr
    # two cases here: zphot->zphotcorr (LS), cz->zphotcorr(RES/ECO) [~photsel cases do not need correction; cz->cz]
    ls_deg.loc[:,'absrmag_uncorr'] = deepcopy(ls_deg.loc[:,'absrmag'])
    ls_deg.loc[:,'ls_only'] = ls_deg.name.str.startswith('ls')
    case1 = (ls_deg.ls_only) & (photsel)
    old_5logd = 5*np.log10(cosmo.luminosity_distance(ls_deg.loc[case1,'Z_PHOT_MEDIAN'].to_numpy()).to('pc').value)
    new_5logd = 5*np.log10(cosmo.luminosity_distance(ls_deg.loc[case1,'zphotcorr'].to_numpy()).to('pc').value)
    ls_deg.loc[case1, 'absrmag'] = ls_deg.loc[case1,'absrmag'] + old_5logd - new_5logd

    case2 = (~ls_deg.ls_only) & (photsel)
    old_5logd = 5*np.log10(cosmo.luminosity_distance(ls_deg.loc[case2,'cz'].to_numpy()/3e5).to('pc').value)
    new_5logd = 5*np.log10(cosmo.luminosity_distance(ls_deg.loc[case2,'zphotcorr'].to_numpy()).to('pc').value)
    ls_deg.loc[case2, 'absrmag'] = ls_deg.loc[case2,'absrmag'] + old_5logd - new_5logd
    ls_deg.drop(labels=['absrmag_1','absrmag_2'],axis=1,inplace=True)
    assert np.isfinite(ls_deg.absrmag).all()

    plt.figure()
    plt.scatter(ls_deg[case2].absrmag_uncorr, ls_deg[case2].absrmag,s=1)
    plt.xlabel('uncorrected Mr')
    plt.ylabel('corrected Mr')
    plt.show()

    # define final sample
    prob_thresh_final = 15/100
    in_footprint_resb = (ls_deg.footprint=='resb')
    in_footprint_eco = (ls_deg.footprint=='eco')
    p_inside_eco = gaussian_cdf(zmax_eco, ls_deg.zbest.to_numpy(), ls_deg.zbesterr.to_numpy())-gaussian_cdf(zmin_eco,\
         ls_deg.zbest.to_numpy(), ls_deg.zbesterr.to_numpy())
    p_inside_resb = gaussian_cdf(zmax_resb, ls_deg.zbest.to_numpy(), ls_deg.zbesterr.to_numpy())-gaussian_cdf(zmin_resb,\
         ls_deg.zbest.to_numpy(), ls_deg.zbesterr.to_numpy())
    in_cz_resb = (p_inside_resb>prob_thresh_final)
    in_cz_eco = (p_inside_eco>prob_thresh_final)
    sel = ((in_footprint_resb & in_cz_resb & (ls_deg.absrmag<=-17.0)) | (in_footprint_eco & in_cz_eco & (ls_deg.absrmag<=-17.33))) #| (ls_deg.name.str.startswith('ECO')) | (ls_deg.name.str.startswith('rf'))
    final_cols = ['name','radeg','dedeg','footprint','r50'] + [kk for kk in ls_deg.keys() if ((('Z' in kk) or ('z' in kk) or ('mag' in kk) or ('g3' in kk)) and ('aster' not in kk))]
    ls_deg = ls_deg.loc[sel, final_cols]
    ls_deg.to_csv(ls_deg_file, index=False)

    
