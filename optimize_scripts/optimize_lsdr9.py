import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.cosmology import LambdaCDM
import sys; sys.path.insert(0,'/srv/one/zhutchen/paper3/codes/')
import prob_g3groupfinder as pg3
from group_purity import get_metrics_by_group, get_metrics_by_halo

def run(eco, Pth, blos=1.1, savepath='./'):
    eco.loc[:,'Pth'] = eco.loc[:,'cz']*0 + Pth
    hubble_const = 70.
    omega_m = 0.3
    omega_de = 0.7
    cosmo=LambdaCDM(hubble_const, omega_m, omega_de)
    ecovolume = 191958.08 / (hubble_const/100.)**3.
    gfargseco = dict({'volume':ecovolume,'rproj_fit_multiplier':3,'vproj_fit_multiplier':4,'vproj_fit_offset':200,\
          'saveplotspdf':False, 'gd_rproj_fit_multiplier':2, 'gd_vproj_fit_multiplier':4, 'gd_vproj_fit_offset':100,\
          'gd_fit_bins':np.arange(-24,-19,0.25), 'gd_rproj_fit_guess':[1e-5, 0.4],\
          'pfof_Pth' : Pth, 'gd_vproj_fit_guess':[3e-5,4e-1], 'H0':hubble_const, 'Om0':omega_m, 'Ode0':omega_de,\
          'iterative_giant_only_groups': True, 'ncores' : None,\
          })
    #for kk in ['radeg','dedeg','simz','simzerr','absrmag','bestz','bestzerr','zphotcorr','zphoterrcorr']:
    #    print(kk, pd.isna(eco[kk]).any(), np.min(eco[kk]), np.max(eco[kk]))
    pg3ob=pg3.pg3(eco.radeg, eco.dedeg, 3e5*eco.simz, 3e5*eco.simzerr, eco.absrmag, -19.5,fof_bperp=0.07,fof_blos=blos,**gfargseco)
    pg3grp=pg3ob.find_groups()[0]
    eco.loc[:,'pg3grp'] = pg3grp

    g3grp = eco.loc[:,'g3grp_l'].to_numpy()
    pc_dataframe = eco.groupby('pg3grp').filter(lambda grp: (grp.g3grp_l>0).any())
    pp,cc = get_metrics_by_group(pc_dataframe.pg3grp, pc_dataframe.g3grp_l, pc_dataframe.absrmag.to_numpy(), enforce_positive_IDs=True)
    pc_dataframe.loc[:,'P_TR'] = pp
    pc_dataframe.loc[:,'C_TR'] = cc
    pp, cc = get_metrics_by_halo(pc_dataframe.pg3grp, pc_dataframe.g3grp_l, pc_dataframe.absrmag.to_numpy())
    pc_dataframe.loc[:,'P_RT'] = pp
    pc_dataframe.loc[:,'C_RT'] = cc
    pc_dataframe.loc[:,'grpn_TR'] = pg3.multiplicity_function(pc_dataframe.pg3grp.to_numpy(),return_by_galaxy=True)
    pc_dataframe.to_csv(savepath+f"eco_Pth{Pth}.csv")

# ========================================================== #

if __name__ == '__main__':
    folder_name = 'realistic_lsdr9/'
    savepath = '/srv/two/zhutchen/paper3/catalogs/'+folder_name
    if not os.path.exists(savepath):
        os.system(f"mkdir {savepath}")

    eco = pd.read_csv("/srv/one/zhutchen/paper3/data/desi/full_lsdr9_w_eco.csv").set_index('name')
    eco.loc[:,'g3grp_l'] = eco.loc[:,'g3grp_l'].fillna(-99.)

    zerrdata = pd.read_csv("/srv/two/zhutchen/ecocleanup/temp_eco_sdss_crossmatch/eco_newczforupload_102120.txt").set_index('name')
    idx = [ix for ix in eco.index if ix.startswith('ECO')]
    zerrdata = zerrdata.loc[idx,'newczerr']
    eco.loc[idx,'czerr'] = zerrdata
    czerr = eco.czerr.to_numpy()
    czerr[pd.isna(czerr)] = 100.
    eco.loc[:,'czerr'] = czerr
    eco.loc[:,'bestz'] = np.where((eco.z_spec_merged>0), eco.z_spec_merged, eco.zphotcorr)
    eco.loc[:,'bestzerr'] = np.where((eco.z_spec_merged>0), eco.czerr/3e5, eco.zphoterrcorr)
   
    # make simulated redshift data with 85-15 phot-spec mix within survey
    eco.loc[:,'simz'] = eco.loc[:,'bestz']
    eco.loc[:,'simzerr'] = eco.loc[:,'bestzerr']
    names_to_choose_from = eco.index[(eco.index.str.startswith('ECO')) & (eco.zphotcorr>0)].to_numpy()
    names_to_replace = np.random.choice(names_to_choose_from, replace=False, size=int(0.85*13861))
    eco.loc[names_to_replace,'simz'] = eco.loc[names_to_replace,'zphotcorr']
    eco.loc[names_to_replace,'simzerr'] = eco.loc[names_to_replace,'zphoterrcorr']
    eco.loc[eco[eco.simzerr<10/3e5].index,'simzerr'] = 10/3e5 # replace low spec-z errors with 10 km/s

    # test
    #eco.loc[:,'simzerr'] = 20/3e5

    Pth = [1e-4, 5e-3, 1e-3, 5e-2, 1e-2, 5e-2, 0.1, 0.15, 0.2, 0.25,  0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8]
    #, 0.9, 0.95, 0.99, 0.999]

    for pth_ in Pth:
        run(eco, pth_, savepath=savepath)
