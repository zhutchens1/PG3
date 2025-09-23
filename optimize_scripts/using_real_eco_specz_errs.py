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
    pg3ob=pg3.pg3(eco.radeg, eco.dedeg, eco.cz, eco.czerr, eco.absrmag, -19.5,fof_bperp=0.07,fof_blos=blos,**gfargseco)
    pg3grp=pg3ob.find_groups()[0]
    eco.loc[:,'pg3grp'] = pg3grp

    g3grp = eco.loc[:,'g3grp_l'].to_numpy()
    pp,cc = get_metrics_by_group(pg3grp, g3grp, eco.absrmag.to_numpy())
    eco.loc[:,'P_TR'] = pp
    eco.loc[:,'C_TR'] = cc
    pp, cc = get_metrics_by_halo(pg3grp, g3grp, eco.absrmag.to_numpy())
    eco.loc[:,'P_RT'] = pp
    eco.loc[:,'C_RT'] = cc
    eco.loc[:,'grpn_TR'] = pg3.multiplicity_function(pg3grp,return_by_galaxy=True)
    eco.to_csv(savepath+f"eco_Pth{Pth}.csv")

if __name__ == '__main__':
    folder_name = 'real_eco_speczerrs/'
    savepath = '/srv/two/zhutchen/paper3/catalogs/'+folder_name
    if not os.path.exists(savepath):
        os.system(f"mkdir {savepath}")

    eco = pd.read_csv("/srv/one/zhutchen/g3groupfinder/resolve_and_eco/ECOdata_G3catalog_luminosity.csv").set_index('name')
    eco.loc[:,'grpn_RT'] = pg3.multiplicity_function(eco.g3grp_l.to_numpy(),return_by_galaxy=True)

    zerrdata = pd.read_csv("/srv/two/zhutchen/ecocleanup/temp_eco_sdss_crossmatch/eco_newczforupload_102120.txt").set_index('name')
    zerrdata = zerrdata.loc[eco.index,'newczerr']
    eco.loc[:,'czerr'] = zerrdata

    Pth = [1e-2, 0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999]
    for pth_ in Pth:
        run(eco, pth_, savepath=savepath)
