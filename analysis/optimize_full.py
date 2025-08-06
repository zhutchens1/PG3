import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys; sys.path.insert(0,'/srv/one/zhutchen/paper3/codes/')
import foftools as fof
import prob_g3groupfinder as pg3
from group_purity import get_metrics_by_group, get_metrics_by_halo
from itertools import product
from copy import deepcopy

basepath = "/srv/two/zhutchen/paper3/catalogs_all_specz/"

def random_search(nn, ranges):
    samples = np.array([np.random.uniform(low=ranges[kk][0], high=ranges[kk][1], size=nn) for kk in ranges.keys()])
    return samples.T

def get_gf_results(savenumber, sigmacz, Pth, rproj_fit_multiplier, vproj_fit_multiplier, vproj_fit_offset, gd_rproj_fit_multiplier, gd_vproj_fit_multiplier, gd_vproj_fit_offset,
         ecofile="/srv/one/zhutchen/g3groupfinder/resolve_and_eco/ECOdata_G3catalog_luminosity.csv"):
    hubble_const=70.
    omega_m=0.3
    omega_de=0.7
    ecovolume = 192351 / ((hubble_const)/100)**3.
    gfargseco = dict({'volume':ecovolume,'rproj_fit_multiplier':rproj_fit_multiplier,'vproj_fit_multiplier':vproj_fit_multiplier,'vproj_fit_offset':vproj_fit_offset,\
           'gd_rproj_fit_multiplier':gd_rproj_fit_multiplier, 'gd_vproj_fit_multiplier':gd_vproj_fit_multiplier, 'gd_vproj_fit_offset':gd_vproj_fit_offset,\
           'pfof_Pth' : Pth,\
           'gd_fit_bins':np.arange(-24,-19,0.25), 'gd_rproj_fit_guess':[1e-5, 0.4],\
           'gd_vproj_fit_guess':[3e-5,4e-1], 'H0':hubble_const, 'Om0':omega_m, 'Ode0':omega_de,  'iterative_giant_only_groups':True,\
           'showplots':False,'saveplotspdf':False,\
    })
    
    eco = pd.read_csv(ecofile).set_index('name')
    eco = eco[eco.absrmag<=-17.33]
    eco.loc[:,'czerr'] = eco.cz * 0 + sigmacz 
    pg3out = pg3.prob_g3groupfinder_luminosity(eco.radeg, eco.dedeg, eco.cz, eco.czerr, eco.absrmag,-19.5,fof_blos=1.1,fof_bperp=0.07,**gfargseco)
    grpid = pg3out[0]
    eco.loc[:,'pg3grp']=grpid
    h23grpid = eco.g3grp_l.to_numpy()
    pp,cc = get_metrics_by_group(grpid, h23grpid, eco.absrmag.to_numpy())
    eco.loc[:,'P_TR'] = pp
    eco.loc[:,'C_TR'] = cc
    eco.loc[:,'ptimesc_TR'] = pp*cc
    pp, cc = get_metrics_by_halo(grpid, h23grpid, eco.absrmag.to_numpy())
    eco.loc[:,'P_RT'] = pp
    eco.loc[:,'C_RT'] = cc
    eco.loc[:,'ptimesc_RT'] = pp*cc
    eco.loc[:,'grpn_TR'] = fof.multiplicity_function(grpid,return_by_galaxy=True)
    outkeys = ['radeg','dedeg','cz','czerr','absrmag','pg3grp','P_TR','C_TR','P_RT','C_RT','ptimesc_TR','ptimesc_RT','grpn_TR','g3grp_l','g3logmh_l']
    eco[outkeys].to_csv(basepath+f"ECOgroupcat_{savenumber}.csv", index=True)

if __name__=='__main__':
    n_samples = 1200
    savenumbers = np.arange(n_samples)
    param_ranges = dict({
        'Pth_values' : [0.5, 0.99],
        'rproj_values' : [1,5],
        'vproj_values' : [1,5],
        'offset_values' : [0,500],
        'gdrproj_values' : [1,5],
        'gdvproj_values' : [1,5],
        'gdoffset_values' : [0,500],
    })
    sigma_cz_values = [35,35*2,35*3,35*4]
    sigmacz = np.concatenate([np.full(n_samples//len(sigma_cz_values), sigma_cz_values[ii]) for ii in range(0,len(sigma_cz_values))])
    combos = np.hstack([savenumbers[:,np.newaxis], sigmacz[:,np.newaxis], random_search(n_samples, param_ranges)])
    np.savetxt(basepath+f'parameter_combinations.csv', combos, delimiter=' ', fmt='%1.5e')

    from pathos.multiprocessing import ProcessingPool
    pool = ProcessingPool(40)
    worker = lambda array: get_gf_results(*array)
    pool.map(worker, combos)
