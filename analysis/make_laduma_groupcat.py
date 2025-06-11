import sys
sys.path.insert(0,'../codes/')
import prob_g3groupfinder as pg3
import foftools as fof
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle

zedges = [0.0,0.2,0.4,0.6,0.8,1.0]
subvoldir = lambda zmin, zmax: f'/srv/one/zhutchen/paper3/data/subvolumes/subvolume_{zmin}_to_{zmax}.hdf5'
volumepath = '/srv/one/zhutchen/paper3/data/subvolumes/volumes_h0.7.pkl'
comppath = '/srv/one/zhutchen/paper3/data/subvolumes/completeness_h0.7.pkl'

hubble_const = 70.
omega_m = 0.3
omega_de = 0.7
speed_of_light = 3e5
pg3_params = {'rproj_fit_multiplier':3,'vproj_fit_multiplier':4,'vproj_fit_offset':200,'saveplotspdf':False,
           'gd_rproj_fit_multiplier':2, 'gd_vproj_fit_multiplier':4, 'gd_vproj_fit_offset':100,\
           'gd_fit_bins':np.arange(-24,-19,0.25), 'gd_rproj_fit_guess':[1e-5, 0.4],\
           'pfof_Pth' : 0.5, 'gd_vproj_fit_guess':[3e-5,4e-1], 'H0':hubble_const, 'Om0':omega_m, 'Ode0':omega_de,
           'iterative_giant_only_groups':True, 'dwarfgiantdivide':-19.5}
nominalzerr = 100/speed_of_light

# ================================================================================= #

ecolimit = -17.33
comps = pickle.load(open(comppath,'rb'))
volumes = pickle.load(open(volumepath,'rb'))

for ii, z_i in enumerate(zedges):
    fnamepath = subvoldir(zedges[ii], zedges[ii+1])
    fname = fnamepath.split('/')[-1]
    floor = (comps[fname])# if (comps[fname]<ecolimit) else ecolimit)
    zmin = (z_i if (z_i!=0.0) else 0.005)
    zmax = (zedges[ii+1])

    df = pd.read_hdf(fnamepath)
    df = df[(df.absrmag<floor) & (df.bestoverallredshift>zmin) & (df.bestoverallredshift<zmax)]
    df.loc[:,'bestoverallredshifterr'] = df.loc[:,'bestoverallredshifterr'].where((df.bestoverallredshifterr>0), nominalzerr)
    pg3out=pg3.prob_g3groupfinder_luminosity(df.RAall, df.DECall, speed_of_light*df.bestoverallredshift, speed_of_light*df.bestoverallredshifterr,\
                 df.absrmag,volume=volumes[fname], summary_page_savepath=fname.replace('.hdf5','.summary.pdf'), **pg3_params)
    df.loc[:,'pg3grp'] = pg3out[0]
    df.loc[:,'pg3grpn'] = fof.multiplicity_function(pg3out[0], return_by_galaxy=True)
    print('getting group centers...')
    grpra, grpdec, grpz, _ = pg3.prob_group_skycoords(df.RAall.to_numpy(), df.DECall.to_numpy(), df.bestoverallredshift.to_numpy(),\
                            df.bestoverallredshifterr.to_numpy(), df.pg3grp.to_numpy(), False)
    df.loc[:,'pg3grpradeg'] = grpra
    df.loc[:,'pg3grpdedeg'] = grpdec
    df.loc[:,'pg3grpz'] = grpz
    df.to_hdf(fname.replace('.hdf5','_pg3.hdf5'), 'group-cat')
