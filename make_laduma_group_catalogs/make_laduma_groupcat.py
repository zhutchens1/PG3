import sys
sys.path.insert(0,'../codes/')
import prob_g3groupfinder as pg3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.cosmology import LambdaCDM
import os
import pickle

subvoldir = '/srv/one/zhutchen/paper3/data/subvolumes/'
metapath = subvoldir+'subvolume_metadata.csv'

hubble_const = 70.
omega_m = 0.3
omega_de = 0.7
speed_of_light = 3e5
pg3_params = {'rproj_fit_multiplier':3,'vproj_fit_multiplier':4,'vproj_fit_offset':200,'saveplotspdf':False,
           'gd_rproj_fit_multiplier':2, 'gd_vproj_fit_multiplier':4, 'gd_vproj_fit_offset':100,\
           'gd_fit_bins':np.arange(-24,-19,0.25), 'gd_rproj_fit_guess':[1e-5, 0.4],\
           'pfof_Pth' : 0.2, 'gd_vproj_fit_guess':[3e-5,4e-1], 'H0':hubble_const, 'Om0':omega_m, 'Ode0':omega_de,
           'iterative_giant_only_groups':True, 'dwarfgiantdivide':-19.5, 'ncores':None}
cosmo = LambdaCDM(pg3_params['H0'], pg3_params['Om0'], pg3_params['Ode0'])

# ================================================================================= #
meta = pd.read_csv(metapath, sep='\t').set_index('File')
ecolimit = -17.33

print(os.listdir(subvoldir))
for fname in os.listdir(subvoldir):
    if ('meta' not in fname) and fname.endswith('.csv'):
        fnamepath = subvoldir + fname
        print(f'###### working on {fnamepath} #######')
        Mrlim = meta.loc[fname,'Mrlim']
        print(ecolimit, Mrlim)
        floor = (Mrlim if (Mrlim<ecolimit) else ecolimit)
        volume = meta.loc[fname,'VolMpc3']

        # Read file and perform group-finding
        df = pd.read_csv(fnamepath)
        df = df[(df.absmag_R_VOICE<floor) & (df.absmag_R_VOICE > -99) & (df.bestoverallz>0.005) & (df.bestoverallzerr<10)] # this last one gets rid of odd z_err~220 obj at z=0.6-0.8
        sel = (df.bestoverallzerr > 10)
        pg3obj=pg3.pg3(df.RA, df.DEC, speed_of_light*df.bestoverallz, speed_of_light*df.bestoverallzerr,\
                     df.absmag_R_VOICE,volume=volume, summary_page_savepath=fname.replace('.csv','.summary.pdf'), **pg3_params)
        pg3out = pg3obj.find_groups()

        # Record group properties
        df.loc[:,'pg3grp'] = pg3out[0]
        df.loc[:,'pg3ssid'] = pg3out[1]
        df.loc[:,'pg3grpn'] = pg3obj.get_grpn(return_by_galaxy=True)
        grpra, grpdec, grpz, grpz16, grpz84, _  = pg3obj.get_group_centers(return_z_pdfs=False)
        df.loc[:,'pg3grpradeg'] = grpra
        df.loc[:,'pg3grpdedeg'] = grpdec
        df.loc[:,'pg3grpz'] = grpz
        df.loc[:,'pg3grpz16'] = grpz16
        df.loc[:,'pg3grpz84'] = grpz84
        speczfrac, largestdz, largestonskydist = pg3.group_z_demographics(df.RA, df.DEC, df.bestoverallz, df.bestoverallzflag, pg3out[0], cosmo)
        df.loc[:,'pg3speczfrac'] = speczfrac
        df.loc[:,'pg3largestdzspec'] = largestdz
        df.loc[:,'pg3largestonskydist'] = largestonskydist
        df.loc[:,'pg3grprproj'] = pg3.get_grprproj_e17(df.RA, df.DEC, df.bestoverallz, pg3out[0], df.pg3grpradeg, df.pg3grpdedeg, df.pg3grpz, cosmo)
        df.loc[:,'pg3grpfwqmradius'] = pg3.fwqm_relative_radius_laduma(df.pg3grpradeg, df.pg3grpdedeg, df.pg3grpz)
        df.loc[:,'pg3grpabsrmag'] = pg3.get_int_mag(df.absmag_R_VOICE.to_numpy(), pg3out[0])
        df.loc[:,'pg3grplogmstar'] = pg3.get_int_mass(np.log10(df.Mstar_gal), pg3out[0])
        df.loc[:,'pg3central'] = pg3.get_central_flag(df.absmag_R_VOICE.to_numpy(), pg3out[0])
        df.to_csv(fname.replace('.csv','_pg3.csv'),index=False)
