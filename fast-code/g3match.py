import numpy as np
import pandas as pd
from pg3group import pg3groupfinder as pg3gf
import matplotlib.pyplot as plt
from pg3tools import *
import time

import sys
sys.path.insert(0,'/srv/one/zhutchen/g3groupfinder/optimized-code/')
from g3group import g3groupfinder as g3gf

def compareMultF(id1,id2):
    n1 = multiplicity_function(id1, False)
    n2 = multiplicity_function(id2, False)
    bins = np.arange(0.5,60.5,1)
    plt.figure()
    plt.hist(n1, label='1', bins=bins, histtype='step', linewidth=3)
    plt.hist(n2, label='2', bins=bins, histtype='step', hatch='//')
    plt.yscale('log')
    plt.legend(loc='upper right')
    plt.show()

if __name__=='__main__':
    pthresh = 0.99

    # test match with G3 group algorithm
    eco = pd.read_csv("/srv/one/zhutchen/g3groupfinder/resolve_and_eco/ECOdata_G3catalog_luminosity.csv")
    eco.loc[:,'z'] = eco.loc[:,'cz'] / 3e5
    eco.loc[:,'zerr'] = np.full(len(eco),1)/3e5

    # set up objects
    pg3 = pg3gf(eco.radeg, eco.dedeg, eco.z, eco.zerr, eco.absrmag, -19.5, precision=np.float64)
    g3 = g3gf(eco.radeg, eco.dedeg, eco.z, eco.absrmag, -19.5, precision=np.float64)

    # FOF
    pg3.giantOnlyPFOF(pthresh, 0.07, 1.1, 4.84)
    g3.giantOnlyFOF(0.07, 1.1, fof_sep=4.84)
    #compareMultF(pg3.g3grpid[pg3.giantsel], g3.g3grpid[g3.giantsel]) # <-- extremely good match for dv = 1 km/s and Pth=0.99

    # Giant-only merging
    # Quite a good match below it is somewhat dependent on kd tree filtering in prob_giantonlyic
    # Here candidate neighbors are defined as closeset in Cartesian redshift space based on the probabilistic
    # point estimates of the center of each group.
    pg3.deriveGiantCalibrations(rproj_fit_multiplier=3, vproj_fit_multiplier=4, vproj_fit_offset=200)
    g3.deriveGiantCalibrations(rproj_fit_multiplier=3, vproj_fit_multiplier=4, vproj_fit_offset=200)
    pg3.giantOnlyMerging(pthresh)
    g3.giantOnlyMerging()
    #compareMultF(pg3.g3grpid[pg3.giantsel], g3.g3grpid[g3.giantsel]) # result

    # Dwarf association - again a very good match, subject to kd tree concern (modest).
    pg3.dwarfAssoc(pthresh)
    g3.dwarfAssoc()
    #compareMultF(pg3.g3grpid, g3.g3grpid) # result

    # dwarf-only groups
    pg3.deriveDwarfBoundaries(gd_rproj_fit_multiplier=2, gd_vproj_fit_multiplier=4, gd_vproj_fit_offs=100, gd_fit_bins=np.arange(-24,-19.5,0.5))
    pg3.findDwarfOnlyGroups(pthresh)

    g3.deriveDwarfBoundaries(gd_rproj_fit_multiplier=2, gd_vproj_fit_multiplier=4, gd_vproj_fit_offs=100, gd_fit_bins=np.arange(-24,-19.5,0.5))
    g3.findDwarfOnlyGroups()


    cat = pg3.getCatalog()
    print(cat)
    #grpra,grpdec,grpz=prob_group_skycoords(eco[pg3.giantsel].radeg,eco[pg3.giantsel].dedeg,eco[pg3.giantsel].cz/3e5,zerr[pg3.giantsel],fofid)
    #grpn = multiplicity_function(fofid)

    #uniqID, NN = np.unique(fofid,return_counts=True)
    #for i,u in enumerate(uniqID):
    #    if NN[i]>=2:
    #        grpsel = (fofid == u)
    #        galz = eco[pg3.giantsel].cz.to_numpy()[grpsel] / 3e5
    #        zcen = grpz[grpsel][0]
    #        minz = galz.min()
    #        maxz = galz.max()
    #        correct = (minz < zcen) and (zcen < maxz)
    #        if not correct:
    #            print(galz)
    #            print(zcen)

    #exit()

    #fofid=kdPFOF(ls.radeg, ls.dedeg, ls.zbest, ls.zbesterr, 0.07*3.5, 1.1*3.5, 0.05, planck, 10)
    ##grpra,grpdec,grpz=prob_group_skycoords(ls.radeg, ls.dedeg, ls.zbest, ls.zbesterr, fofid)
    #ti=time.time()
    #prob_giantOnlyICRoutine(ls.radeg, ls.dedeg, ls.zbest, ls.zbesterr, fofid, lambda x: 1, lambda x: 500, 0.05, planck, 5)
    #dt = time.time()-ti
    #print('dt=',dt)

