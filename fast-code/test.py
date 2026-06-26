import numpy as np
import pandas as pd
from pg3group import pg3groupfinder as pg3gf
from pg3tools import *
import time

if __name__=='__main__':
    # usage example
    #ls = pd.read_csv('/srv/two/zhutchen/paper3/data/lsdr9/ls_deg.csv')
    #pg3 = pg3gf(ls.radeg, ls.dedeg, ls.zbest, ls.zbesterr, ls.absrmag,\
    #                     -19.5, precision=np.float64)

    eco = pd.read_csv("/srv/one/zhutchen/g3groupfinder/resolve_and_eco/ECOdata_G3catalog_luminosity.csv")
    zerr = np.full(len(eco),35)/3e5
    pg3 = pg3gf(eco.radeg, eco.dedeg, eco.cz/3e5, zerr, eco.absrmag,\
         -19.5, precision=np.float64)
    fofid=pg3.giantOnlyPFOF(0.9, 0.07, 1.1, 4.84)

    pg3.deriveGiantCalibrations(rproj_fit_multiplier=3, vproj_fit_multiplier=4, vproj_fit_offset=200)
    #pg3.plotGiantGroupBoundaries(show=True)
    pg3.giantOnlyMerging(0.9)

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

