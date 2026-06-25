from kdpfof import kdPFOF
import numpy as np
import pandas as pd
from pg3tools import *
from prob_giantonlyic import *
from astropy.cosmology import Planck18 as planck
import time

if __name__=='__main__':
    # usage example
    ls = pd.read_csv('/srv/two/zhutchen/paper3/data/lsdr9/ls_deg.csv')
    fofid=kdPFOF(ls.radeg, ls.dedeg, ls.zbest, ls.zbesterr, 0.07*3.5, 1.1*3.5, 0.05, planck, 10)
    #grpra,grpdec,grpz=prob_group_skycoords(ls.radeg, ls.dedeg, ls.zbest, ls.zbesterr, fofid)
    ti=time.time()
    prob_giantOnlyICRoutine(ls.radeg, ls.dedeg, ls.zbest, ls.zbesterr, fofid, lambda x: 1, lambda x: 500, 0.05, planck, 5)
    dt = time.time()-ti
    print('dt=',dt)

