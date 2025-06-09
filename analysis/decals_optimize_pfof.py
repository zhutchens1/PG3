import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys; sys.path.insert(0,'/srv/one/zhutchen/paper3/codes/')
import foftools as fof
import prob_g3groupfinder as pg3
from group_purity import get_metrics_by_group, get_metrics_by_halo
from scipy.stats import binned_statistic, percentileofscore as pos
from copy import deepcopy
import pickle
from pathos.multiprocessing import ProcessingPool

from matplotlib.ticker import MaxNLocator
from matplotlib import rcParams
from matplotlib.colors import LogNorm
rcParams['axes.labelsize'] = 9
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['legend.fontsize'] = 9
rcParams['font.family'] = 'sans-serif'
#rcParams['font.sans-serif'] = ['Helvetica']
#rcParams['text.usetex'] = True
rcParams['grid.color'] = 'k'
rcParams['grid.linewidth'] = 0.2
my_locator = MaxNLocator(6)
singlecolsize = (3.3522420091324205, 2.0717995001590714)
doublecolsize = (7.100005949910059, 4.3880449973709)

def get_groups(eco, sigmacz, Pth, blos):
    eco.loc[:,'czerr'] = eco.cz * 0 + sigmacz 
    vol = 192351 / (0.7**3.)
    sep = (vol/len(eco))**(1/3.)
    grpid = pg3.pfof_comoving(eco.radeg.to_numpy(),eco.dedeg.to_numpy(),eco.cz.to_numpy(),eco.czerr.to_numpy(), 0.07*sep, blos*sep, Pth)
    eco.loc[:,'pfofid']=grpid
    fofid = eco.fofid.to_numpy()
    pp,cc = get_metrics_by_group(grpid, fofid, eco.absrmag.to_numpy())
    eco.loc[:,'P_TR'] = pp
    eco.loc[:,'C_TR'] = cc
    pp, cc = get_metrics_by_halo(grpid, fofid, eco.absrmag.to_numpy())
    eco.loc[:,'P_RT'] = pp
    eco.loc[:,'C_RT'] = cc
    eco.loc[:,'grpn_TR'] = fof.multiplicity_function(grpid,return_by_galaxy=True)
    df_TR = eco.groupby('pfofid').first()
    df_RT = eco.groupby('fofid').first()
    df2_TR = df_TR[df_TR.grpn_TR>1]
    df2_RT = df_RT[df_RT.grpn_RT>1]
    if len(df2_TR)==0 or len(df2_RT)==0:
        print(len(df2_TR), len(df2_RT))
        print(sigmacz)
    #output = [df_TR.P_TR.mean(), df_TR.C_TR.mean(), df_RT.P_RT.mean(), df_RT.C_RT.mean(),\
    #          df2_TR.P_TR.mean(), df2_TR.C_TR.mean(), df2_RT.P_RT.mean(), df2_RT.C_RT.mean()]
    output = [len(df_TR.P_TR[df_TR.P_TR>0.5])/len(df_TR), len(df_TR.C_TR[df_TR.C_TR>0.5])/len(df_TR), len(df_RT.P_RT[df_RT.P_RT>0.5])/len(df_RT), len(df_RT.C_RT[df_RT.C_RT>0.5])/len(df_RT)] 
    #len(df2_TR.P_TR[df2_TR.P_TR>0.5])/len(df2_TR), len(df2_TR.C_TR[df2_TR.C_TR>0.5])/len(df2_TR), len(df2_RT.P_RT[df2_RT.P_RT>0.5])/len(df2_RT), len(df2_RT.C_RT[df2_RT.C_RT>0.5])/len(df2_RT)]
    return output

if __name__=='__main__':
    eco = pd.read_csv("/srv/one/zhutchen/g3groupfinder/resolve_and_eco/ECOdata_G3catalog_luminosity.csv").set_index('name')
    eco = eco[eco.absrmag<=-19.5]

    vol = 192351 / (0.7**3.)
    sep = (vol/len(eco))**(1/3.)
    eco.loc[:,'fofid'] = fof.fast_fof(eco.radeg.to_numpy(), eco.dedeg.to_numpy(), eco.cz.to_numpy(), 0.07, 1.1, sep)
    eco.loc[:,'grpn_RT'] = fof.multiplicity_function(eco.fofid.to_numpy(),return_by_galaxy=True)

    pzdata = pd.read_csv("/srv/one/hperk4/eco_resb_decals_photoz.csv")['e_tab_corr'].to_numpy()
    plt.figure()
    plt.hist(pzdata, bins='fd')
    plt.yscale('log')
    plt.show()

    bpara=np.arange(0.4,1.4,0.1)
    #Pth = [1e-2, 0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9]
    #Pth = [1e-4, 1e-3, 1e-2, 5e-2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    Pth = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.2, 0.3, 0.4]
    sigmaz = np.percentile(pzdata, q=[1,5,16,50,84,95,99])#q=[5,15,30,50,70,85,95]))
    print(sigmaz)
    params=[]
    outputs=[]
    for bb_ in bpara:
        for pp in Pth:
            for sz in sigmaz:
                params.append((sz,pp,bb_))
                #outputs.append(get_groups(eco, sz, pp, bb_))
    parallel_fn = lambda args: get_groups(eco,*args)
    pool = ProcessingPool(30)
    outputs=pool.map(parallel_fn, params)

    print(outputs)
    result = {'params':params, 'outputs':outputs}
    pickle.dump(result, open('decals_pfof_optimize_result.pkl','wb'))


    #print(get_groups(eco, 100, 0.01, 1.1)) 

