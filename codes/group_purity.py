import numpy as np
from scipy.stats import mode

def get_metrics_by_group(groupid, haloid, galproperty):
    """
    For a group catalog constructed from a mock galaxy catalog,
    compute galaxy-wise purity and completeness metrics using
    true halo IDs for comparison. This function computes metrics
    on a group-by-group basis.

    Parameters
    ---------------------------
    groupid : iterable
        Group ID numbers after applying group-finding algorithm, length = # galaxies.
    haloid : iterable
        Halo ID numbers extracted from mock catalog halos, length = # galaxies = len(groupid).
    galproperty : iterable
        Group property by which to determine the central galaxy in the group. If all values are
        >-15 and <-27, then galproperty is assumed to be a magnitude, and the central will be the
        brightest galaxy. If all values are >0, this value is assumed to be mass, and the central
        will be selected by the maximum.

    Returns
    ---------------------------
    Suppose we map groups to halos, and define
        - N_g as the number of galaxies in the group
        - N_h as the number of galaxies in the corresponding true halo
        - N_s as the number of galaxies in the group that are correctly classified as
            members of the corresponding halo
        - N_i as the number of interlopers in the group; galaxies classified to the group
            but that do not belong to the true halo.
    
    purity : np.array
        At index `i`, purity of the group to which galaxy `i` belongs (duplicated for every
        group member). Purity is defined as the percentage of the number of galaxies in the group
        that are correctly identified as part of the halo, N_s/N_g. Because N_g = N_s + N_i,
        the contamination fraction is given by 1 - purity = (N_g - N_s)/N_g = (N_i)/N_g. 
    completeness : np.array
        At index `i`, completeness of the group to which galaxy `i` belongs (duplicated for every
        group member). Completeness is defined as the percentage of galaxies in the halo that are 
        correctly identified as part of the group, N_s/N_h. 
    """
    groupid = np.array(groupid)
    haloid = np.array(haloid)
    galproperty=np.array(galproperty)
    ngal=len(groupid)
    completeness=np.full(ngal,-999.)
    purity=np.full(ngal,-999.)
    unique_groups = np.unique(groupid)

    if (galproperty>-30).all() and (galproperty<-12).all():
        central_selection = np.min # minimum mag is brightest galaxy=central
        sortfactor=1
    elif (galproperty>0).all():
        central_selection = np.max # maximum mass is central
        sortfactor=-1
    
    for gg in unique_groups:
        groupsel = np.where(groupid==gg)
        N_g = len(groupsel[0])
        sortidx = np.argsort(sortfactor*galproperty[groupsel])
        hh = mode(haloid[groupsel][sortidx], keepdims=True)[0][0]
        halosel = np.where(haloid==hh)
        N_h = len(halosel[0])
        N_s = np.sum(haloid[groupsel]==hh)
        purity[groupsel]=N_s/N_g
        completeness[groupsel]=N_s/N_h
    return purity, completeness
    
def get_metrics_by_halo(groupid, haloid, galproperty):
    """
    For a group catalog constructed from a mock galaxy catalog,
    compute galaxy-wise purity and completeness metrics using
    true halo IDs for comparison. This function computes metrics
    on a halo-by-halo basis.

    Parameters
    ---------------------------
    groupid : iterable
        Group ID numbers after applying group-finding algorithm, length = # galaxies.
    haloid : iterable
        Halo ID numbers extracted from mock catalog halos, length = # galaxies = len(groupid).
    galproperty : iterable
        Group property by which to determine the central galaxy in the group. If all values are
        >-15 and <-27, then galproperty is assumed to be a magnitude, and the central will be the
        brightest galaxy. If all values are >0, this value is assumed to be mass, and the central
        will be selected by the maximum.

    Returns
    ---------------------------
    Suppose we map halos to groups, and define
        - N_g as the number of galaxies in the group
        - N_h as the number of galaxies in the corresponding true halo
        - N_s as the number of galaxies in the group that are correctly classified as
            members of the corresponding halo
        - N_i as the number of interlopers in the group; galaxies classified to the group
            but that do not belong to the true halo.
    
    purity : np.array
        At index `i`, purity of the group to which galaxy `i` belongs (duplicated for every
        group member). Purity is defined as the percentage of the number of galaxies in the group
        that are correctly identified as part of the halo, N_s/N_g. Because N_g = N_s + N_i,
        the contamination fraction is given by 1 - purity = (N_g - N_s)/N_g = (N_i)/N_g. 
    completeness : np.array
        At index `i`, completeness of the group to which galaxy `i` belongs (duplicated for every
        group member). Completeness is defined as the percentage of galaxies in the halo that are 
        correctly identified as part of the group, N_s/N_h. 
    """
    groupid = np.array(groupid)
    haloid = np.array(haloid)
    galproperty=np.array(galproperty)
    ngal=len(groupid)
    completeness=np.full(ngal,-999.)
    purity=np.full(ngal,-999.)
    unique_halos = np.unique(haloid)

    if (galproperty>-30).all() and (galproperty<-12).all():
        central_selection = np.min # minimum mag is brightest galaxy=central
        sortfactor=1
    elif (galproperty>0).all():
        central_selection = np.max # maximum mass is central
        sortfactor=-1
        
    for hh in unique_halos:
        halosel = np.where(haloid==hh)
        N_h = len(halosel[0])
        sortidx = np.argsort(sortfactor*galproperty[halosel])
        gg = mode(groupid[halosel][sortidx], keepdims=True)[0][0]
        groupsel = np.where(groupid==gg)
        N_g = len(groupsel[0])
        N_s = np.sum(haloid[groupsel]==hh)
        purity[halosel]=N_s/N_g
        completeness[halosel]=N_s/N_h
    return purity, completeness

if __name__=='__main__':
    import pandas as pd
    data = pd.read_hdf("./halobiasmocks/fiducial/ECO_cat_0_Planck_memb_cat.hdf5")
    pur,comp=get_metrics_by_halo(data.groupid, data.haloid, data.M_r)
    data['pur']=pur
    data['comp']=comp

    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    data=data[data.g_galtype==1]
    fig,axs = plt.subplots(ncols=2, sharey=True)
    axs[0].scatter(data.M_group, data.pur, s=2, alpha=0.9)
    axs[0].set_xlabel("FoF+HAM Mass")
    axs[0].set_ylabel("group purity")
    axs[1].hist(data.pur, log=True, histtype='step', bins=np.arange(0,1.05,0.05), orientation='horizontal')
    axs[1].axhline(np.mean(data.pur), label='Mean', color='red')
    axs[1].axhline(np.median(data.pur), label='Median', color='purple')
    axs[1].legend(loc='best')
    plt.show()
    

    fig,axs = plt.subplots(ncols=2, sharey=True)
    axs[0].scatter(data.M_group, data.comp, s=2, alpha=0.9)
    axs[0].set_xlabel("FoF+HAM Mass")
    axs[0].set_ylabel("group completeness")
    axs[1].hist(data.comp, log=True, histtype='step', bins=np.arange(0,1.05,0.05), orientation='horizontal')
    axs[1].axhline(np.mean(data.comp), label='Mean', color='red')
    axs[1].axhline(np.median(data.comp), label='Median', color='purple')
    axs[1].legend(loc='best')
    plt.show()
    print(data[['pur','comp']].mean())
    print(data[['pur','comp']].median())
