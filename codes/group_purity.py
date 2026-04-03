import numpy as np
from scipy.stats import mode

def get_metrics_by_group(groupid, haloid, galproperty, enforce_positive_IDs=False):
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

    if (galproperty>-30).all() and (galproperty<-1).all():
        central_selection = np.min # minimum mag is brightest galaxy=central
        sortfactor=1
    elif (galproperty>0).all():
        central_selection = np.max # maximum mass is central
        sortfactor=-1
    
    for gg in unique_groups:
        groupsel = np.where(groupid==gg)
        N_g = len(groupsel[0])
        sortidx = np.argsort(sortfactor*galproperty[groupsel])
        if enforce_positive_IDs:
            take_mode_of = haloid[groupsel][sortidx]
            take_mode_of = take_mode_of[take_mode_of>0]
        else:
            take_mode_of = haloid[groupsel][sortidx]
        hh = mode(take_mode_of, keepdims=True)[0][0]
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

    if (galproperty>-30).all() and (galproperty<-1).all():
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
        if (hh<0):
            purity[halosel] = -99.
            completeness[halosel] = -99.
        else:
            purity[halosel]=N_s/N_g
            completeness[halosel]=N_s/N_h
    return purity, completeness

# ---------------------------------------------------- #
# ---------------------------------------------------- #
# Functions for ref-test mappings (PG3 paper)

def PC_test_to_ref(test_group_id, test_group_refid, galproperty, uniq_ref_group_id, ref_grpn):
    """
    Calculate purity and completeness of test groups relative to reference groups,
    based on the methodology of Hutchens+2026 (PG3).

    Parameters
    --------------
    test_group_id : array
        Test group identifier for each galaxy in the test group catalog.
    test_group_refid : array
        Reference group identifier for each galaxy in the test group catalog.
    galproperty : array
        Mass or absolute magnitude for each galaxy in the test group catalog.
    uniq_ref_group_id : array
        Array of unique reference group identifiers.
    ref_grpn : array
        Array of group N values for reference groups. Length matches uniq_ref_group_id.

    Returns
    --------------
    purity, completeness : array
        Purity and completeness of test groups (see H26 definitions).
        -999 in cases where there is no matched reference group.
        Length matches uniq_test_group_id below.
    uniq_test_group_id : array
        Unique identifiers of test groups.
    uniq_test_grpn : array
        Galaxy counts for each unique test group.
    """
    test_group_id = np.array(test_group_id)
    test_group_refid = np.array(test_group_refid)
    galproperty = np.array(galproperty)
    uniq_ref_group_id = np.array(uniq_ref_group_id)
    ref_grpn = np.array(ref_grpn)
    assert len(uniq_ref_group_id)==len(ref_grpn)

    if (galproperty>-30).all() and (galproperty<-1).all():
        central_selection = np.min # minimum mag is brightest galaxy=central
        sortfactor=1
    elif (galproperty>0).all():
        central_selection = np.max # maximum mass is central
        sortfactor=-1
    
    uniq_test_group_id, uniq_test_grpn = np.unique(test_group_id, return_counts=True)
    purity = np.zeros(len(uniq_test_group_id))
    completeness = np.zeros(len(uniq_test_group_id))
    for ii,Tid in enumerate(uniq_test_group_id):
        testgroupsel = (test_group_id == Tid)
        refIDs = test_group_refid[testgroupsel]
        if (refIDs < 0).all():
            purity[ii] = -999.
            completeness[ii] = -999.
        else:
            sortidx = np.argsort(sortfactor*galproperty[testgroupsel])
            take_mode_of = refIDs[sortidx]
            take_mode_of = take_mode_of[take_mode_of>0] # ignore -999.
            mapped_refID = mode(take_mode_of,keepdims=True)[0][0]
            Ntest = np.sum(testgroupsel)
            Nref = ref_grpn[uniq_ref_group_id == mapped_refID]
            Ns = np.sum(test_group_refid[testgroupsel]==mapped_refID)
            purity[ii] = Ns/Ntest
            completeness[ii] = Ns/Nref
    return purity, completeness, uniq_test_group_id, uniq_test_grpn

def PC_ref_to_test(test_group_id, test_group_refid, galproperty, uniq_ref_group_id, ref_grpn):
    test_group_id = np.array(test_group_id)
    test_group_refid = np.array(test_group_refid)
    galproperty = np.array(galproperty)
    uniq_ref_group_id = np.array(uniq_ref_group_id)
    ref_grpn = np.array(ref_grpn)
    assert len(uniq_ref_group_id)==len(ref_grpn)

    # do I need to assign galproperty for reference galaxies in this case??


    if (galproperty>-30).all() and (galproperty<-1).all():
        central_selection = np.min # minimum mag is brightest galaxy=central
        sortfactor=1
    elif (galproperty>0).all():
        central_selection = np.max # maximum mass is central
        sortfactor=-1
    
ass
            
if __name__=='__main__':
    import pandas as pd
    import matplotlib.pyplot as plt
    data = pd.read_hdf("/srv/one/zhutchen/g3groupfinder/halobiasmocks/fiducial/ECO_cat_0_Planck_memb_cat.hdf5")
    haloid, halon = np.unique(data.haloid, return_counts=True)
    pp,cc,_,_ = PC_test_to_ref(data.groupid, data.haloid, data.M_r, haloid, halon)
    plt.figure()
    plt.plot(pp, cc, 'k.')
    plt.xlabel('pur')
    plt.ylabel('compl')
    plt.show()
    #pur,comp=get_metrics_by_halo(data.groupid, data.haloid, data.M_r)
    #data['pur']=pur
    #data['comp']=comp

    #import matplotlib
    #matplotlib.use('TkAgg')
    #import matplotlib.pyplot as plt
    #data=data[data.g_galtype==1]
    #fig,axs = plt.subplots(ncols=2, sharey=True)
    #axs[0].scatter(data.M_group, data.pur, s=2, alpha=0.9)
    #axs[0].set_xlabel("FoF+HAM Mass")
    #axs[0].set_ylabel("group purity")
    #axs[1].hist(data.pur, log=True, histtype='step', bins=np.arange(0,1.05,0.05), orientation='horizontal')
    #axs[1].axhline(np.mean(data.pur), label='Mean', color='red')
    #axs[1].axhline(np.median(data.pur), label='Median', color='purple')
    #axs[1].legend(loc='best')
    #plt.show()
    #

    #fig,axs = plt.subplots(ncols=2, sharey=True)
    #axs[0].scatter(data.M_group, data.comp, s=2, alpha=0.9)
    #axs[0].set_xlabel("FoF+HAM Mass")
    #axs[0].set_ylabel("group completeness")
    #axs[1].hist(data.comp, log=True, histtype='step', bins=np.arange(0,1.05,0.05), orientation='horizontal')
    #axs[1].axhline(np.mean(data.comp), label='Mean', color='red')
    #axs[1].axhline(np.median(data.comp), label='Median', color='purple')
    #axs[1].legend(loc='best')
    #plt.show()
    #print(data[['pur','comp']].mean())
    #print(data[['pur','comp']].median())
