import numpy as np
from scipy.spatial import cKDTree
from copy import deepcopy
from g3misc import *
SPEED_OF_LIGHT = 3.0e5

#
# Python code for the giant-only merging subroutine
# from Hutchens et al. 2023 / 2023ApJ...956...51H
#

def giantOnlyICRoutine(galaxyra, galaxydec, galaxyz, galaxyzerr, giantfofid, rprojboundary, vprojboundary, cosmo):
    """
    Iteratively combine giant-only FoF groups using group N_giants-based boundaries.

    Parameters
    --------------
    galaxyra : array_like
        RA of giant galaxies in decimal degrees.
    galaxydec : array_like
        Dec of giant galaxies in decimal degrees.
    galaxyz : array_like
        Redshift of giant galaxies.
    galaxyzerr : array_like
        Redshift uncertainties of giant galaxies.
    giantfofid : array_like
        FoF group ID for each giant galaxy, length matches `galaxyra`.
    rprojboundary : callable
        Search boundary to apply on-sky. Should be callable function of group N_giants.
        Units: Mpc consistent with `cosmo`.
    vprojboundary : callable
        Search boundary to apply in line-of-sight. Should be callable function of group N_giants.
        Units: km/s
    cosmo : astropy.cosmology object
       Astropy cosmology for computing cosmological distances.
    
    Returns
    --------------
    giantgroupid : np.array
        Array of group ID numbers following iterative combination.
        Unique values match that of `giantfofid`.
    """
    galaxyra=np.asarray(galaxyra)
    galaxydec=np.asarray(galaxydec)
    galaxyz=np.asarray(galaxyz)
    galaxyzerr=np.asarray(galaxyzerr)
    giantfofid=np.asarray(giantfofid)
    assert callable(rprojboundary),"Argument `rprojboundary` must callable function of N_giants."
    assert callable(vprojboundary),"Argument `vprojboundary` must callable function of N_giants."

    giantgroupid = np.copy(giantfofid)
    converged=False
    niter=0
    while (not converged):
        print(f"Giant-only iterative combination {niter+1} in progress...")
        oldgiantgroupid = giantgroupid
        giantgroupid = prob_nearest_neighbor_assign(galaxyra,galaxydec,galaxyz,oldgiantgroupid,\
                        rprojboundary,vprojboundary,cosmo)
        converged = np.array_equal(oldgiantgroupid,giantgroupid)
        niter+=1
    print("Giant-only iterative combination complete.")
    return giantgroupid

def prob_nearest_neighbor_assign(galaxyra,galaxydec,galaxyz,galaxyzerr,grpid,rprojboundary,vprojboundary,cosmo):
    """
    Refine input group ID by merging nearest-neighbor groups subject to boundary constraints.
    For info on arguments, see "giantOnlyICRoutine"

    Returns
    --------------
    refinedgrpid : np.array
        Refined group ID numbers based on nearest-neighbor merging.
    """
    # Prepare output array
    groupra, groupdec, groupz = group_skycoords(galaxyra, galaxydec, galaxyz, grpid)
    gX, gY, gZ = cartesian_from_spherical_z(galaxyra, galaxydec, galaxyz)

    # Get unique potential seed groups, defining a sphere of radius z_min to catch
    # all possible neighbors regardless of zphot values
    uniqgrpid, uniqind, galaxyidx, seedN = np.unique(grpid, return_index=True, return_inverse=True, return_counts=True)
    seedra, seeddec, seedz = groupra[uniqind], groupdec[uniqind], groupz[uniqind]
    seeddm = cosmo.comoving_transverse_distance(seedz).value
    zmin = np.zeros_like(seedz) + np.min(seedz)
    seedX, seedY, seedZ = cartesian_from_spherical_z(seedra, seeddec, zmin)
    xyz = np.array([seedX, seedY, seedZ]).T
    kdt = cKDTree(xyz)
    nndist, nnind = kdt.query(xyz,k=2)
    nndist=nndist[:,1] # ignore self match
    nnind=nnind[:,1]
 
    # check whether, supposing the two seed groups were combined into a tentative          <------- stopped here.
    # larger group, if both seed groups would satisfy the Rproj and Vproj requirements
    # when calculated from the center of the larger tentative group. In probabilistic 
    # space, this means checking if 
    n_tent = seedN + seedN[nnind]
    G = len(uniqgrpid)
    sumX = np.zeros(G)
    sumY = np.zeros(G)
    sumZ = np.zeros(G)
    np.add.at(sumX, galaxyidx, gX)
    np.add.at(sumY, galaxyidx, gY)
    np.add.at(sumZ, galaxyidx, gZ)
    tent_Xcen = (sumX + sumX[nnind]) / n_tent
    tent_Ycen = (sumY + sumY[nnind]) / n_tent
    tent_Zcen = (sumZ + sumZ[nnind]) / n_tent


    tent_z = np.sqrt(tent_Xcen*tent_Xcen + tent_Ycen*tent_Ycen + tent_Zcen*tent_Zcen)
    tent_ra = (np.degrees(np.arctan2(tent_Ycen,tent_Xcen))+360.) % 360.0
    tent_dec = np.degrees(np.arcsin(tent_Zcen / tent_z))
    tent_dm = cosmo.comoving_transverse_distance(tent_z).value

    alpha_i = angular_separation(seedra, seeddec, tent_ra, tent_dec)
    alpha_j = angular_separation(seedra[nnind], seeddec[nnind], tent_ra, tent_dec)
    dperp_i = 0.5 * (seeddm + tent_dm) * alpha_i
    dperp_j = 0.5 * (seeddm[nnind] + tent_dm) * alpha_j
    one_plus_z = (1 + tent_z)
    vpec_i = SPEED_OF_LIGHT*np.abs(tent_z - seedz) / one_plus_z
    vpec_j = SPEED_OF_LIGHT*np.abs(tent_z - seedz[nnind]) / one_plus_z 

    seedi_in = (dperp_i < rprojboundary(n_tent)) & (vpec_i < vprojboundary(n_tent))
    seedj_in = (dperp_j < rprojboundary(n_tent)) & (vpec_j < vprojboundary(n_tent))
    recip = nnind[nnind] == np.arange(len(nnind)) # are they each other's own nearest neighbor?
    merge = seedi_in & seedj_in & recip
  
    ii=np.where(merge)[0]
    jj=nnind[ii]
    keep = (ii<jj)
    ii, jj = ii[keep], jj[keep]
    revisedid = np.minimum(uniqgrpid[ii],uniqgrpid[jj])
    uniqgrpid[ii] = revisedid
    uniqgrpid[jj] = revisedid
    refinedgrpid = uniqgrpid[galaxyidx]
    return refinedgrpid
