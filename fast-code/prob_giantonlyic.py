import numpy as np
from scipy.spatial import cKDTree
from numba import prange, njit 
from pg3tools import *
SPEED_OF_LIGHT = 3.0e5
sqrt_2pi = 2.5066282746

#
# Python code for the giant-only merging subroutine
# from Hutchens et al. 2023 / 2023ApJ...956...51H
#

def prob_giantOnlyICRoutine(galaxyra, galaxydec, galaxyz, galaxyzerr, giantfofid, rprojboundary, vprojboundary, Pth, cosmo, n_pts_per_sigma):
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
    Pth : float
        Threshold probability.
    cosmo : astropy.cosmology object
       Astropy cosmology for computing cosmological distances.
    n_pts_per_sigma : int
        Number of points per standard deviation in p(z) integration.
    
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
        giantgroupid = prob_nearest_neighbor_assign(galaxyra,galaxydec,galaxyz,galaxyzerr,\
                       oldgiantgroupid,rprojboundary,vprojboundary,Pth,cosmo,n_pts_per_sigma)
        converged = np.array_equal(oldgiantgroupid,giantgroupid)
        niter+=1
    print("Giant-only iterative combination complete.")
    return giantgroupid

def prob_nearest_neighbor_assign(galaxyra,galaxydec,galaxyz,galaxyzerr,grpid,rprojboundary,vprojboundary,Pth,cosmo,n_pts_per_sigma):
    """
    Refine input group ID by merging nearest-neighbor groups subject to boundary constraints.
    For info on arguments, see "prob_giantOnlyICRoutine"

    Returns
    --------------
    refinedgrpid : np.array
        Refined group ID numbers based on nearest-neighbor merging.
    """
    gauss_norm = 1 / (sqrt_2pi * galaxyzerr)
    invden2 = -0.5 / (galaxyzerr * galaxyzerr)

    uniqgrpid, uniqind, galaxyidx, seedN = np.unique(grpid, return_index=True, return_inverse=True, return_counts=True)
    gX, gY, gZ = cartesian_from_spherical_z(galaxyra, galaxydec, galaxyz)
    seedra, seeddec, seedz = prob_group_skycoords(galaxyra,galaxydec,galaxyz,galaxyzerr,grpid,n_pts_per_sigma)
    seedra, seeddec, seedz = seedra[uniqind], seeddec[uniqind], seedz[uniqind]
    seedX, seedY, seedZ = cartesian_from_spherical_z(seedra, seeddec, seedz)#.min())
    seeddm = cosmo.comoving_transverse_distance(seedz).value

    # identify neighbor pairs
    seed_xyz_est = np.array([seedX,seedY,seedZ]).T
    kdt = cKDTree(seed_xyz_est)
    nndist, nnind = kdt.query(seed_xyz_est,k=2)
    nndist=nndist[:,1]
    nnind=nnind[:,1]

    # check spatial and LOS requirements for neighboring seed groups
    # assuming they are merged into a larger tentative ('tent') group
    n_tent = seedN + seedN[nnind]
    G = len(uniqgrpid)
    sumX = np.zeros(G)
    sumY = np.zeros(G)
    sumZ = np.zeros(G)
    np.add.at(sumX, galaxyidx, gX)
    np.add.at(sumY, galaxyidx, gY)
    np.add.at(sumZ, galaxyidx, gZ)
    tent_Xcen = (sumX + sumX[nnind]) / n_tent # <-- this is a simple average. Weight by 1/sigma2?
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

    seedi_Rcond = (dperp_i < rprojboundary(n_tent))
    seedj_Rcond = (dperp_j < rprojboundary(n_tent)) 
    recip = nnind[nnind] == np.arange(len(nnind))
    idx_to_integrate = np.where(seedi_Rcond & seedj_Rcond & recip)
    zavg = 0.5 * (seedz + seedz[nnind])
    eps = (1 + zavg)/SPEED_OF_LIGHT * vprojboundary(n_tent)
    prob = integrate_IC(idx_to_integrate[0], galaxyz, galaxyzerr, grpid, gauss_norm, invden2,\
            uniqgrpid, nnind, seedN, eps, n_pts_per_sigma)
   
    
    check = (prob>1.01)
    if check.any():
        print(f"WARNING: prob_giantonlyic finds {check.sum()} group pairs with P_ij > 1.0 (max value {prob.max():0.3f}).")
 
    merge = (prob > Pth)
    ii=np.where(merge)[0]
    jj=nnind[ii]
    keep = (ii<jj)
    ii, jj = ii[keep], jj[keep]
    revisedid = np.minimum(uniqgrpid[ii],uniqgrpid[jj])
    uniqgrpid[ii] = revisedid
    uniqgrpid[jj] = revisedid
    refinedgrpid = uniqgrpid[galaxyidx]
    return refinedgrpid
