import numpy as np
from scipy.spatial import cKDTree
from numba import prange, njit 
from pg3tools import *
SPEED_OF_LIGHT = 3.0e5
sqrt_2pi = 2.5066282746

#
# Python code for the dwarf-only merging subroutine
# from Hutchens et al. 2023 / 2023ApJ...956...51H
#

def prob_dwarfOnlyICRoutine(galaxyra, galaxydec, galaxyz, galaxyzerr, galaxyabsmag, rprojboundary, vprojboundary, Pth, cosmo, starting_id, n_pts_per_sigma):
    """
    Iteratively combine dwarf-only FoF groups using group Lr-based boundaries.

    Parameters
    --------------
    galaxyra : array_like
        RA of dwarf galaxies in decimal degrees.
    galaxydec : array_like
        Dec of dwarf galaxies in decimal degrees.
    galaxyz : array_like
        Redshift of dwarf galaxies.
    galaxyzerr : array_like
        Redshift uncertainties of dwarf galaxies.
    galaxyabsrmag : array_like
        Absolute magnitude of input galaxies.
    rprojboundary : callable
        Search boundary to apply on-sky. Should be callable function of group Lr.
        Units: Mpc consistent with `cosmo`.
    vprojboundary : callable
        Search boundary to apply in line-of-sight. Should be callable function of group Lr.
        Units: km/s
    Pth : float
        Threshold probability to merge dwarf-only seed groups.
    cosmo : astropy.cosmology object
       Astropy cosmology for computing cosmological distances.
    starting_id : int
        Group ID to start at when assigning dwarf group IDs. This is avoid confusion
        with previously assigned group ID numbers.
    n_pts_per_sigma : int
        Number of points per redshift uncertainty to resolve in probabilistic integration.
    
    Returns
    --------------
    grpid : np.array
        Array of group ID numbers following iterative combination.
    """
    galaxyra=np.asarray(galaxyra)
    galaxydec=np.asarray(galaxydec)
    galaxyz=np.asarray(galaxyz)
    galaxyzerr=np.asarray(galaxyzerr)
    galaxyabsmag=np.asarray(galaxyabsmag)
    assert callable(rprojboundary),"Argument `rprojboundary` must callable function of group Lr."
    assert callable(vprojboundary),"Argument `vprojboundary` must callable function of group Lr."

    grpid = np.arange(starting_id, starting_id+len(galaxyra), dtype=np.int32)
    converged=False
    niter=0
    while (not converged):
        print(f"Dwarf-only iterative combination {niter+1} in progress...")
        oldgrpid = grpid
        grpid = dwarf_prob_nearest_neighbor_assign(galaxyra,galaxydec,galaxyz,galaxyzerr,galaxyabsmag,\
                       oldgrpid,rprojboundary,vprojboundary,Pth,cosmo,n_pts_per_sigma)
        converged = np.array_equal(oldgrpid,grpid)
        niter+=1
    print("Dwarf-only iterative combination complete.")
    return grpid

def dwarf_prob_nearest_neighbor_assign(galaxyra,galaxydec,galaxyz,galaxyzerr,galaxyabsmag,grpid,rprojboundary,vprojboundary,Pth,cosmo,n_pts_per_sigma):
    """
    Refine input group ID by merging nearest-neighbor groups subject to boundary constraints.
    For info on arguments, see "prob_dwarfOnlyICRoutine"

    Returns
    --------------
    refinedgrpid : np.array
        Refined group ID numbers based on nearest-neighbor merging.
    """
    gauss_norm = 1 / (sqrt_2pi * galaxyzerr)
    invden2 = -0.5 / (galaxyzerr * galaxyzerr)

    uniqgrpid, uniqind, galaxyidx, seedN = np.unique(grpid, return_index=True, return_inverse=True, return_counts=True)
    groupMint = get_int_mag(galaxyabsmag, grpid)
    gX, gY, gZ = cartesian_from_spherical_z(galaxyra, galaxydec, galaxyz)
    seedra, seeddec, seedz = prob_group_skycoords(galaxyra,galaxydec,galaxyz,galaxyzerr,grpid,n_pts_per_sigma)
    seedra, seeddec, seedz = seedra[uniqind], seeddec[uniqind], seedz[uniqind]
    seedMint = groupMint[uniqind]
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
    Mr_tent = -2.5*np.log10(10**(-0.4*seedMint) + 10**(-0.4*seedMint[nnind]))
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

    seedi_Rcond = (dperp_i < rprojboundary(Mr_tent))
    seedj_Rcond = (dperp_j < rprojboundary(Mr_tent)) 
    recip = nnind[nnind] == np.arange(len(nnind))
    idx_to_integrate = np.where(seedi_Rcond & seedj_Rcond & recip)
    zavg = 0.5 * (seedz + seedz[nnind])
    eps = (1 + zavg)/SPEED_OF_LIGHT * vprojboundary(Mr_tent)
    prob = integrate_IC(idx_to_integrate[0], galaxyz, galaxyzerr, grpid, gauss_norm, invden2,\
            uniqgrpid, nnind, seedN, eps, n_pts_per_sigma)

    check = (prob>1.01)
    if check.any():
        print(f"WARNING: prob_dwarfonlyic finds {check.sum()} group pairs with P_ij > 1.0 (max value {prob.max():0.3f}).")

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
