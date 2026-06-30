import numpy as np
from scipy.spatial import cKDTree
from numba import njit, prange
from tqdm import tqdm
from pg3tools import *
SPEED_OF_LIGHT = 3e5
sqrt_2pi = 2.5066282746

#
# Python code for the dwarf association routine
# from Hutchens et al. 2023 / 2023ApJ...956...51H
#

def prob_dwarfAssocRoutine(dwarfra, dwarfdec, dwarfz, dwarfzerr, giantra, giantdec, giantz, giantzerr, giantgrpid, radius_boundary, velocity_boundary, Pth, cosmo, n_pts_per_sigma):
    """ 
    Associate galaxies to a group catalog based on given radius and velocity boundaries, based on a method
    similar to that presented in Eckert+ 2016. As used in Hutchens+2023 

    Parameters
    ----------
    dwarfra : iterable
        Right-ascension of dwarf galaxies in degrees.
    dwarfdec : iterable
        Declination of dwarf galaxies in degrees.
    dwarfz : iterable
        Redshifts of dwarf galaxies
    dwarfz : iterable
        Redshift uncertainties of dwarf galaxies
    giantra : iterable
        Right-ascension of giant galaxies in degrees.
    giantdec : iterable
        Declination of giant galaxies in degrees. Length matches `giantra`.
    giantz : iterable
        Redshift of giant galaxies. Length matches `giantra`.
    giantzerr : iterable
        Redshift uncertainties of giant galaxies. Length matches `giantra`.
    giantgrpid : iterable
        group ID of each giant galaxy (i.e., from `foftools.fast_fof`.) Length matches `giantra`.
    radius_boundary : callable
        Radius within which to search for dwarf galaxies around giant-only groups. Callable function of group Ngiants.
    velocity_boundary : callable 
        Velocity from group center within which to search for dwarf galaxies around giant-only groups.
        Callable function of group Ngiants.
    cosmo : astropy.cosmology object
        Astropy cosmology to specify cosmological distances.
    Pth : float
        Threshold probability.
    n_pts_per_sigma : int
        Number of points per standard deviation in redshift grid (for numeric integration).

    Returns
    -------
    assoc_grpid : iterable
        group ID of every dwarf galaxy. Length matches `dwarfra`.
    assoc_flag : iterable
        association flag for every galaxy (see function description). Length matches `dwarfra`.
    """
    dwarfra = np.asarray(dwarfra)
    dwarfdec = np.asarray(dwarfdec)
    dwarfz = np.asarray(dwarfz)
    dwarfzerr = np.asarray(dwarfzerr)
    giantra = np.asarray(giantra)
    giantdec = np.asarray(giantdec)
    giantz = np.asarray(giantz)
    giantzerr = np.asarray(giantzerr)
    giantgrpid = np.asarray(giantgrpid)
    assert callable(radius_boundary), "`radius_boundary` must be callable"
    assert callable(velocity_boundary), "`velocity_boundary` must be callable"
    giantpznorm = 1 / (sqrt_2pi * giantzerr)
    giantinvden2 = -0.5 / (giantzerr * giantzerr)

    uniqgrpid, uniqidx, uniqN = np.unique(giantgrpid, return_index=True, return_counts=True)
    grpra, grpdec, grpz = prob_group_skycoords(giantra, giantdec, giantz, giantzerr, giantgrpid, n_pts_per_sigma)
    grpra = grpra[uniqidx]
    grpdec = grpdec[uniqidx]
    grpz = grpz[uniqidx]
    grp_cmvg = cosmo.comoving_transverse_distance(grpz).value
    velocity_boundary=velocity_boundary(uniqN)
    radius_boundary=radius_boundary(uniqN)
    epsilon = (1 + grpz)/SPEED_OF_LIGHT * velocity_boundary
    dwarf_cmvg = cosmo.comoving_transverse_distance(dwarfz).value

    # get cartesian positions of groups, form a kd tree, then find closest dwarf neighbors 
    # using a query ball point.
    zmin = min([grpz.min(),dwarfz.min()])
    grpXmin, grpYmin, grpZmin = comoving_cartesian_from_spherical(grpra, grpdec, zmin, cosmo)
    dwarfXmin, dwarfYmin, dwarfZmin = comoving_cartesian_from_spherical(dwarfra, dwarfdec, zmin, cosmo)
    grpcoords = np.array([grpXmin,grpYmin,grpZmin]).T
    grp_tree = cKDTree(grpcoords)
    dwarf_tree = cKDTree(np.array([dwarfXmin, dwarfYmin, dwarfZmin]).T)
   
    Ndwarf = len(dwarfra)
    assoc_grpid = np.zeros(Ndwarf, dtype=np.int32) - 1
    assoc_flag = np.zeros(Ndwarf, dtype=np.bool_)
    r2plusv2 = np.zeros(Ndwarf)
    N_G = len(grpra)
    for grp_i in tqdm(range(N_G)):
        dwarf_i = np.array(dwarf_tree.query_ball_point(grpcoords[grp_i], r=radius_boundary[grp_i]))
        if len(dwarf_i) == 0:
            continue
       
        alpha_ij = angular_separation(grpra[grp_i], grpdec[grp_i], dwarfra[dwarf_i], dwarfdec[dwarf_i]) 
        Rp = 0.5 * (grp_cmvg[grp_i] + dwarf_cmvg[dwarf_i]) * alpha_ij
        to_integrate = np.where(Rp < radius_boundary[grp_i])
        if len(to_integrate[0]) == 0:
            continue

        plink = np.zeros(len(dwarf_i))
        plink[to_integrate] = get_dwarf_assoc_prob(giantz, giantzerr, giantpznorm, giantinvden2, giantgrpid,\
             dwarfz[dwarf_i[to_integrate]], dwarfzerr[dwarf_i[to_integrate]], uniqgrpid[grp_i], epsilon[grp_i],\
             n_pts_per_sigma)
        assoc_condition = plink > Pth
        assoc_sep = (Rp*Rp)/(radius_boundary[grp_i]*radius_boundary[grp_i])
      
        # `associate` is a bool: it is true when Rproj requirement is met (`assoc_condition`) and
        # either (i) the dwarf was never previously associated or (ii) the new giant-only group is a better fit. 
        associate = assoc_condition & (~assoc_flag[dwarf_i] | (assoc_flag[dwarf_i] & (assoc_sep < r2plusv2[dwarf_i])))
        assoc_grpid[dwarf_i[associate]] = uniqgrpid[grp_i]
        assoc_flag[dwarf_i[associate]] = True
        r2plusv2[dwarf_i[associate]] = assoc_sep[associate]

    still_isolated = (assoc_grpid < 0)
    maxgrpid = np.max(giantgrpid)
    assoc_grpid[still_isolated] = np.arange(maxgrpid+1, maxgrpid+np.sum(still_isolated)+1)
    return assoc_grpid, assoc_flag

@njit
def get_dwarf_assoc_prob(giantz, giantzerr, giantpznorm, giantpzinvden2, giantgrpid, dwarfz, dwarfzerr, grpid, eps, n_pts_per_sigma):
    """
    Helper function to obtain integrated p(z) probabilities for prob_dwarfAssocRoutine.
    See that function for details regarding arguments.
    """
    giantsel = np.where(giantgrpid == grpid)
    zvals = giantz[giantsel]
    zerrvals = giantzerr[giantsel]
    zgrid = get_adaptive_zgrid(zvals, zerrvals, dwarfz, dwarfzerr, n_pts_per_sigma)
    pz1 = get_pz_group(zgrid, zvals, giantpznorm[giantsel], giantpzinvden2[giantsel])
    passoc = np.zeros_like(dwarfz)
    for k in range(len(passoc)):
        passoc[k] = dbint_pz_jgauss(zgrid, pz1, dwarfz[k], dwarfzerr[k], eps)
    return passoc
