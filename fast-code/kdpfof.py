import time
import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components
from scipy.interpolate import interp1d
from numba import njit, prange, vectorize
from math import erf
from pg3tools import prob_group_skycoords

SPEED_OF_LIGHT = 3.0e5
sqrt_2pi = 2.506628274631
sqrt2 = 1.41421356237

def kdPFOF(ra, dec, zz, zerr, perpll, losll, Pth, cosmo, npoints_per_std=3):
    """
    Compute probability friends-of-friends group memberships using the algorithm
    described by Liu et al. 2008 [2008ApJ...681.1046L]. This code uses a kd-tree
    to find candidate neighbor pairs prior to computing linking probabilities,
    thereby reducing computation time. Linking probabilities are calculated
    with get_pfof_probabilities which is compiled with numba.

    Parameters
    --------------
    ra : np.array
        Right ascension in decimal degrees.
    dec : np.array
        Declination in decimal degrees.
    zz : np.array
        Redshifts
    zerr : np.array
        Redshift uncertainties
    perpll : float
        Perpendicular linking length in Mpc
    losll : float
        LOS linking length in Mpc
    Pth : float
        PFoF probability threshold
    cosmo : astropy.cosmology object
        Assumed cosmology (for comoving distances)
    npoints_per_std : int
        Number of points per standard deviation to be included
        in PFoF numeric integration. A higher value will result
        in more accurate integration results. Default 3.

    Returns
    ---------------
    pfofID : np.array
        PFoF group ID for each input galaxy.
        Length matches `ra`.
    """
    t1 = time.time()
    Ngal = len(ra)
    ra = np.asarray(ra)
    dec = np.asarray(dec)
    zz = np.asarray(zz)
    zerr = np.asarray(zerr)
    assert (len(ra)==len(dec) and len(dec)==len(zz)),"RA/Dec/zz arrays must equivalent length."

    # coordinate info
    phi = np.deg2rad(ra)
    theta = np.pi/2. - np.deg2rad(dec) 
    dc = cosmo.comoving_distance(zz).value
    dm = cosmo.comoving_transverse_distance(zz).value
    xyz = np.zeros((Ngal,3))
    xyz[:,0] = dc * np.sin(theta) * np.cos(phi)
    xyz[:,1] = dc * np.sin(theta) * np.sin(phi)
    xyz[:,2] = dc * np.cos(theta)
    xyz_angular = xyz / dc[:,np.newaxis]

    # linking lengths
    dc_upper = dc + losll
    z_arr_interp = np.arange(0.0001, 2*np.max(zz), np.min(zerr)/3)
    z_dc_interp = interp1d(cosmo.comoving_distance(z_arr_interp).value, z_arr_interp, fill_value=0, bounds_error=False)
    VL = z_dc_interp(dc_upper) - zz

    # neighbor search
    tree = cKDTree(xyz_angular)
    rmax = perpll / np.min(dc)
    pairs = tree.query_pairs(r=rmax, output_type='ndarray')
    i_idx, j_idx = pairs[:,0], pairs[:,1]
    ri = xyz[i_idx]
    rj = xyz[j_idx]
    dot = (ri * rj).sum(axis=1)
    mu = dot / (dc[i_idx] * dc[j_idx]) #mu=cos(alpha_ij)
    if (mu>1).any():
        mu[mu>1] = 1.
    dperp = (dm[i_idx]+dm[j_idx]) * np.sqrt((1-mu)/2.) # Eq 1. H23, but approximating sin(alpha_ij/2) ~ alpha_ij/2.

    # probabilistic linking w/ embedded dperp<=perpll cut
    assert isinstance(npoints_per_std,int)
    prob = get_pfof_probabilities(zz, zerr, i_idx, j_idx, dperp, perpll, VL, npoints_per_std)

    # Compute outputs
    friendship = (prob >= Pth)
    friendship = sp.coo_array((friendship[friendship], (i_idx[friendship],j_idx[friendship])), shape=(Ngal,Ngal))
    pfofID = 1+connected_components(friendship)[1] 
    print(f'PFoF completed in {time.time()-t1:0.3f} s.')
    return pfofID

# ---------------------------------------------------- #
# PFoF helpers
# ---------------------------------------------------- #
@njit(parallel=True,fastmath=True)
def get_pfof_probabilities(zz, zerr, i_idx, j_idx, dperp, perpLL, VL, npoints_per_std=3):
    """
    Calculate PFoF linking probabilities, given on-sky and line-of-sight information.
    The line-of-sight linking probability for galaxies i and j is calculated as:
    
    P_ij = int[Gi(z) * g(z) * dz] from 0 -> infty

    where Gi(z) is a Gaussian for galaxy i and g(z) = int[Gj(z')dz'] from z-VL -> z+VL,
    with VL being the line-of-sight linking length. 
    
    Parameters
    --------------------
    zz : np.array
        Input redshifts
    zerr : np.array
        Input redshift uncertainties
    i_idx, j_idx : np.array
        Indices of zz that yield neighbor pairs from kd tree.
        (see main PFOF code.)
    dperp : np.array
        Transverse on-sky distance between neighbors.
    perpLL : float
        On-sky linking length.
        Linking probabilities are calculated only for neighbor pairs for
        which dperp<=perpLL, to reduce compute time.
    VL : float
        line-of-sight linking length in redshift units
    npoints_per_std : int, default 3
        Sampling of Gaussian photo-z PDFs - number of points per standard
        deviation to be included in the numeric integration.

    Returns 
    --------------------
    prob : np.array
        Linking probability for each neighbor pair. 0 if dperp > perpLL,
        even if the line-of-sight linking probability is truly nonzero.
    """
    prob = np.zeros(len(dperp))
    idx_to_integrate = np.where(dperp <= perpLL)
    i_idx_integrate = i_idx[idx_to_integrate]
    j_idx_integrate = j_idx[idx_to_integrate]
    zz_i, zz_j = zz[i_idx_integrate], zz[j_idx_integrate]
    zerr_i, zerr_j = zerr[i_idx_integrate], zerr[j_idx_integrate]
    VL_i = VL[i_idx_integrate]

    norm_i = 1 / (zerr_i * sqrt_2pi)
    sig_i_squared = np.square(zerr_i)
    inv_den2 = 1 / (sqrt2 * zerr_j)
    maxerr = np.where(zerr_i > zerr_j, zerr_i, zerr_j)
    zmax = np.where(zz_i > zz_j, zz_i, zz_j)
    zmin = np.where(zz_i < zz_j, zz_i, zz_j)

    Pij = np.zeros(len(zz_i))
    for k in prange(len(Pij)):
        dz = maxerr[k] / npoints_per_std
        meshZ = np.arange(zmin[k]-5*maxerr[k], zmax[k]+5*maxerr[k], dz)
        dz_i_squared = np.square(zz_i[k] - meshZ)
        Gi = norm_i[k] * np.exp(-0.5 * dz_i_squared / sig_i_squared[k])
        dz_j = zz_j[k] - meshZ
        arghi = (dz_j + VL_i[k]) * inv_den2[k]
        arglo = (dz_j - VL_i[k]) * inv_den2[k]
        gz = 0.5 * (erf_vec(arghi) - erf_vec(arglo))
        Pij[k] = np.sum(dz * Gi * gz)
    prob[idx_to_integrate] = Pij 
    return prob

@vectorize(['float64(float64)'])
def erf_vec(x):
    return erf(x)

if __name__=='__main__':
    # usage example
    from astropy.cosmology import Planck18 as planck
    import pandas as pd
    ls = pd.read_csv('/srv/two/zhutchen/paper3/data/lsdr9/ls_deg.csv')
    fofid=kdPFOF(ls.radeg, ls.dedeg, ls.zbest, ls.zbesterr, 0.07*3.5, 1.1*3.5, 0.1, planck, 10)
    grpra,grpdec,grpz=prob_group_skycoords(ls.radeg, ls.dedeg, ls.zbest, ls.zbesterr, fofid)
