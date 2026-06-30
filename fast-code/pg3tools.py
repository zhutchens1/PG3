import numpy as np
from math import erf
from numba import njit, vectorize, float64, prange
from scipy.stats import binned_statistic
from sklearn.utils import check_random_state

def prob_group_skycoords(galaxyra, galaxydec, galaxyz, galaxyzerr, galaxygrpid, n_pts_per_sigma=5):
    """ 
    -----
    Obtain a list of group centers (RA/Dec/cz) given a list of galaxy coordinates (equatorial)
    and their corresponding group ID numbers.
    
    Inputs (all same length)
       galaxyra : 1D iterable,  list of galaxy RA values in decimal degrees
       galaxydec : 1D iterable, list of galaxy dec values in decimal degrees
       galaxyz : 1D iterable, list of galaxy redshifts
       galaxyzerr : 1D iterable, list of galaxy redshift uncertainties
       galaxygrpid : 1D iterable, group ID number for every galaxy in previous arguments.
    
    Outputs (all shape match `galaxyra`)
       groupra : RA in decimal degrees of galaxy i's group center.
       groupdec : Declination in decimal degrees of galaxy i's group center.
       groupz : Redshift of galaxy i's group center.
    """
    galaxyra = np.asarray(galaxyra)
    galaxydec = np.asarray(galaxydec)
    galaxyz = np.asarray(galaxyz)
    galaxyzerr = np.asarray(galaxyzerr)
    galaxygrpid = np.asarray(galaxygrpid)

    uniqgrpid, firstidx, galaxyidx, grpn = np.unique(galaxygrpid, return_index=True, return_inverse=True, return_counts=True)
    Ngrps = len(uniqgrpid)
    z_grp = galaxyz[firstidx]
    zmin = np.zeros(Ngrps,dtype=np.float64)+99.
    zmax = np.zeros(Ngrps,dtype=np.float64)-99.
    zerrmin = np.zeros(Ngrps,dtype=np.float64)+99.
    zerrmax = np.zeros(Ngrps,dtype=np.float64)-99.
    np.minimum.at(zmin, galaxyidx, galaxyz)
    np.maximum.at(zmax, galaxyidx, galaxyz)
    np.minimum.at(zerrmin, galaxyidx, galaxyzerr)
    np.maximum.at(zerrmax, galaxyidx, galaxyzerr)
    dz = zerrmin / n_pts_per_sigma
    groupz = get_group_redshift(galaxyz, galaxyzerr, galaxygrpid, uniqgrpid, grpn, zmin, zmax, zerrmin, zerrmax, dz, z_grp)
    groupz = groupz[galaxyidx]

    nmembers = np.bincount(galaxyidx, minlength=Ngrps)
    galaxyX, galaxyY, galaxyZ = cartesian_from_spherical_z(galaxyra, galaxydec, groupz)
    Xcen = np.bincount(galaxyidx, weights=galaxyX, minlength=Ngrps) / nmembers
    Ycen = np.bincount(galaxyidx, weights=galaxyY, minlength=Ngrps) / nmembers
    Zcen = np.bincount(galaxyidx, weights=galaxyZ, minlength=Ngrps) / nmembers
    zcen = np.sqrt(Xcen*Xcen + Ycen*Ycen + Zcen*Zcen)
    racen = (np.degrees(np.arctan2(Ycen,Xcen))+360.) % 360.0
    deccen = np.degrees(np.arcsin(Zcen / zcen))
    groupra = racen[galaxyidx]
    groupdec = deccen[galaxyidx]
    return groupra, groupdec, groupz

@njit(parallel=True)
def get_group_redshift(galaxyz, galaxyzerr, galaxygrpid, uniqgrpid, grpn, zmin, zmax, zerrmin, zerrmax, dz, z_grp):
    norm = 1 / (np.sqrt(2*np.pi) * galaxyzerr)
    inverr = 1 / galaxyzerr
    inv_den2 = -0.5 * inverr * inverr
    ngrps = len(uniqgrpid)
    for i in prange(ngrps):
        if grpn[i] > 1:
            gg = uniqgrpid[i]
            grpsel = np.where(galaxygrpid == gg)
            zgrid = np.arange(zmin[i]-4*zerrmax[i], zmax[i]+4*zerrmax[i], dz[i])
            pz = np.zeros_like(zgrid)
            galzvalues = galaxyz[grpsel]
            deltaz = galzvalues.reshape(galzvalues.shape[0],1) - zgrid
            deltaz2 = deltaz * deltaz
            for j,k in enumerate(grpsel[0]):
                pz += (norm[k] * np.exp(inv_den2[k] * deltaz2[j]))
            z_grp[i] = np.average(zgrid, weights=pz)
    return z_grp

def cartesian_from_spherical_z(ra, dec, redshift):
    """
    Convert RA/Dec/z to comoving (x,y,z).

    Parameters
    ----------------
    ra : array_like
        RA in decimal degrees.
    dec : array_like
        Decl. in decimal degrees.
    redshift : array_like
        Redshift (z).
    cosmo:
        Astropy cosmology object (specifies
        comoving distance formulation).

    Returns
    ----------------
    XX, YY, ZZ : array_like
        Cartesian coordinates in comoving Mpc
        according to the input cosmology.
    """
    phi = np.deg2rad(ra)
    theta = np.pi/2. - np.deg2rad(dec)
    XX = redshift * np.sin(theta) * np.cos(phi)
    YY = redshift * np.sin(theta) * np.sin(phi)
    ZZ = redshift * np.cos(theta)
    return XX, YY, ZZ

def comoving_cartesian_from_spherical(ra, dec, redshift, cosmo):
    """ 
    Convert RA/Dec/z to comoving (x,y,z).

    Parameters
    ----------------
    ra : array_like
        RA in decimal degrees.
    dec : array_like
        Decl. in decimal degrees.
    redshift : array_like
        Redshift (z).
    cosmo:
        Astropy cosmology object (specifies
        comoving distance formulation).

    Returns
    ----------------
    XX, YY, ZZ : array_like
        Cartesian coordinates in comoving Mpc
        according to the input cosmology.
    """
    phi = np.deg2rad(ra)
    theta = np.pi/2. - np.deg2rad(dec) 
    dc = cosmo.comoving_distance(redshift).value
    XX = dc * np.sin(theta) * np.cos(phi)
    YY = dc * np.sin(theta) * np.sin(phi)
    ZZ = dc * np.cos(theta)
    return XX, YY, ZZ

@vectorize(['float64(float64)'])
def erf_vec(x):
    return erf(x)

@njit
def get_pz_group(zgrid, zz, norm, invden2):
    dz = zgrid.reshape(zgrid.shape[0],1) - zz
    pz = np.sum(norm * np.exp(invden2 * dz * dz), axis=1)
    return pz

@njit
def dbint_pz_general(zgrid, pz1, pz2, eps):
    cum_D2 = np.zeros(len(zgrid))
    cum_D2[1:] = np.cumsum(pz2[:-1] * np.diff(zgrid))
    lower = np.searchsorted(zgrid, zgrid - eps, side='left')
    upper = np.searchsorted(zgrid, zgrid + eps, side='right')
    lower = numba_clip(lower, 0, len(zgrid)-1)
    upper = numba_clip(upper, 0, len(zgrid)-1)
    f_of_z = cum_D2[upper] - cum_D2[lower]
    P12 = np.sum(pz1 * f_of_z * (zgrid[1]-zgrid[0]))
    return P12

@njit
def dbint_pz_jgauss(zgrid, pz1, z2, zerr2, eps):
    """ for when p(z|z_j, zerr_j) is Gaussian """
    den = 1.41421356 * zerr2
    erf_term = erf_vec((z2 - zgrid + eps)/den) - erf_vec((z2 - zgrid - eps)/den)
    P12 = np.sum(0.5 * pz1 * erf_term * (zgrid[1]-zgrid[0]))
    return P12

@njit
def numba_clip(arr, a_min, a_max):
    # This matches the behavior of np.clip(arr, a_min, a_max)
    return np.minimum(a_max, np.maximum(arr, a_min))

@njit
def get_adaptive_zgrid(z1, z2, e1, e2, npts=5):
    zmin = min([z1.min(),z2.min()])
    zmax = max([z1.max(),z2.max()])
    emin = min([e1.min(),e2.min()])
    emax = max([e1.max(),e2.max()])
    dz = emin / npts
    buff = 4 * emax
    grid = np.arange(zmin - buff, zmax + buff, dz)
    return grid

@njit(parallel=True)
def integrate_IC(idx_to_integrate, galaxyz, galaxyzerr, grpid, gauss_norm, invden2, uniqgrpid, nnind, seedN, eps, n_pts_per_sigma):
    prob = np.zeros(len(uniqgrpid))
    for k in prange(len(idx_to_integrate)):
        ii = idx_to_integrate[k]
        jj = nnind[ii]
        grp_ii = uniqgrpid[ii]
        grp_jj = uniqgrpid[jj]
        grp_ii_sel = np.where(grpid == grp_ii)
        grp_jj_sel = np.where(grpid == grp_jj)
        z_ii = galaxyz[grp_ii_sel]
        z_jj = galaxyz[grp_jj_sel]
        zerr_ii = galaxyzerr[grp_ii_sel]
        zerr_jj = galaxyzerr[grp_jj_sel]
        zgrid = get_adaptive_zgrid(z_ii, z_jj, zerr_ii, zerr_jj, npts=n_pts_per_sigma)
        pz_ii = get_pz_group(zgrid, z_ii, gauss_norm[grp_ii_sel], invden2[grp_ii_sel])
        if seedN[jj]==1:
            prob[ii] = dbint_pz_jgauss(zgrid, pz_ii, z_jj, zerr_jj, eps[ii])
        else:
            pz_jj = get_pz_group(zgrid, z_jj, gauss_norm[grp_jj_sel], invden2[grp_jj_sel])
            prob[ii] = dbint_pz_general(zgrid, pz_ii, pz_jj, eps[ii])
    return prob

@njit                                                      
def angular_separation(ra1,dec1,ra2,dec2):
    """ 
    Compute the angular separation between two lists of galaxies using the Haversine formula.
    
    Parameters
    ------------
    ra1, dec1, ra2, dec2 : array-like
       Lists of right-ascension and declination values for input targets, in decimal degrees. 
    
    Returns
    ------------
    angle : np.array
       Array containing the angular separations between coordinates in list #1 and list #2, as above.
       Return value expressed in radians, NOT decimal degrees.
    """
    phi1 = np.deg2rad(ra1)
    phi2 = np.deg2rad(ra2)
    theta1 = np.pi/2. - np.deg2rad(dec1)
    theta2 = np.pi/2. - np.deg2rad(dec2)
    sin_dt = np.sin((theta2-theta1)/2.0)
    sin_dp = np.sin((phi2 - phi1)/2.0)
    return 2*np.arcsin(np.sqrt(sin_dt*sin_dt + np.sin(theta1)*np.sin(theta2) * (sin_dp*sin_dp)))

def multiplicity_function(grpids, return_by_galaxy=False):
    """
    Obtain the number of galaxies in each host group.

    Parameters
    ----------
    grpids : iterable
        List of group ID numbers. Length must match # galaxies.
    Returns
    -------
    occurences : list
        Number of galaxies in each galaxy group (length matches # groups).
    """
    grpids=np.asarray(grpids)
    uniqid, inv, counts = np.unique(grpids, return_counts=True, return_inverse=True)
    if return_by_galaxy:
        return counts[inv]
    else:
        return counts

def get_int_mag(galmag, grpid):
    """
    Given a list of galaxy absolute magnitudes and group ID numbers,
    compute group-integrated total magnitudes.

    Parameters
    ------------
    galmag : iterable
       List of absolute magnitudes for every galaxy (SDSS r-band).
    grpid : iterable
       List of group ID numbers for every galaxy.

    Returns
    ------------
    Mint_grp : np array
       Array containing group-integrated magnitudes for each galaxy. Length matches `galmag`.
    """
    galmag=np.asarray(galmag)
    grpid=np.asarray(grpid)
    grpmags = np.zeros(len(galmag))
    uniqgrpid, galaxyidx = np.unique(grpid, return_inverse=True)
    Ngroups = len(uniqgrpid)
    # Mint_grp = -2.5 * log10(Sum[10 ^ -0.4M]) for galaxy abs mag M.
    Mi_terms = np.power(10, -0.4*galmag)
    sum_terms = np.bincount(galaxyidx, weights=Mi_terms, minlength=Ngroups)
    Mint_grp = -2.5 * np.log10(sum_terms)
    return Mint_grp[galaxyidx]

def central_flag(grpid, galproperty):
    grpid = np.asarray(grpid)
    galproperty = np.asarray(galproperty)
    if (galproperty<=0).all():
        # absolute magnitude
        galproperty = galproperty * -1
    elif (galproperty>=0).all():
        # mass
        pass
    else:
        print(f"Could not determine if `galproperty` is a mass or absolute magnitude.")

# ------------------------------------------------------------------------------ #
# Misc. supporting functions
# ------------------------------------------------------------------------------ #
def center_binned_stats(*args, **kwargs):
    """ 
     Same as scipy.stats.binned_statistic, but returns
     the bin centers (matching length of `statistic`)
     instead of the binedges.

     See docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binned_statistic.html
    """
    stat, binedges, binnumber = binned_statistic(*args,**kwargs)
    bincenters = (binedges[:-1]+binedges[1:])/2.
    return stat, bincenters, binedges, binnumber

def sigmarange(x):
    q84, q16 = np.percentile(x, [84 ,16])
    return (q84-q16)/2.

def giantmodel(x, a, b):
    return np.abs(a)*np.log10(np.abs(b)*x+1)

def decayexp(x, a, b):
    return np.abs(a)*np.exp(-1*np.abs(b)*x)

def smoothedbootstrap(data, n_bootstraps, user_statistic, kwargs=None, random_state=None):
    """Compute smoothed bootstrapped statistics of a data set.
    Parameters
    ----------
    data : array_like
        A 1-dimensional data array of size n_samples
    n_bootstraps : integer
        the number of bootstrap samples to compute.  Note that internally,
        two arrays of size (n_bootstraps, n_samples) will be allocated.
        For very large numbers of bootstraps, this can cause memory issues.
    user_statistic : function
        The statistic to be computed.  This should take an array of data
        of size (n_bootstraps, n_samples) and return the row-wise statistics
        of the data.
    kwargs : dictionary (optional)
        A dictionary of keyword arguments to be passed to the
        user_statistic function.
    random_state: RandomState or an int seed (0 by default)
        A random number generator instance
    Returns
    -------
    distribution : ndarray
        the bootstrapped distribution of statistics (length = n_bootstraps)
    """
    # we don't set kwargs={} by default in the argument list, because using
    # a mutable type as a default argument can lead to strange results
    if kwargs is None:
        kwargs = {}

    rng = check_random_state(random_state)

    data = np.asarray(data)
    n_datapts = data.size

    if data.ndim != 1:
        raise ValueError("bootstrap expects 1-dimensional data")

    # Generate random indices with repetition
    ind = rng.randint(n_datapts, size=(n_bootstraps, n_datapts))

    # smoothing noise
    noisemean = 0.
    noisesigma = np.std(data,ddof=1) / np.sqrt(n_datapts)
    noise = np.random.normal(noisemean,noisesigma,(n_bootstraps, n_datapts))
    databroadcast = data[ind] + noise

    # Call the function
    stat_bootstrap = user_statistic(databroadcast, **kwargs)

    # compute the statistic on the data
    return stat_bootstrap
