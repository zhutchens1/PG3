import numpy as np
from numba import njit, vectorize, float64, prange
from math import erf

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

@vectorize(['float64(float64)'])
def erf_vec(x):
    return erf(x)

if __name__=='__main__':
    import pandas as pd
    eco = pd.read_csv("/srv/one/zhutchen/g3groupfinder/resolve_and_eco/ECOdata_G3catalog_luminosity.csv")
    eco = eco[(eco.absrmag<=-17.33) & (eco.g3grpcz_l>3000) & (eco.g3grpcz_l<7000)]
    ra, dec, z = prob_group_skycoords(eco.radeg, eco.dedeg, eco.cz/3e5, 0*eco.cz + 50/3e5, eco.g3grp_l, 5)

    #import matplotlib.pyplot as plt
    #comasel = (eco.g3grp_l==14)
    #plt.figure()
    #plt.plot(eco[comasel].radeg, eco[comasel].dedeg, '.', color='gray') 
    #plt.plot(eco[comasel].g3grpradeg_l, eco[comasel].g3grpdedeg_l, '.', color='blue')
    #plt.plot(ra[comasel], dec[comasel], 'x', color='red')
    #plt.show()

    #plt.figure()
    #plt.plot(eco[comasel].cz, eco[comasel].dedeg, '.', color='gray') 
    #plt.plot(eco[comasel].g3grpcz_l, eco[comasel].g3grpdedeg_l, '.', color='blue')
    #plt.plot(3e5*z[comasel], dec[comasel], 'x', color='red')
    #plt.show()
        
    
    
