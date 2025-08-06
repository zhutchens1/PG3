from astropy.cosmology import LambdaCDM
from scipy.stats import binned_statistic
import numpy as np

def comoving_volume(ra1,ra2,dec1,dec2,z1,z2,H0,Om0,Ode0):
    """
    Compute the comoving volume of a lat/lon rectangle on-sky,
    bounded by (ra1,dec1)->(ra2,dec2), and extended over the 
    redshift range (z1,z2).

    Parameters
    -----------------
    ra1 : scalar
        Lower RA in decimal degrees.
    ra2 : scalar
        Upper RA in decimal degrees.
    dec1 : scalar
        Lower declination in decimal degrees.
    dec2 : scalar
        Upper declination in decimal degrees.
    z1 : scalar
        Inner redshift.
    z2 : scalar
        Outer redshift.
    H0 : scalar
        Hubble constant for LambdaCDM cosmology.
    Om0 : scalar
        Density of non-relativistic matter in units of critical
        density at z=0.
    Ode0 : scalar
        Density of dark matter at z=0 in units of critical density.

    Returns
    ------------------
    volume : scalar
        Volume of survey spanned by input coordinates.
        Units of Mpc^3.
    """
    cosmo = LambdaCDM(H0,Om0,Ode0)
    dtor=np.pi/180.
    if ra2>=ra1:
        delta_ra_rad = (ra2-ra1)*dtor
    else:
        delta_ra_rad = ((360.-ra1)+ra2)*dtor
    delta_dec_rad = np.sin(dec2*dtor)-np.sin(dec1*dtor)
    solidangle = delta_ra_rad * delta_dec_rad
    dv = cosmo.comoving_volume(z2).value - cosmo.comoving_volume(z1).value
    return (solidangle/(4*np.pi)) * dv


def comoving_volume_shell(z1,z2,H0,Om0,Ode0):
    """
    Compute the comoving volume per steradian of a spherical
    shell with inner redshift `z1` and outer radius `z2`. To
    determine the comoving volume associated with some solid
    angle A on sky, compute A*comoving_volume_per_skyarea(*).
    This function allows for generalization of the `comoving_
    volume` to surveys that do not carve out lat/lon rectang-
    les on sky.

    Parameters
    -----------------
    z1 : scalar
        Inner redshift.
    z2 : scalar
        Outer redshift.
    H0 : scalar
        Hubble constant for LambdaCDM cosmology.
    Om0 : scalar
        Density of non-relativistic matter in units of critical
        density at z=0.
    Ode0 : scalar
        Density of dark matter at z=0 in units of critical density.

    Returns
    ------------------
    volume : scalar
        Volume of shell spanned by the two input redshifts, with
        units of (Mpc)^3/sr.
    """
    cosmo = LambdaCDM(H0,Om0,Ode0)
    dv = cosmo.comoving_volume(z2).value - cosmo.comoving_volume(z1).value
    return (1/(4*np.pi)) * dv


def integrate_volume(redshifts, solid_angle, H0, Om0, Ode0):
    """
    Compute the comoving volume for a survey
    whose field-of-view changes with redshift.
    This function integrates A(z)W(z)dz, where
    A(z) is the solid angle in str, W(z) is the
    differential comoving volume per z per str.

    Parameters
    ---------------------
    redshifts : array_like
        Redshifts spanned by the volume. The spacing
        between elements determines the value of dz.
        Example: np.linspace(0.1,0.2,1000)
        Here, dz=(0.2-0.1)/1000 = 1E-4.
    solid_angle : array_like
        Solid angle at each `z` in redshifts. Length
        should match `redshifts` and its units should
        be expressed in steradians.
    
    Returns
    ---------------------
    volume : scalar
        Volume spanned by the input redshift and
        solid angle, in units Mpc. 
    """
    cosmo = LambdaCDM(H0,Om0,Ode0)
    vol_per_zstr = cosmo.differential_comoving_volume(redshifts)
    integrand = (solid_angle)*vol_per_zstr
    return np.sum(integrand*(redshifts[1]-redshifts[0])).value

def solid_angle(ra, dec, bins):
    ramax, bin_edges, _ = binned_statistic(dec, ra,'max', bins=bins)
    ramin, _, _ = binned_statistic(dec, ra, 'min', bins=bins)
    ddec = bin_edges[1:]-bin_edges[:-1]
    decbincenters = 0.5*(bin_edges[1:]+bin_edges[:-1])
    if (ddec>1).any(): print("sold_angle warning: declination slices are greater than 1 degree")
    dtr=np.pi/180.
    ramax=ramax*dtr
    ramin=ramin*dtr
    ddec=ddec*dtr
    decbincenters=decbincenters*dtr
    omega = np.sum((ramax-ramin)*ddec*np.cos(decbincenters)) # str
    if np.isnan(omega): print("Solid angle estimate is NaN. Increase bin size.")
    return omega*3282.8 # sq deg/str


if __name__=='__main__':
    #print(comoving_volume(130.05,237.45,-1,49.85,2530/3e+5,7470/3e5,100,0.3,0.7))
    #x=integrate_volume(np.linspace(2530/3e5,7470/3e5,int(1e4)),np.full(int(1e4),1.465492683585301),100.,0.3,0.7)
    SA = 1.456
    SA = solid_angle(np.random.uniform(130.06,237.445,1000000),np.random.uniform(-1,49.85,1000000),bins=500) / 3282.8
    x=integrate_volume(np.linspace(0.01,0.023333,int(1e4)),np.full(int(1e4),SA),100.,0.3,0.7)
    print(SA, x)
    #print(comoving_volume(130.05,237.45,-1,49.85,3000/3e+5,7000/3e5,100,0.3,0.7))
    #x=integrate_volume(np.linspace(3000/3e5,7000/3e5,int(1e4)),np.full(int(1e4),1.465492683585301),100.,0.3,0.7)
    #print(x)
