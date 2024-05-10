import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import foftools as fof
import iterativecombination as ic
from astropy.cosmology import LambdaCDM, z_at_value
import astropy.units as uu
from scipy.interpolate import interp1d 
from scipy.optimize import curve_fit
#from smoothedbootstrap import smoothedbootstrap as sbs
#from giantonlyic import iterative_combination_giants
from numba import njit
from scipy.integrate import quad, simpson
import math

SPEED_OF_LIGHT = 2.998e5

def sigmarange(x):
    q84, q16 = np.percentile(x, [84 ,16])
    return (q84-q16)/2.

def giantmodel(x, a, b):
    return np.abs(a)*np.log10(np.abs(b)*x+1)

def decayexp(x, a, b):
    return np.abs(a)*np.exp(-1*np.abs(b)*x)#+np.abs(d)

def g3groupfinder_luminosity(radeg,dedeg,cz,czerr,absrmag,dwarfgiantdivide,fof_bperp=0.07,fof_blos=1.1,fof_sep=None, volume=None, center_mode='average',\
                 iterative_giant_only_groups=False, n_bootstraps=10000, rproj_fit_guess=None, rproj_fit_params = None, rproj_fit_multiplier=None,\
                 vproj_fit_guess = None, vproj_fit_params = None, vproj_fit_multiplier=None, vproj_fit_offset=0, gd_rproj_fit_guess=None, gd_rproj_fit_params = None,\
                 gd_rproj_fit_multiplier=None, gd_vproj_fit_guess=None, gd_vproj_fit_params = None, gd_vproj_fit_multiplier=None,gd_vproj_fit_offset=None,
                 gd_fit_bins=None,ic_center_mode='arithmetic', ic_decision_mode='centers',H0=100., Om0=0.3, Ode0=0.7, showplots=False, saveplotspdf=False):
    """
    Identify galaxy groups in redshift space using the RESOLVE-G3 algorithm (Hutchens et al. 2022).

    Parameters
    -------------------
    radeg : array_like
        Right ascension of input galaxies in decimal degrees.
    dedeg : array_like
        Declination of input galaxies in decimal degrees.
    cz : array_like
        Recessional velocities of input galaxies in decimal degrees.
    czerr : array_like
        1-sigma errors on cz
    absrmag : array_like
        Absolute magnitude for galaxies, used to select giants vs. dwarfs.
    dwarfgiantdivide : float
        Value that will divide giants and dwarfs.
    fof_bperp : float
        Perpendicular FoF linking length, default 0.07.
    fof_blos : float
        Line-of-sight FoF linking length, default 1.1.
    fof_sep : float
        Mean galaxy separation used for FoF. Should be expressed in units of (Mpc/h) with 
        h corresponding to the `H0` argument (i.e. use h=0.7 if setting H0=70.). If None
        (default), fof_sep will be determined using the number of galaxies and `volume`.
    volume : float
        Group finding volume in (Mpc/h)^3 with h corresponding to the `H0` argument, default
        None. This argument is unnecessary if fof_sep is provided. `fof_sep` and `volume`
        cannot both be `None`.
    center_mode : str
        Specifies how group centers for giant-hosting groups should be computed when iteratively
        combining giant-only FoF groups, or associating dwarfs to giant-only groups. 
        Can be 'average', 'giantaverage', 'BCG', or a two-element tuple. If a tuple, a group center
        is calculated that smoothly varies between the average and BCG. The elements of the tuple
        set the steepness and critical transition (in N_galaxies) of the sigmoid.
    iterative_giant_only_groups : bool 
        If False (default), giant-only groups are determined with a single run of FoF.
        If True, giant-only groups are determined iteratively, starting with FoF and refining
        based on iteratively-updated group boundaries.
    n_bootstraps : int
        Number of bootstraps to perform when computing errors on medians, default 10,000.
    rproj_fit_guess : iterable
        Guess supplied to scipy.optimize.curve_fit when fitting rproj,gal vs. N_giants.
    rproj_fit_params : iterable
        Parameters to use when associating dwarfs and/or iteratively combining giant-only groups.
        If this parameter is passed, then the fit to rproj,gal vs. N_giants is not performed.
    rproj_fit_multiplier : float
        Scalar multiplier for rproj_fit.
    vproj_fit_guess : iterable
        Guess supplied to scipy.optimize.curve_fit when fitting rproj,gal vs. N_giants.
    vproj_fit_params : iterable
        Parameters to use when associating dwarfs and/or iteratively combining giant-only groups.
        If this parameter is passed, then the fit to vproj,gal vs. N_giants is not performed.
    vproj_fit_multiplier : float
        Scalar multiplier for vproj_fit.
    vproj_fit_offset : float
        Vertical offset to fitted boundary model for giant-only merging and dwarf association.
        i.e. association boundary of vproj_fit_multiplier * model(Ngiant) + vproj_fit_offset.
        Units: km/s (default 0 km/s)
    gd_rproj_fit_guess : iterable
        Guess supplied to scipy.optimize.curve_fit when fitting gdrproj,gal vs. Ltot.
    gd_rproj_fit_params : iterable
        Parameters to use when iteratively combining dwarf-only seed groups.
        If this parameter is passed, then the fit to gdrproj,gal vs. Ltot is not performed.
    gd_rproj_fit_multiplier : float
        Scalar multiplier of gd_rproj_fit for use in dwarf-only group finding.
    gd_vproj_fit_guess : iterable
        Guess supplied to scipy.optimize.curve_fit when fitting gd_vproj,gal vs. Ltot.
    gd_vproj_fit_params : iterable
        Parameters to use for iterative combination dwarf-only groups.
        If this parameter is passed, then the fit to gd_vproj,gal vs. N_giants is not performed.
    gd_vproj_fit_multiplier : float
        Scalar multiplier of gd_vproj_fit for use in dwarf-only group finding.
    gd_vproj_fit_offset : float
        Vertical offset to fitted boundary model for dwarf-only group finding.
        i.e. boundary of gd_vproj_fit_multiplier * model(group Lr) + gd_vproj_fit_offset
    gd_fit_bins : iterable
        Array of bin edges for binning and fitting properties of giant+dwarf groups prior to
        dwarf-only group finding. 
    ic_center_mode : str
        Mode of computing group center in dwarf-only group finding mode (default `arithmetic`).
    ic_decision_mode : str
        Mode of deciding whether to merge giant-only or dwarf-only seed groups. Default `centers`, which
        evaluates whether seed group centers are close enough. If `allgalaxies`, all galaxies
        must within the specified boundaries.
    H0 : float
        z=0 Hubble constant in units of (km/s)/Mpc, default 100.0. Return parameters will
        be consistent with this choice.
    showplots : False
        If True, plots are rendered using matplotlib at each group finding step.
    saveplotspdf : False
        If True, rendered plots will be saved in a ./figures/ subfolder.

    Returns
    -------------------------
    g3grpid : np.array
        Group ID number for each galaxy from G3 algorithm.
    g3ssid : np.array
        Group substructure ID number for each galaxy. Equals `g3grpid` if 
        `iterative_giant_only_groups` is set to `False`.
    fof_sep : float
        Mean galaxy separation in FoF, if volume was provided, in units of Mpc/h
        with h matching H0 parameter. 
    rproj_bestfit : np.array
        Best-fitting values to rproj,gal vs. Ngiants, matches `rproj_fit_params` if provided.
    rproj_bestfit_err : np.array
        Errors on best-fitting values to rproj,gal vs. Ngiants, None if `rproj_fit_params` was provided.
    vproj_bestfit : np.array
        Best-fitting values to vproj,gal vs. Ngiants, matches `vproj_fit_params` if provided.
    vproj_bestfit_err : np.array
        Errors on best-fitting values to vproj,gal vs. Ngiants, None if `vproj_fit_params` was provided.
    gd_rproj_bestfit : np.array
        Best-fitting values to rproj,gal vs. Ltot, matches `gd_rproj_fit_params if provided.
    gd_rproj_bestfit_err : np.array
        Best-fitting values to rproj,gal vs. Ltot, None if `gd_rproj_fit_params was provided.
    gd_vproj_bestfit : np.array
        Best-fitting values to vproj,gal vs. Ltot, matches `gd_vproj_fit_params if provided.
    gd_vproj_bestfit_err : np.array
        Best-fitting values to vproj,gal vs. Ltot, None if `gd_vproj_fit_params was provided.
    """
    ### prepare arrays ---------------------------- #
    radeg=np.array(radeg)
    dedeg=np.array(dedeg)
    cz=np.array(cz)
    czerr=np.array(czerr)
    absrmag=np.array(absrmag)
    g3grpid = np.zeros_like(radeg)-99.
    g3ssid = np.zeros_like(radeg)-99.
    cosmo = LambdaCDM(H0=H0,Om0=Om0,Ode0=Ode0)
    SPEED_OF_LIGHT=2.998e+5
    ### giant-only FoF ----------------- # 
    giantsel = (absrmag<=dwarfgiantdivide)
    if fof_sep is not None:
        giantfofid = pfof(radeg[giantsel], dedeg[giantsel], cz[giantsel], czerr[giantsel], fof_bperp*fof_sep,) # << fix here 
        #giantfofid = fof.fast_fof(radeg[giantsel],dedeg[giantsel],cz[giantsel],fof_bperp,fof_blos,fof_sep,H0=H0,Om0=Om0,Ode0=Ode0)
    else:
        fof_sep = (volume/np.sum(giantsel))**(1/3.)
        giantfofid = fof.fast_fof(radeg[giantsel],dedeg[giantsel],cz[giantsel],fof_bperp,fof_blos,fof_sep,H0=H0,Om0=Om0,Ode0=Ode0)
    g3grpid[giantsel] = giantfofid

    ### if values not passed, fit rproj and vproj vs. N_giants
    if (rproj_fit_params is None) or (vproj_fit_params is None):
        if center_mode=='average' or center_mode=='giantaverage':
            giantgrpra, giantgrpdec, giantgrpcz = fof.group_skycoords(radeg[giantsel], dedeg[giantsel], cz[giantsel], giantfofid)
        elif center_mode=='BCG':
            giantgrpra, giantgrpdec, giantgrpcz = agc.BCG_skycoords(radeg[giantsel], dedeg[giantsel], cz[giantsel], absrmag[giantsel], giantfofid)
        elif (type(center_mode) is tuple):
            assert len(center_mode)==2,"Logistic skycoord tuple requires two parameters."
            giantgrpra, giantgrpdec, giantgrpcz = agc.logistic_skycoords(radeg[giantsel], dedeg[giantsel], cz[giantsel], absrmag[giantsel],\
                 giantfofid,center_mode[0],center_mode[1])
        relvel = np.abs(giantgrpcz - cz[giantsel])
        grp_ctd = cosmo.comoving_transverse_distance(giantgrpcz/SPEED_OF_LIGHT).value
        gia_ctd = cosmo.comoving_transverse_distance(cz[giantsel]/SPEED_OF_LIGHT).value
        relprojdist = (grp_ctd+gia_ctd)*np.sin(ic.angular_separation(giantgrpra, giantgrpdec, radeg[giantsel], dedeg[giantsel])/2.0)
        giantgrpn = fof.multiplicity_function(giantfofid, return_by_galaxy=True)
        uniqgiantgrpn, uniqindex = np.unique(giantgrpn, return_index=True)
        keepcalsel = np.where(uniqgiantgrpn>1)
        median_relprojdist = np.array([np.median(relprojdist[np.where(giantgrpn==sz)]) for sz in uniqgiantgrpn[keepcalsel]])
        median_relvel = np.array([np.median(relvel[np.where(giantgrpn==sz)]) for sz in uniqgiantgrpn[keepcalsel]])
        rproj_median_error = np.std(np.array([sbs(relprojdist[np.where(giantgrpn==sz)], n_bootstraps, np.median, kwargs=dict({'axis':1 })) for sz in uniqgiantgrpn[keepcalsel]]), axis=1)
        dvproj_median_error = np.std(np.array([sbs(relvel[np.where(giantgrpn==sz)], n_bootstraps, np.median, kwargs=dict({'axis':1})) for sz in uniqgiantgrpn[keepcalsel]]), axis=1)
    if rproj_fit_params is None:    
        rproj_bestfit, rproj_bestfit_cov = curve_fit(giantmodel, uniqgiantgrpn[keepcalsel], median_relprojdist, sigma=rproj_median_error, p0=rproj_fit_guess)
        rproj_bestfit_err = np.sqrt(np.diag(rproj_bestfit_cov))
    else:
        rproj_bestfit = np.array(rproj_fit_params)
        rproj_bestfit_err = np.zeros(2)*1.
    if vproj_fit_params is None:
        vproj_bestfit, vproj_bestfit_cov  = curve_fit(giantmodel, uniqgiantgrpn[keepcalsel], median_relvel, sigma=dvproj_median_error, p0=vproj_fit_guess)
        vproj_bestfit_err = np.sqrt(np.diag(vproj_bestfit_cov))
    else:
        vproj_bestfit = np.array(vproj_fit_params)
        vproj_bestfit_err = np.zeros(2)*1.
    
    rproj_boundary = lambda Ngiants: rproj_fit_multiplier*giantmodel(Ngiants, *rproj_bestfit)
    vproj_boundary = lambda Ngiants: vproj_fit_multiplier*giantmodel(Ngiants, *vproj_bestfit) + vproj_fit_offset
    ### if requested, merge giant-only FoF groups through iterative combination
    if iterative_giant_only_groups:
        revisedgiantgrpid = iterative_combination_giants(radeg[giantsel],dedeg[giantsel],cz[giantsel],giantfofid,rproj_boundary,vproj_boundary,ic_decision_mode,H0)
        g3ssid[giantsel] = giantfofid
        g3grpid[giantsel] = revisedgiantgrpid
    else:
        pass

    ### associate dwarfs to giant-only groups
    dwarfsel = (absrmag>dwarfgiantdivide)
    if center_mode=='average' or center_mode=='giantaverage':
        giantgrpra, giantgrpdec, giantgrpcz = fof.group_skycoords(radeg[giantsel], dedeg[giantsel], cz[giantsel], g3grpid[giantsel])
    elif center_mode=='BCG':
        giantgrpra, giantgrpdec, giantgrpcz = agc.BCG_skycoords(radeg[giantsel], dedeg[giantsel], cz[giantsel], absrmag[giantsel], g3grpid[giantsel])
    elif (type(center_mode) is tuple):
        assert len(center_mode)==2,"Logistic skycoord tuple requires two parameters."
        giantgrpra, giantgrpdec, giantgrpcz = agc.logistic_skycoords(radeg[giantsel], dedeg[giantsel], cz[giantsel], absrmag[giantsel],\
             giantfofid,center_mode[0],center_mode[1])

    giantgrpn = fof.multiplicity_function(g3grpid[giantsel],return_by_galaxy=True)
    dwarfassocid, _ = fof.fast_faint_assoc(radeg[dwarfsel],dedeg[dwarfsel],cz[dwarfsel],giantgrpra,giantgrpdec,giantgrpcz,g3grpid[giantsel],\
        rproj_boundary(giantgrpn),vproj_boundary(giantgrpn), H0=H0,Om0=Om0,Ode0=Ode0)
    g3grpid[dwarfsel]=dwarfassocid

    ### if values not passed, fit rproj and vproj for giants+dwarfs vs. Ltot
    if (gd_rproj_fit_params is None) or (gd_vproj_fit_params is None):
        gdgrpn = fof.multiplicity_function(g3grpid, return_by_galaxy=True)
        gdsel = np.logical_not(np.logical_or(g3grpid==-99., ((gdgrpn==1) & (absrmag>dwarfgiantdivide))))
        gdgrpra, gdgrpdec, gdgrpcz = fof.group_skycoords(radeg[gdsel], dedeg[gdsel], cz[gdsel], g3grpid[gdsel])
            
        gdrelvel = np.abs(gdgrpcz - cz[gdsel])
        ctd1 = cosmo.comoving_transverse_distance(gdgrpcz/SPEED_OF_LIGHT).value
        ctd2 = cosmo.comoving_transverse_distance(cz[gdsel]/SPEED_OF_LIGHT).value
        gdrelprojdist = (ctd1 + ctd2) * np.sin(ic.angular_separation(gdgrpra, gdgrpdec, radeg[gdsel], dedeg[gdsel])/2.0)
        #gdrelprojdist = (gdgrpcz + cz[gdsel])/H0 * np.sin(ic.angular_separation(gdgrpra, gdgrpdec, radeg[gdsel], dedeg[gdsel])/2.0)
        gdn = gdgrpn[gdsel]
        gdtotalmag = ic.get_int_mag(absrmag[gdsel], g3grpid[gdsel])
        binsel = np.where(np.logical_and(gdn>1, gdtotalmag>-24))#np.min(gd_fit_bins))) # test here
        gdmedianrproj, magbincenters, agbinedges, jk = center_binned_stats(gdtotalmag[binsel], gdrelprojdist[binsel], np.median, bins=gd_fit_bins)
        gdmedianrproj_err, jk, jk, jk = center_binned_stats(gdtotalmag[binsel], gdrelprojdist[binsel], sigmarange, bins=gd_fit_bins)
        gdmedianrelvel, jk, jk, jk = center_binned_stats(gdtotalmag[binsel], gdrelvel[binsel], np.median, bins=gd_fit_bins)
        gdmedianrelvel_err, jk, jk, jk = center_binned_stats(gdtotalmag[binsel], gdrelvel[binsel], sigmarange, bins=gd_fit_bins)
        nansel = np.isnan(gdmedianrproj)
    if (gd_rproj_fit_params is None):
        gd_rproj_bestfit, gd_rproj_cov=curve_fit(decayexp, magbincenters[~nansel], gdmedianrproj[~nansel], p0=gd_rproj_fit_guess)
        gd_rproj_bestfit_err = np.sqrt(np.diag(gd_rproj_cov))
    else:
        gd_rproj_bestfit = np.array(gd_rproj_fit_params)
        gd_rproj_bestfit_err = np.zeros(len(gd_rproj_fit_params))*1.
    if (gd_vproj_fit_params is None):
        gd_vproj_bestfit, gd_vproj_cov=curve_fit(decayexp, magbincenters[~nansel], gdmedianrelvel[~nansel], p0=gd_vproj_fit_guess)
        gd_vproj_bestfit_err = np.sqrt(np.diag(gd_vproj_cov))
    else:
        gd_vproj_bestfit = np.array(gd_vproj_fit_params)
        gd_vproj_bestfit_err = np.zeros(len(gd_vproj_fit_params))*1.
    rproj_for_iteration = lambda M: gd_rproj_fit_multiplier*decayexp(M, *gd_rproj_bestfit)
    vproj_for_iteration = lambda M: gd_vproj_fit_multiplier*decayexp(M, *gd_vproj_bestfit) + gd_vproj_fit_offset

    ### --------- iterative combination to make dwarf-only groups
    assert (g3grpid[(absrmag<=dwarfgiantdivide)]!=-99.).all(), "Not all giants are grouped." 
    grpnafterassoc = fof.multiplicity_function(g3grpid, return_by_galaxy=True)
    _ungroupeddwarf_sel = (absrmag>dwarfgiantdivide) & (grpnafterassoc==1)    
    itassocid = ic.iterative_combination(radeg[_ungroupeddwarf_sel], dedeg[_ungroupeddwarf_sel], cz[_ungroupeddwarf_sel], absrmag[_ungroupeddwarf_sel],\
                   rproj_for_iteration, vproj_for_iteration, starting_id=np.max(g3grpid)+1, centermethod=ic_center_mode, decisionmode=ic_decision_mode, H0=H0)
    g3grpid[_ungroupeddwarf_sel]=itassocid
    ### ------------  return quantities
    return g3grpid, g3ssid, fof_sep, rproj_bestfit, rproj_bestfit_err, vproj_bestfit, vproj_bestfit_err, gd_rproj_bestfit, gd_rproj_bestfit_err, gd_vproj_bestfit, gd_vproj_bestfit_err 


#################################################
#################################################
#################################################
def pfof_comoving(ra, dec, cz, czerr, perpll, losll, Pth, H0=100., Om0=0.3, Ode0=0.7, printConf=True):
    """
    -----
    Compute group membership from galaxies' equatorial  coordinates using a probabilitiy
    friends-of-friends (PFoF) algorithm, based on the method of Liu et al. 2008. PFoF is
    a variant of FoF (see `foftools.fast_fof`, Berlind+2006), which treats galaxies as Gaussian
    probability distributions, allowing group membership selection to account for the 
    redshift errors of photometric redshift measurements. 
    In this function, the linking length must be fixed.   
 
    Arguments:
        ra (iterable): list of right-ascesnsion coordinates of galaxies in decimal degrees.
        dec (iterable): list of declination coordinates of galaxies in decimal degrees.
        cz (iterable): line-of-sight recessional velocities of galaxies in km/s.
        czerr (iterable): errors on redshifts of galaxies in km/s.
        perpll (float): perpendicular linking length in Mpc. 
        losll (float): line-of-sight linking length in Mpc.
        Pth (float): Threshold probability from which to construct the group catalog. If None, the
            function will return a NxN matrix of friendship probabilities.
        printConf (bool, default True): bool indicating whether to print confirmation at the end.
    Returns:
        grpid (np.array): list containing unique group ID numbers for each target in the input coordinates.
                The list will have shape len(ra).
    -----
    """
    cosmo = LambdaCDM(H0=H0, Om0=Om0, Ode0=Ode0) # this puts everything in "per h" units.
    Ngalaxies = len(ra)
    ra = np.float32(ra)
    dec = np.float32(dec)
    cz = np.float32(cz)
    czerr = np.float32(czerr)
    assert (len(ra)==len(dec) and len(dec)==len(cz)),"RA/Dec/cz arrays must equivalent length."

    phi = (ra * np.pi/180.)
    theta = (np.pi/2. - dec*(np.pi/180.))
    transv_cmvgdist = (cosmo.comoving_transverse_distance(cz/SPEED_OF_LIGHT).value)
    los_cmvgdist = (cosmo.comoving_distance(cz/SPEED_OF_LIGHT).value)
    dc_upper = los_cmvgdist + losll
    dc_lower = los_cmvgdist - losll
    VL_lower = cz - SPEED_OF_LIGHT*z_at_value(cosmo.comoving_distance, dc_lower*uu.Mpc, zmin=0.0001, zmax=2, method='Bounded')
    VL_upper = SPEED_OF_LIGHT*z_at_value(cosmo.comoving_distance, dc_upper*uu.Mpc, zmin=0.0001, zmax=2, method='Bounded') - cz
    friendship = np.zeros((Ngalaxies, Ngalaxies))
    # Compute on-sky perpendicular distance
    column_phi = phi[:, None]
    column_theta = theta[:, None]
    half_angle = np.arcsin((np.sin((column_theta-theta)/2.0)**2.0 + np.sin(column_theta)*np.sin(theta)*np.sin((column_phi-phi)/2.0)**2.0)**0.5)
    column_transv_cmvgdist = transv_cmvgdist[:, None]
    dperp = (column_transv_cmvgdist + transv_cmvgdist) * half_angle # In Mpc/h
    # Compute line-of-sight probabilities
    prob_dlos=np.zeros((Ngalaxies, Ngalaxies))
    c=SPEED_OF_LIGHT
    VL_lower = VL_lower / c
    VL_upper = VL_upper / c
    for i in range(0,Ngalaxies):
        for j in range(0, i+1):
            if j<i and dperp[i][j]<=perpll:
                val = quad(pfof_integral_asym, 0, 100, args=(cz[i], czerr[i], cz[j], czerr[j], VL_upper[i], VL_lower[i]),\
                           points=np.float64([cz[i]/c-5*czerr[i]/c,cz[i]/c-3*czerr[i]/c, cz[i]/c, cz[i]/c+3*czerr[i]/c, cz[i]/c+5*czerr[i]/c]),\
                            wvar=cz[i]/c)
                prob_dlos[i][j]=val[0]
                prob_dlos[j][i]=val[0]
            elif i==j:
                prob_dlos[i][j]=1

    # Produce friendship matrix and return groups
    index = np.where(np.logical_and(prob_dlos>Pth, dperp<=perpll))
    friendship[index]=1
    assert np.all(np.abs(friendship-friendship.T) < 1e-8), "Friendship matrix must be symmetric."

    if printConf:
        print('PFoF complete!')
    return collapse_friendship_matrix(friendship)

def pfof_integral_asym(z, czi, czerri, czj, czerrj, VLupper, VLlower):
    c=SPEED_OF_LIGHT
    #return gauss(z, czi/c, czerri/c) * (0.5*math.erf((z+VL-czj/c)/((2**0.5)*czerrj/c)) - 0.5*math.erf((z-VL-czj/c)/((2**0.5)*czerrj/c)))
    return gauss(z, czi/c, czerri/c) * 0.5 * (math.erf((czj/c - z + VLlower)/(1.41421*czerrj/c)) - math.erf((czj/c - z - VLupper)/(1.41421*czerrj/c)))

def gauss(x, mu, sigma):
    """
    Gaussian function.
    Arguments:
        x - dynamic variable
        mu - centroid of distribution
        sigma - standard error of distribution
    Returns:
        PDF value evaluated at `x`
    """
    return 1/(math.sqrt(2*np.pi) * sigma) * math.exp(-1 * 0.5 * ((x-mu)/sigma) * ((x-mu)/sigma))

def collapse_friendship_matrix(friendship_matrix):
    """
    ----
    Collapse a friendship matrix resultant of a FoF computation into an array of
    unique group numbers. 
    
    Arguments:
        friendship_matrix (iterable): iterable of shape (N, N) where N is the number of targets.
            Each element (i,j) of the matrix should represent the galaxy i and galaxy j are friends,
            as determined by the FoF linking length.
    Returns:
        grpid (iterable): 1-D array of size N containing unique group ID numbers for every target.
    ----
    """
    friendship_matrix=np.array(friendship_matrix)
    Ngalaxies = len(friendship_matrix[0])
    grpid = np.zeros(Ngalaxies)
    grpnumber = 1

    for row_num,row in enumerate(friendship_matrix):
        if not grpid[row_num]:
            group_indices = get_group_ind(friendship_matrix, row_num, visited=[row_num])
            grpid[group_indices]=grpnumber
            grpnumber+=1
    return grpid

def get_group_ind(matrix, active_row_num, visited):
    """
    ----
    Recursive algorithm to form a tree of indices from a friendship matrix row. Similar 
    to the common depth-first search tree-finding algorithm, but enabling identification
    of isolated nodes and no backtracking up the resultant trees' edges. 
    
    Example: Consider a group formed of the indices [10,12,133,53], but not all are 
    connected to one another.
                
                10 ++++ 12
                +
               133 ++++ 53
    
    The function `collapse_friendship_matrix` begins when 10 is the active row number. This algorithm
    searches for friends of #10, which are #12 and #133. Then it *visits* the #12 and #133 galaxies
    recursively, finding their friends also. It adds 12 and 133 to the visited array, noting that
    #10 - #12's lone friend - has already been visited. It then finds #53 as a friend of #133,
    but again notes that #53's only friend it has been visited. It then returns the array
    visited=[10, 12, 133, 53], which form the FoF group we desired to find.
    
    Arguments:
        matrix (iterable): iterable of shape (N, N) where N is the number of targets.
            Each element (i,j) of the matrix should represent the galaxy i and galaxy j are friends,
            as determined from the FoF linking lengths.
        active_row_num (int): row number to start the recursive row searching.
        visited (int): array containing group members that have already been visited. The recursion
            ends if all friends have been visited. In the initial call, use visited=[active_row_num].
    ----
    """
    friends_of_active = np.where(matrix[active_row_num])
    for friend_ind in [k for k in friends_of_active[0] if k not in visited]:
        visited.append(friend_ind)
        visited = get_group_ind(matrix, friend_ind, visited)
    return visited


###########################################################################################
###########################################################################################
# Group Center Definition Functions 

def gauss_vectorized(x, mu, sigma):
    """
    Gaussian function.
    Arguments:
        x - dynamic variable
        mu - centroid of distribution
        sigma - standard error of distribution
    Returns:
        PDF value evaluated at `x`
    """
    return 1/(np.sqrt(2*np.pi) * sigma) * np.exp(-1 * 0.5 * ((x-mu)/sigma) * ((x-mu)/sigma))

def get_median_eCDF(xx,aa):
    """
    xx : values in distribution
    aa : PDF of data (arbitrary units)
    """
    #xx=np.array(xx)
    #aa=np.array(aa)
    cs = np.cumsum(aa)
    return xx[np.argmin(np.abs(cs-0.5*cs[-1]))]
    
def prob_group_skycoords(galaxyra, galaxydec, galaxyz, galaxyzerr, galaxygrpid, return_z_pdfs=False):
    """
    -----
    Obtain a list of group centers (RA/Dec/z) given a list of galaxy coordinates (equatorial)
    and their corresponding group ID numbers. This is based on Hutchens+2024 (paper 3) and incorporates
    the photometric redshift errors when determining group centers.
    
    Inputs (all same length)
       galaxyra : 1D iterable,  list of galaxy RA values in decimal degrees
       galaxydec : 1D iterable, list of galaxy dec values in decimal degrees
       galaxyz : 1D iterable, list of galaxy z values in km/s
       galaxyzerr : 1D iterable, list of galaxy z 
       galaxygrpid : 1D iterable, group ID number for every galaxy in previous arguments.
       return_z_pdfs: True/False (default False), dictates whether group z PDFs are returned.
    
    Outputs (all shape match `galaxyra`)
       groupra : RA in decimal degrees of galaxy i's group center.
       groupdec : Declination in decimal degrees of galaxy i's group center.
       groupz : Redshift velocity in km/s of galaxy i's group center.
       pdfoutput: If return_z_pdfs is True, this will be returned as a dictionary
                   with keys 'zmesh', 'pdf', and 'grpid'. Otherwise `None` is returned.
    
    Note: the FoF code of AA Berlind uses theta_i = declination, with theta_cen = 
    the central declination. This version uses theta_i = pi/2-dec, with some trig functions
    changed so that the output *matches* that of Berlind's FoF code (my "deccen" is the same as
    his "thetacen", to be exact.)
    -----
    """
    # Prepare cartesian coordinates of input galaxies
    ngalaxies = len(galaxyra)
    galaxyphi = galaxyra * np.pi/180.
    galaxytheta = np.pi/2. - galaxydec*np.pi/180.
    galaxyxx = np.expand_dims((np.sin(galaxytheta)*np.cos(galaxyphi)),axis=1) # equivalent to [:,np.newaxis]
    galaxyyy = np.expand_dims((np.sin(galaxytheta)*np.sin(galaxyphi)),axis=1)
    galaxyzz = np.expand_dims(((np.cos(galaxytheta))),axis=1)
    # Prepare output arrays
    uniqidnumbers = np.unique(galaxygrpid)
    groupra = np.zeros(ngalaxies)
    groupdec = np.zeros(ngalaxies)
    groupz = np.zeros(ngalaxies)
    cspeed=3e5
    galaxyz = np.expand_dims(galaxyz,axis=1)
    galaxyzerr = np.expand_dims(galaxyzerr,axis=1) 
    for i,uid in enumerate(uniqidnumbers):
        sel=np.where(galaxygrpid==uid)
        if len(sel[0])==1:
            groupra[sel] = galaxyra[sel]
            groupdec[sel] = galaxydec[sel]
            groupz[sel] = galaxyz[sel]*cspeed
        else:
            xmesh = np.arange(np.min(galaxyz[sel]*galaxyxx[sel])-5*np.max(np.abs(galaxyzerr[sel]*galaxyxx[sel])), np.max(galaxyz[sel]*galaxyxx[sel])+5*np.max(np.abs(galaxyzerr[sel]*galaxyxx[sel])),\
                              np.min(galaxyzerr[sel])/1000.)
            xcen = get_median_eCDF(xmesh, np.sum(gauss_vectorized(xmesh, galaxyz[sel]*galaxyxx[sel], galaxyzerr[sel]*galaxyxx[sel]),axis=0))
            ymesh = np.arange(np.min(galaxyz[sel]*galaxyyy[sel])-5*np.max(np.abs(galaxyzerr[sel]*galaxyyy[sel])), np.max(galaxyz[sel]*galaxyyy[sel])+5*np.max(np.abs(galaxyzerr[sel]*galaxyyy[sel])),\
                              np.min(galaxyzerr[sel])/1000.)
            ycen = get_median_eCDF(ymesh, np.sum(gauss_vectorized(ymesh, galaxyz[sel]*galaxyyy[sel], galaxyzerr[sel]*galaxyyy[sel]),axis=0))
            zmesh = np.arange(np.min(galaxyz[sel]*galaxyzz[sel])-5*np.max(np.abs(galaxyzerr[sel]*galaxyzz[sel])), np.max(galaxyz[sel]*galaxyzz[sel])+5*np.max(np.abs(galaxyzerr[sel]*galaxyzz[sel])),\
                              np.min(galaxyzerr[sel])/1000.)
            zcen = get_median_eCDF(zmesh, np.sum(gauss_vectorized(zmesh, galaxyz[sel]*galaxyzz[sel], galaxyzerr[sel]*galaxyzz[sel]),axis=0))    
            redshiftcen = np.sqrt(xcen*xcen + ycen*ycen + zcen*zcen)
            deccen = np.arcsin(zcen/redshiftcen)*180.0/np.pi # degrees
            if (ycen >=0 and xcen>=0):
                phicor = 0.0
            elif (ycen < 0 and xcen < 0):
                 phicor = 180.0
            elif (ycen >= 0 and xcen < 0):
                phicor = 180.0
            elif (ycen < 0 and xcen >=0):
                phicor = 360.0
            elif (xcen==0 and ycen==0):
                 print("Warning: xcen=0 and ycen=0 for group ", uid)
            racen=np.arctan(ycen/xcen)*(180/np.pi)+phicor # in degrees
            groupra[sel] = racen # in degrees
            groupdec[sel] = deccen # in degrees
            groupz[sel] = redshiftcen
    if return_z_pdfs:
        zmesh = np.arange(0,np.max(galaxyz)+8*np.max(galaxyzerr), 1/cspeed) # 1 km/s resolution
        z_pdfs = np.zeros((len(galaxygrpid), len(zmesh)))
        for i,uid in enumerate(uniqidnumbers):
            sel=np.where(galaxygrpid==uid)
            z_pdfs[i]=np.sum(gauss_vectorized(zmesh, galaxyz[sel], galaxyzerr[sel]),axis=0)
            z_pdfs[i]=z_pdfs[i]/simpson(z_pdfs[i],zmesh)
        pdfoutput = {'zmesh':zmesh, 'pdf':z_pdfs, 'grpid':uniqidnumbers}
    else:
        pdfoutput=None
    return groupra, groupdec, groupz, pdfoutput


######################################################################################################
######################################################################################################
######################################################################################################
######################################################################################################
# Iterative Combination for Giant Galaxies
def prob_iterative_combination_giants(galaxyra,galaxydec,galaxycz,galaxyczerr,giantfofid,rprojboundary,vprojboundary,decisionmode,cosmo):
    """
    Iteratively combine giant-only FoF groups using group N_giants-based boundaries. This method is probabilistic for Hutchens+2024
    and incorporates photo-z errors.

    Parameters
    --------------
    galaxyra : array_like
        RA of giant galaxies in decimal degrees.
    galaxydec : array_like
        Dec of giant galaxies in decimal degrees.
    galaxycz : array_like
        cz of giant galaxies in km/s.
    galaxyczerr : array_like
        cz err of giant galaxies in km/s.
    giantfofid : array_like
        FoF group ID for each giant galaxy, length matches `galaxyra`.
    rprojboundary : callable
        Search boundary to apply on-sky. Should be callable function of group N_giants.
        Units Mpc/h with h being consistent with `H0` argument.
    vprojboundary : callable
        Search boundary to apply in line-of-sight.. Should be callable function of group N_giants.
        Units: km/s
    decisionmode : str
        'allgalaxies' or 'centers'. Specifies how to evaluate whether seed group pairs should be merged. 
    H0 : float
       Hubble constant in (km/s)/Mpc, default 100. 
    
    Returns
    --------------
    giantgroupid : np.array
        Array of group ID numbers following iterative combination. Unique values match that of `giantfofid`.

    """
    centermethod='arithmetic'
    galaxyra=np.array(galaxyra)
    galaxydec=np.array(galaxydec)
    galaxycz=np.array(galaxycz)
    galaxyczerr=np.array(galaxyczerr)
    giantfofid=np.array(giantfofid)
    assert callable(rprojboundary),"Argument `rprojboundary` must callable function of N_giants."
    assert callable(vprojboundary),"Argument `vprojboundary` must callable function of N_giants."

    giantgroupid = np.copy(giantfofid)
    converged=False
    niter=0
    while (not converged):
        print("Giant-only iterative combination {} in progress...".format(niter))
        oldgiantgroupid = giantgroupid
        giantgroupid = prob_nearest_neighbor_assign(galaxyra,galaxydec,galaxycz,galaxyczerr,oldgiantgroupid,rprojboundary,vprojboundary,centermethod,decisionmode,H0)
        converged = np.array_equal(oldgiantgroupid,giantgroupid)
        niter+=1
    print("Giant-only iterative combination complete.")
    return giantgroupid

def prob_nearest_neighbor_assign(galaxyra, galaxydec, galaxycz, galaxyczerr, grpid, rprojboundary, vprojboundary, centermethod, decisionmode, cosmo):
    """
    Refine input group ID by merging nearest-neighbor groups subject to boundary constraints. For info on arguments, 
    see `giantonly_iterative_combination`

    Returns
    ---------------
    Refined group ID numbers based on nearest-neighbor merging
    """
    # Prepare output array
    refinedgrpid = deepcopy(grpid)
    # Get the group RA/Dec/cz for every galaxy
    groupra, groupdec, groupcz = fof.group_skycoords(galaxyra, galaxydec, galaxycz, grpid)
    groupN = fof.multiplicity_function(grpid,return_by_galaxy=True)
    # Get unique potential groups
    uniqgrpid, uniqind = np.unique(grpid, return_index=True)
    potra, potdec, potcz = groupra[uniqind], groupdec[uniqind], groupcz[uniqind]
    # Build & query the K-D Tree
    potphi = potra*np.pi/180.
    pottheta = np.pi/2. - potdec*np.pi/180.
    #zmpc = potcz/HUBBLE_CONST
    #xmpc = 2.*np.pi*zmpc*potra*np.cos(np.pi*potdec/180.) / 360.
    #ympc = np.float64(2.*np.pi*zmpc*potdec / 360.)
    cmvgdist = cosmo.comoving_distance(galaxycz / 3e5).value
    zmpc = cmvgdist * np.cos(pottheta)
    xmpc = cmvgdist * np.sin(pottheta)*np.cos(potphi)
    ympc = cmvgdist * np.sin(pottheta)*np.sin(potphi)
    coords = np.array([xmpc, ympc, zmpc]).T
    kdt = cKDTree(coords)
    nndist, nnind = kdt.query(coords,k=2)
    nndist=nndist[:,1] # ignore self match
    nnind=nnind[:,1]

    # go through potential groups and adjust membership for input galaxies
    alreadydone=np.zeros(len(uniqgrpid)).astype(int)
    ct=0
    for idx, uid in enumerate(uniqgrpid):
        # find the nearest neighbor group
        nbridx = nnind[idx]
        Gpgalsel=np.where(grpid==uid)
        GNNgalsel=np.where(grpid==uniqgrpid[nbridx])
        combinedra,combineddec,combinedcz = np.hstack((galaxyra[Gpgalsel],galaxyra[GNNgalsel])),np.hstack((galaxydec[Gpgalsel],galaxydec[GNNgalsel])),np.hstack((galaxycz[Gpgalsel],galaxycz[GNNgalsel]))
        #combinedgroupN = int(groupN[Gpgalsel][0])+int(groupN[GNNgalsel][0])
        combinedgalgrpid = np.hstack((grpid[Gpgalsel],grpid[GNNgalsel]))
        if prob_giants_fit_in_group(combinedra, combineddec, combinedcz, combinedczerr, combinedgalgrpid, rprojboundary, vprojboundary, centermethod, decisionmode, cosmo) and (not alreadydone[idx]) and (not alreadydone[nbridx]):
            # check for reciprocity: is the nearest-neighbor of GNN Gp? If not, leave them both as they are and let it be handled during the next iteration.
            nbrnnidx = nnind[nbridx]
            if idx==nbrnnidx:
                # change group ID of NN galaxies
                refinedgrpid[GNNgalsel]=int(grpid[Gpgalsel][0])
                alreadydone[idx]=1
                alreadydone[nbridx]=1
            else:
                alreadydone[idx]=1
        else:
            alreadydone[idx]=1
    return refinedgrpid

def prob_giants_fit_in_group(combinedra, combineddec, combinedcz, combinedczerr, combinedgalgrpid, rprojboundary, vprojboundary, centermethod, decisionmode, cosmo):
    """
    Evalaute whether two giant-only groups satisfy the specified boundary criteria.

    Parameters
    --------------------
    galra, galdec, galcz : iterable
        Coordinates of input galaxies -- all galaxies belonging to the pair of groups that are being assessed.
    galgrpid : iterable
        Seed group ID number for each galaxy (should be two unique values).
    totalgrpn : int
        Total group N (group N as if two seed groups were a single giant-only group).
    rprojboundary : callable
        Search boundary to apply on-sky. Should be callable function of group N_giants.
        Units Mpc/h with h being consistent with `H0` argument.
    vprojboundary : callable
        Search boundary to apply in line-of-sight.. Should be callable function of group N_giants.
        Units: km/s
    centermethod : str
        'arithmetic' or 'luminosity'. Specifies how to propose group centers during the combination process.
    decisionmode : str
        'allgalaxies' or 'centers'. Specifies how to evaluate whether seed group pairs should be merged.
    cosmo : astropy.cosmology 
        astropy cosmology object for computing comoving distances and other cosmological quantities
    """
    if decisionmode=='centers':
        uniqIDnums = np.unique(galgrpid)
        assert len(uniqIDnums)==2, "galgrpid must have two unique entries (two seed groups)."
        seed1sel = (galgrpid==uniqIDnums[0])
        assert False, "correct lines below to be cosmologically correct"
        seed1grpra,seed1grpdec,seed1grpcz = fof.group_skycoords(galra[seed1sel],galdec[seed1sel],galcz[seed1sel],galgrpid[seed1sel])
        seed2sel = (galgrpid==uniqIDnums[1])
        seed2grpra,seed2grpdec,seed2grpcz = fof.group_skycoords(galra[seed2sel],galdec[seed2sel],galcz[seed2sel],galgrpid[seed2sel])
        allgrpra,allgrpdec,allgrpcz = fof.group_skycoords(galra, galdec, galcz, np.zeros_like(galra)) # center of all galaxies
        seed1radialsep = (seed1grpcz[0]+allgrpcz[0])/100. * np.sin(fof.angular_separation(allgrpra[0],allgrpdec[0],seed1grpra[0],seed1grpdec[0])/2.)
        seed1lossep = np.abs(seed1grpcz[0]-allgrpcz[0])
        seed2radialsep = (seed2grpcz[0]+allgrpcz[0])/100. * np.sin(fof.angular_separation(allgrpra[0],allgrpdec[0],seed2grpra[0],seed2grpdec[0])/2.)
        seed2lossep = np.abs(seed2grpcz[0]-allgrpcz[0])
        totalgrpN = len(seed1grpra)+len(seed2grpra)
        fitingroup1 = (seed1radialsep<rprojboundary(totalgrpN)).all() and (seed1lossep<vprojboundary(totalgrpN)).all()
        fitingroup2 = (seed2radialsep<rprojboundary(totalgrpN)).all() and (seed2lossep<vprojboundary(totalgrpN)).all()
        fitingroup = fitingroup1 and fitingroup2
    else:
        raise ValueError("Only decisionmode `centers` is currently supported")
