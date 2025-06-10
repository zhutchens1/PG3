import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import foftools as fof
import iterativecombination as ic
from astropy.cosmology import LambdaCDM, z_at_value
import astropy.units as uu
from scipy.interpolate import interp1d 
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree
from smoothedbootstrap import smoothedbootstrap as sbs
from center_binned_stats import center_binned_stats
from numba import njit
from scipy.integrate import quad, simpson, dblquad, IntegrationWarning 
import math
from scipy.special import erf as scipy_erf
from robustats import weighted_median
from copy import deepcopy
from datetime import datetime

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("once",IntegrationWarning)

from matplotlib.ticker import MaxNLocator, AutoLocator
from matplotlib import rcParams
rcParams['axes.labelsize'] = 9
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['legend.fontsize'] = 9
rcParams['font.family'] = 'sans-serif'
rcParams['grid.color'] = 'k'
rcParams['grid.linewidth'] = 0.2
my_locator = MaxNLocator(6)
singlecolsize = (3.3522420091324205, 2.0717995001590714)
doublecolsize = (7.500005949910059, 4.3880449973709)
SPEED_OF_LIGHT = 2.998e5


def prob_g3groupfinder_luminosity(radeg,dedeg,cz,czerr,absrmag,dwarfgiantdivide,fof_bperp=0.07,fof_blos=1.1,fof_sep=None, volume=None,pfof_Pth=0.01, center_mode='average',\
                 iterative_giant_only_groups=False, n_bootstraps=1000, rproj_fit_guess=None, rproj_fit_params = None, rproj_fit_multiplier=None,\
                 vproj_fit_guess = None, vproj_fit_params = None, vproj_fit_multiplier=None, vproj_fit_offset=0, gd_rproj_fit_guess=None, gd_rproj_fit_params = None,\
                 gd_rproj_fit_multiplier=None, gd_vproj_fit_guess=None, gd_vproj_fit_params = None, gd_vproj_fit_multiplier=None,gd_vproj_fit_offset=None,
                 gd_fit_bins=None,H0=100., Om0=0.3, Ode0=0.7, saveplotspdf=False, summary_page_savepath=None):
    """
    Identify galaxy groups in redshift space using the RESOLVE-G3 algorithm (Hutchens et al. 2022).

    Parameters
    -------------------
    radeg : array_like
        Right ascension of input galaxies in decimal degrees.
    dedeg : array_like
        Declination of input galaxies in decimal degrees.
    cz : array_like
        Recessional velocities of input galaxies in km/s
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
    if (cz<20).all():
        raise ValueError("all cz<20; did you accidentally provide raw redshifts rather than cz in km/s?")
    absrmag=np.array(absrmag)
    g3grpid = np.zeros_like(radeg)-99.
    g3ssid = np.zeros_like(radeg)-99.
    cosmo = LambdaCDM(H0=H0,Om0=Om0,Ode0=Ode0)

    if summary_page_savepath is not None:
        make_summary_page = True
        PDF = PdfPages(summary_page_savepath)


    ### giant-only FoF ----------------- # 
    giantsel = (absrmag<=dwarfgiantdivide)
    if fof_sep is not None:
        giantfofid = pfof_comoving(radeg[giantsel], dedeg[giantsel], cz[giantsel], czerr[giantsel], fof_bperp*fof_sep, fof_blos*fof_sep, pfof_Pth, H0=H0, Om0=Om0, Ode0=Ode0)
    else:
        fof_sep = (volume/np.sum(giantsel))**(1/3.)
        giantfofid = pfof_comoving(radeg[giantsel], dedeg[giantsel], cz[giantsel], czerr[giantsel], fof_bperp*fof_sep, fof_blos*fof_sep, pfof_Pth, H0=H0, Om0=Om0, Ode0=Ode0)
    g3grpid[giantsel] = giantfofid

    ### if values not passed, fit rproj and vproj vs. N_giants
    if (rproj_fit_params is None) or (vproj_fit_params is None):
        if center_mode=='average' or center_mode=='giantaverage':
            #giantgrpra, giantgrpdec, giantgrpcz = fof.group_skycoords(radeg[giantsel], dedeg[giantsel], cz[giantsel], giantfofid)
            giantgrpra, giantgrpdec, giantgrpz, zpdfs = prob_group_skycoords(radeg[giantsel], dedeg[giantsel], cz[giantsel]/SPEED_OF_LIGHT, czerr[giantsel]/SPEED_OF_LIGHT, giantfofid)
            giantgrpcz = giantgrpz * SPEED_OF_LIGHT
        else:
            raise ValueError('check group center definition (`center_mode`), only `average` or `giantaverage` (equivalent) currently supported')
        relvel = np.abs(giantgrpcz - cz[giantsel])/(1+giantgrpz) # from https://academic.oup.com/mnras/article/442/2/1117/983284#30931438
        grp_ctd = cosmo.comoving_transverse_distance(giantgrpz).value
        relprojdist = (grp_ctd+grp_ctd)*np.sin(fof.angular_separation(giantgrpra, giantgrpdec, radeg[giantsel], dedeg[giantsel])/2.0)
        giantgrpn = fof.multiplicity_function(giantfofid, return_by_galaxy=True)
        uniqgiantgrpn, uniqindex = np.unique(giantgrpn, return_index=True)
        keepcalsel = np.where(uniqgiantgrpn>1)
        wavg_relprojdist = np.array([weighted_median(relprojdist[np.where(giantgrpn==sz)], 1/czerr[np.where(giantgrpn==sz)]) for sz in uniqgiantgrpn[keepcalsel]])
        wavg_relvel = np.array([weighted_median(relvel[np.where(giantgrpn==sz)], 1/czerr[np.where(giantgrpn==sz)]) for sz in uniqgiantgrpn[keepcalsel]])
        wavg_relprojdist_err = np.zeros_like(wavg_relprojdist)
        wavg_relvel_err = np.zeros_like(wavg_relvel)
        for ii,nn in enumerate(uniqgiantgrpn[keepcalsel]):
            df_ = pd.DataFrame({'czerr':czerr[np.where(giantgrpn==nn)], 'rpdist':relprojdist[np.where(giantgrpn==nn)], 'dv':relvel[np.where(giantgrpn==nn)]})
            resamples = [df_.sample(frac=1, replace=True) for ii in range(0,n_bootstraps)]
            wavg_relprojdist_err[ii] = np.std([weighted_median(resamp.rpdist, 1/resamp.czerr) for resamp in resamples])
            wavg_relvel_err[ii] = np.std([weighted_median(resamp.dv, 1/resamp.czerr) for resamp in resamples])
        rproj_bestfit, rproj_bestfit_cov = curve_fit(giantmodel, uniqgiantgrpn[keepcalsel], wavg_relprojdist,  p0=rproj_fit_guess, maxfev=2000,sigma=wavg_relprojdist_err)
        rproj_bestfit_err = np.sqrt(np.diag(rproj_bestfit_cov))
    else:
        rproj_bestfit = np.array(rproj_fit_params)
        rproj_bestfit_err = np.zeros(2)*1.
    if vproj_fit_params is None:
        try:
            vproj_bestfit, vproj_bestfit_cov  = curve_fit(giantmodel, uniqgiantgrpn[keepcalsel], wavg_relvel,  p0=vproj_fit_guess, maxfev=2000,sigma=wavg_relvel_err)
            vproj_bestfit_err = np.sqrt(np.diag(vproj_bestfit_cov))
        except RuntimeError:
            plt.figure()
            plt.plot(uniqgiantgrpn[keepcalsel], wavg_relvel, wavg_relvel_err)
            plt.xlabel("Ngiants")
            plt.ylabel("rel vel")
            plt.xlim(0,20)
            plt.show()
            print("Code failed.")
            exit()
    else:
        vproj_bestfit = np.array(vproj_fit_params)
        vproj_bestfit_err = np.zeros(2)*1.
    
    rproj_boundary = lambda Ngiants: rproj_fit_multiplier*giantmodel(Ngiants, *rproj_bestfit)
    vproj_boundary = lambda Ngiants: vproj_fit_multiplier*giantmodel(Ngiants, *vproj_bestfit) + vproj_fit_offset
    if make_summary_page or saveplotspdf:
        fig1 = plot_rproj_vproj_1(uniqgiantgrpn, giantgrpn, relprojdist, wavg_relprojdist, wavg_relprojdist_err, rproj_bestfit, relvel,\
            wavg_relvel, wavg_relvel_err, vproj_bestfit, keepcalsel, saveplotspdf)
        

    ### if requested, merge giant-only FoF groups through iterative combination
    if iterative_giant_only_groups:
        revisedgiantgrpid = prob_iterative_combination_giants(radeg[giantsel],dedeg[giantsel],cz[giantsel]/SPEED_OF_LIGHT,czerr[giantsel]/SPEED_OF_LIGHT,giantfofid,\
            rproj_boundary,vproj_boundary,pfof_Pth,cosmo)
        g3ssid[giantsel] = giantfofid
        g3grpid[giantsel] = revisedgiantgrpid
    else:
        pass

    ### associate dwarfs to giant-only groups
    dwarfsel = (absrmag>dwarfgiantdivide)
    if center_mode=='average' or center_mode=='giantaverage':
        giantgrpra, giantgrpdec, giantgrpz, pdfdict = prob_group_skycoords(radeg[giantsel], dedeg[giantsel], cz[giantsel]/SPEED_OF_LIGHT, czerr[giantsel]/SPEED_OF_LIGHT, g3grpid[giantsel], True)
    else:
        raise ValueError("center_mode must be `average` or `giantaverage`")
    
    giantgrpn = fof.multiplicity_function(g3grpid[giantsel],return_by_galaxy=True)
    dwarfassocid, _ = prob_faint_assoc(radeg[dwarfsel],dedeg[dwarfsel],cz[dwarfsel]/SPEED_OF_LIGHT,czerr[dwarfsel]/SPEED_OF_LIGHT,giantgrpra,giantgrpdec,giantgrpz,pdfdict,\
                        g3grpid[giantsel],rproj_boundary(giantgrpn),vproj_boundary(giantgrpn), pfof_Pth, H0=H0,Om0=Om0,Ode0=Ode0)
    g3grpid[dwarfsel]=dwarfassocid
    print('Finished associating dwarfs to giant-only groups.')

    #### -------- Derive boundaries for dwarf iterative combination
    if (gd_rproj_fit_params is None) or (gd_vproj_fit_params is None):
        gdgrpn = fof.multiplicity_function(g3grpid, return_by_galaxy=True)
        gdsel = np.logical_not(np.logical_or(g3grpid==-99., ((gdgrpn==1) & (absrmag>dwarfgiantdivide))))
        gdgrpra, gdgrpdec, gdgrpz, _ = prob_group_skycoords(radeg[gdsel], dedeg[gdsel], cz[gdsel]/SPEED_OF_LIGHT, czerr[gdsel]/SPEED_OF_LIGHT,\
             g3grpid[gdsel])

        gdrelvel = SPEED_OF_LIGHT*np.abs(cz[gdsel]/SPEED_OF_LIGHT - gdgrpz)/(1+gdgrpz)
        ctd1 = cosmo.comoving_transverse_distance(gdgrpz).value
        ctd2 = cosmo.comoving_transverse_distance(cz[gdsel]/SPEED_OF_LIGHT).value
        gdrelprojdist = (ctd1 + ctd2) * np.sin(fof.angular_separation(gdgrpra, gdgrpdec, radeg[gdsel], dedeg[gdsel])/2.0)
        gdn = gdgrpn[gdsel]
        gdtotalmag = ic.get_int_mag(absrmag[gdsel], g3grpid[gdsel])
        binsel = np.where(np.logical_and(gdn>1, gdtotalmag>-24))
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
    if make_summary_page or saveplotspdf: 
        fig2=plot_rproj_vproj_2(g3grpid, absrmag, gdsel, gdtotalmag, gdrelprojdist, gdrelvel, magbincenters, binsel, gdmedianrproj,\
             gdmedianrelvel, gd_rproj_bestfit, gd_vproj_bestfit, saveplotspdf)

    #### --------- iterative combination to make dwarf-only groups
    assert (g3grpid[(absrmag<=dwarfgiantdivide)]!=-99.).all(), "Not all giants are grouped." 
    grpnafterassoc = fof.multiplicity_function(g3grpid, return_by_galaxy=True)
    _ungroupeddwarf_sel = (absrmag>dwarfgiantdivide) & (grpnafterassoc==1)
    itassocid = dwarf_iterative_combination(radeg[_ungroupeddwarf_sel], dedeg[_ungroupeddwarf_sel], cz[_ungroupeddwarf_sel]/SPEED_OF_LIGHT, \
        czerr[_ungroupeddwarf_sel]/SPEED_OF_LIGHT, absrmag[_ungroupeddwarf_sel],rproj_for_iteration, vproj_for_iteration, pfof_Pth, cosmo, starting_id=np.max(g3grpid)+1)
    g3grpid[_ungroupeddwarf_sel]=itassocid

    ### ------------  return quantities
    if make_summary_page: 
        figs = get_extra_biopage_plots(g3grpid,radeg,dedeg,cz/SPEED_OF_LIGHT,czerr/SPEED_OF_LIGHT,absrmag,dwarfgiantdivide,volume,H0)
        for fig in figs:
            PDF.savefig(fig)
        PDF.savefig(fig1)
        PDF.savefig(fig2)
        PDF.close()
    return g3grpid, g3ssid, fof_sep, rproj_bestfit, rproj_bestfit_err, vproj_bestfit, vproj_bestfit_err, gd_rproj_bestfit,\
         gd_rproj_bestfit_err, gd_vproj_bestfit, gd_vproj_bestfit_err 


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
    VL_lower = cz - SPEED_OF_LIGHT*z_at_value(cosmo.comoving_distance, dc_lower*uu.Mpc, zmin=0.0, zmax=2, method='Bounded')
    VL_upper = SPEED_OF_LIGHT*z_at_value(cosmo.comoving_distance, dc_upper*uu.Mpc, zmin=0.0, zmax=2, method='Bounded') - cz
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
       galaxyz : 1D iterable, list of galaxy z values in redshift units (NOT km/s)
       galaxyzerr : 1D iterable, list of galaxy z 
       galaxygrpid : 1D iterable, group ID number for every galaxy in previous arguments.
       return_z_pdfs: True/False (default False), dictates whether group z PDFs are returned.
    
    Outputs (all shape match `galaxyra`)
       groupra : RA in decimal degrees of galaxy i's group center.
       groupdec : Declination in decimal degrees of galaxy i's group center.
       groupz : Redshift of galaxy i's group center.
       pdfoutput: If return_z_pdfs is True, this will be returned as a dictionary
                   with keys 'zmesh', 'pdf', and 'grpid'. Otherwise `None` is returned.
    
    Note: the FoF code of AA Berlind uses theta_i = declination, with theta_cen = 
    the central declination. This version uses theta_i = pi/2-dec, with some trig functions
    changed so that the output *matches* that of Berlind's FoF code (my "deccen" is the same as
    his "thetacen", to be exact.)
    -----
    """
    # 
    assert len(galaxyzerr)==len(galaxyz)
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
    cspeed=SPEED_OF_LIGHT
    galaxyz = np.expand_dims(galaxyz,axis=1)
    galaxyzerr = np.expand_dims(galaxyzerr,axis=1) 
    for i,uid in enumerate(uniqidnumbers):
        sel=np.where(galaxygrpid==uid)
        if len(sel[0])==1:
            groupra[sel] = galaxyra[sel]
            groupdec[sel] = galaxydec[sel]
            groupz[sel] = galaxyz[sel]#*cspeed
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
        #zmesh = np.arange(0,np.max(galaxyz)+8*np.max(galaxyzerr), 1/cspeed) # 1 km/s resolution
        try:
            zmesh = np.arange(0, np.max(galaxyz)+0.1, 1/cspeed) 
        except MemoryError:
            print("MemoryWarning: zmesh is too fine at line 523 in prob_g3groupfinder; trying at 20 km/s resolution")
            zmesh = np.arange(0, np.max(galaxyz)+0.1, 20/cspeed)
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
def prob_iterative_combination_giants(galaxyra,galaxydec,galaxyz,galaxyzerr,giantfofid,rprojboundary,vprojboundary,pthresh,cosmo):
    """
    Iteratively combine giant-only FoF groups using group N_giants-based boundaries. This method is probabilistic for Hutchens+2025
    and incorporates photo-z errors.

    Parameters
    --------------
    galaxyra : array_like
        RA of giant galaxies in decimal degrees.
    galaxydec : array_like
        Dec of giant galaxies in decimal degrees.
    galaxyz : array_like
        z of giant galaxies in redshift units NOT km/s
    galaxyzerr : array_like
        z err of giant galaxies in redshift units NOT km/s
    giantfofid : array_like
        FoF group ID for each giant galaxy, length matches `galaxyra`.
    rprojboundary : callable
        Search boundary to apply on-sky. Should be callable function of group N_giants.
        Units Mpc/h with h being consistent with `H0` argument.
    vprojboundary : callable
        Search boundary to apply in line-of-sight.. Should be callable function of group N_giants.
        Units: km/s
    cosmo : astropy.cosmology
        cosmology for distance calcs
    
    Returns
    --------------
    giantgroupid : np.array
        Array of group ID numbers following iterative combination. Unique values match that of `giantfofid`.

    """
    centermethod='arithmetic'
    galaxyra=np.array(galaxyra)
    galaxydec=np.array(galaxydec)
    galaxyz=np.array(galaxyz)
    galaxyzerr=np.array(galaxyzerr)
    giantfofid=np.array(giantfofid)
    assert callable(rprojboundary),"Argument `rprojboundary` must callable function of N_giants."
    assert callable(vprojboundary),"Argument `vprojboundary` must callable function of N_giants."

    giantgroupid = np.copy(giantfofid)
    converged=False
    niter=0
    while (not converged):
        print("Giant-only iterative combination {} in progress...".format(niter))
        oldgiantgroupid = giantgroupid
        giantgroupid = prob_giant_nearest_neighbor_assign(galaxyra,galaxydec,galaxyz,galaxyzerr,oldgiantgroupid,rprojboundary,vprojboundary,pthresh,cosmo)
        converged = np.array_equal(oldgiantgroupid,giantgroupid)
        niter+=1
    print("Giant-only iterative combination complete.")
    return giantgroupid

def prob_giant_nearest_neighbor_assign(galaxyra, galaxydec, galaxyz, galaxyzerr, grpid, rprojboundary, vprojboundary, pthresh, cosmo):
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
    groupra, groupdec, groupz, _ = prob_group_skycoords(galaxyra, galaxydec, galaxyz, galaxyzerr, grpid)
    groupN = fof.multiplicity_function(grpid,return_by_galaxy=True)
    # Get unique potential groups
    uniqgrpid, uniqind = np.unique(grpid, return_index=True)
    potra, potdec, potz = groupra[uniqind], groupdec[uniqind], groupz[uniqind]
    # Build & query the K-D Tree
    potphi = potra*np.pi/180.
    pottheta = np.pi/2. - potdec*np.pi/180.
    cmvgdist = cosmo.comoving_distance(potz).value
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
        combinedra,combineddec,combinedz,combinedzerr= np.hstack((galaxyra[Gpgalsel],galaxyra[GNNgalsel])),np.hstack((galaxydec[Gpgalsel],galaxydec[GNNgalsel])),np.hstack((galaxyz[Gpgalsel],galaxyz[GNNgalsel])),np.hstack((galaxyzerr[Gpgalsel],galaxyzerr[GNNgalsel]))
        #combinedgroupN = int(groupN[Gpgalsel][0])+int(groupN[GNNgalsel][0])
        combinedgalgrpid = np.hstack((grpid[Gpgalsel],grpid[GNNgalsel]))
        if prob_giants_fit_in_group(combinedra, combineddec, combinedz, combinedzerr, combinedgalgrpid, rprojboundary, vprojboundary, pthresh, cosmo) and (not alreadydone[idx]) and (not alreadydone[nbridx]):
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

def prob_giants_fit_in_group(combinedra, combineddec, combinedz, combinedzerr, combinedgalgrpid, rprojboundary, vprojboundary, pthresh, cosmo):
    """
    Evalaute whether two giant-only groups satisfy the specified boundary criteria.

    Parameters
    --------------------
    combinedra, combineddec, combinedz, combinedzerr : iterable
        Coordinates of input galaxies -- all galaxies belonging to the pair of groups that are being assessed.
        z should be in redshift units NOT km/s
    combinedgalgrpid : iterable
        Seed group ID number for each galaxy (should be two unique values).
    rprojboundary : callable
        Search boundary to apply on-sky. Should be callable function of group N_giants.
        Units Mpc/h with h being consistent with `H0` argument.
    vprojboundary : callable
        Search boundary to apply in line-of-sight.. Should be callable function of group N_giants.
        Units: km/s
    pthresh : float
        probability threshold for merging giant-only groups
    cosmo : astropy.cosmology 
        astropy cosmology object for computing comoving distances and other cosmological quantities
    """
    cc=SPEED_OF_LIGHT
    uniqIDnums = np.unique(combinedgalgrpid)
    assert len(uniqIDnums)==2, "galgrpid must have two unique entries (two seed groups)."
    seed1sel = (combinedgalgrpid==uniqIDnums[0])
    seed1grpra,seed1grpdec,seed1grpz,seed1pdf = prob_group_skycoords(combinedra[seed1sel],combineddec[seed1sel],combinedz[seed1sel], combinedzerr[seed1sel] ,combinedgalgrpid[seed1sel],True)
    seed2sel = (combinedgalgrpid==uniqIDnums[1])
    seed2grpra,seed2grpdec,seed2grpz,seed2pdf = prob_group_skycoords(combinedra[seed2sel],combineddec[seed2sel],combinedz[seed2sel], combinedzerr[seed2sel] ,combinedgalgrpid[seed2sel],True)
    allgrpra,allgrpdec,allgrpz,_ = prob_group_skycoords(combinedra, combineddec, combinedz, combinedzerr, np.zeros(len(combinedra)), False)
    totalgrpN = len(seed1grpra)+len(seed2grpra)
    #grp1v = (cosmo.H(seed1grpz)*cosmo.scale_factor(seed1grpz)*cosmo.comoving_distance(seed1grpz)).value
    #grp2v = (cosmo.H(seed2grpz)*cosmo.scale_factor(seed2grpz)*cosmo.comoving_distance(seed2grpz)).value
    #v12 = np.abs(grp2v[0]-grp1v[0])/(1-(grp2v[0]*grp1v[0]/(cc*cc)))
    #as_ = fof.angular_separation(seed1grpra[0], seed1grpdec[0], seed2grpra[0], seed2grpdec[0])
    #r12 = 0.5*(cosmo.comoving_transverse_distance(seed1grpz[0]).value + cosmo.comoving_transverse_distance(seed2grpz[0]).value)*as_
    #radialcondition = (v12 < vprojboundary(totalgrpN))
    #transvcondition = (r12 < rprojboundary(totalgrpN))
    #fitingroup = radialcondition & transvcondition
    seed1radialsep = (cosmo.comoving_transverse_distance(seed1grpz[0]).to_value()+cosmo.comoving_transverse_distance(allgrpz[0]).to_value())*(fof.angular_separation(allgrpra[0],\
        allgrpdec[0],seed1grpra[0],seed1grpdec[0])/2.)
    seed2radialsep = (cosmo.comoving_transverse_distance(seed2grpz[0]).to_value()+cosmo.comoving_transverse_distance(allgrpz[0]).to_value())*(fof.angular_separation(allgrpra[0],\
        allgrpdec[0],seed2grpra[0],seed2grpdec[0])/2.)
    fitingroup1 = (seed1radialsep<rprojboundary(totalgrpN)).all()
    fitingroup2 = (seed2radialsep<rprojboundary(totalgrpN)).all()
    if fitingroup1 and fitingroup2:
        eps_z = (1+seed1grpz[0])/SPEED_OF_LIGHT * vprojboundary(totalgrpN)
        interpkwargs = {'bounds_error' : False, 'fill_value' : 0}
        D1 = interp1d(seed1pdf['zmesh'], seed1pdf['pdf'][0], **interpkwargs)
        D2 = interp1d(seed2pdf['zmesh'], seed2pdf['pdf'][0], **interpkwargs)
        integrand = lambda zprime, z: D1(z)*D2(zprime)
        pcombine, pcombine_err = dblquad(integrand, 0.01, 0.02, lambda z: z-eps_z, lambda z: z+eps_z, epsrel=0.0001)
        fitingroup = (pcombine > pthresh)
    else:
        fitingroup = False
    return fitingroup

#######################################################################
#######################################################################
#######################################################################
## Dwarf galaxy association code

def prob_faint_assoc(faintra, faintdec, faintz, faintzerr, grpra, grpdec, grpz, grpzpdf, grpid, radius_boundary, velocity_boundary, Pth, H0=100., Om0=0.3, Ode0=0.7):
    """
    Associate galaxies to a group catalog based on given radius and velocity boundaries, based on a method
    similar to that presented in Eckert+ 2016. As used in Hutchens+2023 

    Parameters
    ----------
    faintra : iterable
        Right-ascension of faint galaxies in degrees.
    faintdec : iterable
        Declination of faint galaxies in degrees.
    faintz : iterable
        Redshifts of galaxies to be associated in redshift units NOT km/s
    faintzerr : iterable
        Redshifts errors of galaxies to be associated in redshift units NOT km/s
    grpra : iterable
        Right-ascension of group centers in degrees.
    grpdec : iterable
        Declination of group centers in degrees. Length matches `grpra`.
    grpz : iterable
        Redshift velocity of group center in km/s. Length matches `grpra`.
    grpzpdf : dict
        zpdfs for giant-only groups
    grpid : iterable
        group ID of each FoF group (i.e., from `foftools.fast_fof`.) Length matches `grpra`.
    radius_boundary : iterable
        Radius within which to search for faint galaxies around FoF groups. Length matches `grpra`.
    velocity_boundary : iterable
        Velocity from group center within which to search for faint galaxies around FoF groups. Length matches `grpra`.

    Returns
    -------
    assoc_grpid : iterable
        group ID of every faint galaxy. Length matches `faintra`.
    assoc_flag : iterable
        association flag for every galaxy (see function description). Length matches `faintra`.
    """
    cosmo = LambdaCDM(H0=H0, Om0=Om0, Ode0=Ode0) # this puts everything in "per h" units.
    velocity_boundary=np.asarray(velocity_boundary)
    radius_boundary=np.asarray(radius_boundary)
    Nfaint = len(faintra)
    assoc_grpid = np.zeros(Nfaint).astype(int)
    assoc_flag = np.zeros(Nfaint).astype(int)
    prob_values=np.zeros(Nfaint)

    # resize group coordinates to be the # of groups, not # galaxies
    junk, uniqind = np.unique(grpid, return_index=True)
    grpra = grpra[uniqind]
    grpdec = grpdec[uniqind]
    grpz = grpz[uniqind]
    grpid = grpid[uniqind]
    velocity_boundary=velocity_boundary[uniqind]
    radius_boundary=radius_boundary[uniqind]

    # Make Nfaints x Ngroups grids for transverse/LOS distances from group centers
    faintphi = (faintra * np.pi/180.)[:,None]
    fainttheta = (np.pi/2. - faintdec*(np.pi/180.))[:,None]
    faint_cmvg = (cosmo.comoving_transverse_distance(faintz).value)[:, None]
    grpphi = (grpra * np.pi/180.)
    grptheta = (np.pi/2. - grpdec*(np.pi/180.))
    grp_cmvg = cosmo.comoving_transverse_distance(grpz).value
    half_angle = np.arcsin((np.sin((fainttheta-grptheta)/2.0)**2.0 + np.sin(fainttheta)*np.sin(grptheta)*np.sin((faintphi-grpphi)/2.0)**2.0)**0.5)
    Rp = (faint_cmvg + grp_cmvg) * (half_angle)/2
    DeltaV = SPEED_OF_LIGHT*np.abs(faintz[:,None] - grpz)/(1+grpz)
    for gg in range(0,len(grpid)):
        for fg in range(0,Nfaint):
            quick__test = ((Rp[fg][gg]<radius_boundary[gg]) & (DeltaV[fg][gg]<6000))
            if quick__test:
                zrange = (1+grpz[gg])*velocity_boundary[gg]/SPEED_OF_LIGHT
                sel = np.where(grpzpdf['grpid']==grpid[gg])
                pdfneeded = grpzpdf['pdf'][sel]
                Poverlap = dwarf_association_integral(grpzpdf['zmesh'], pdfneeded[0], faintz[fg], faintzerr[fg], zrange, zrange)
                if (Poverlap>Pth) and (not bool(assoc_flag[fg])):
                    prob_values[fg]=Poverlap
                    assoc_grpid[fg]=grpid[gg]
                    assoc_flag[fg]=1
                elif (Poverlap>Pth) and (bool(assoc_flag[fg])):
                    # has already been assocated; is our new Poverlap better?
                    if Poverlap>prob_values[fg]:
                        prob_values[fg]=Poverlap
                        assoc_grpid[fg]=grpid[gg]
                        assoc_flag[fg]=1
                else:
                    pass
            else:
                pass
    #plt.figure()
    #print("prob values", prob_values)
    #plt.hist(np.log10(prob_values[prob_values!=0]),bins='fd')
    #plt.axvline(np.log10(Pth),color='k')
    #plt.xlabel("log prob_values")
    #plt.show()
    # assign group ID numbers to galaxies that didn't associate
    still_isolated = np.where(assoc_grpid==0)
    assoc_grpid[still_isolated]=np.arange(np.max(grpid)+1, np.max(grpid)+1+len(still_isolated[0]), 1)
    assoc_flag[still_isolated]=-1
    return assoc_grpid, assoc_flag

def dwarf_association_integral(zgrid, grpzpdf, zdwarf, zerrdwarf, zrangeup, zrangelow):
    """ 
    calculate entire integral 
    P = int_0^inf DG(z) * gamma(z) * dz where gamma(z) is as below and DG(z) is the group z distribution function
    """
    zgridnew = np.arange(0,max((np.max(zgrid),zdwarf+5*zerrdwarf)), zgrid[1]-zgrid[0]) # expand grid to be larger if needed
    grpzpdfnew = np.interp(zgridnew, zgrid, grpzpdf, left=0, right=0)
    grpzpdfnew /= simpson(grpzpdfnew,zgridnew) # normalize
    integrand = grpzpdfnew * gamma_dwarf_assoc_subintegral(zgridnew, zdwarf, zerrdwarf, zrangeup, zrangelow)
    return simpson(integrand, zgridnew)

def gamma_dwarf_assoc_subintegral(zgrid, zdwarf, zerrdwarf, zrangeup, zrangelow):
    """
    defined as gamma(z) = int_(z-zrangelow)^(z+zrangeup) G(zgrid | zdwarf, zerrdwarf) d(zgrid)
    zgrid should be array
    zrangeup, zrangelow scalars
    zdwarf, zerrdwarf are redshift and uncertainty for dwarf galaxy being tested for association
    """
    return gaussian_integral(zdwarf, zerrdwarf, zgrid-zrangelow, zgrid+zrangeup)

def gaussian_integral(mu, sigma, a, b):
    """
    mean mu, dispersion sigma, limits of integration a, b (a is lower limit)
    """
    den = sigma*1.41421356
    term1 = scipy_erf((b-mu)/den)
    term2 = scipy_erf((a-mu)/den)
    return 0.5*(term1 - term2)

# =============================================================================== #
# =============================================================================== #
# code for dwarf-only group-finding

def dwarf_iterative_combination(galaxyra, galaxydec, galaxyz, galaxyzerr, galaxymag, rprojboundary, vprojboundary, pthresh, cosmo, starting_id=1):
    """
    Perform iterative combination on a list of input galaxies.
    
    Parameters
    ------------
    galaxyra, galaxydec : iterable
       Right-ascension and declination of the input galaxies in decimal degrees.
    galaxycz : iterable
       Redshift velocity (corrected for Local Group motion) of input galaxies in km/s.
    galaxyczerr : iterable
        Redshift velocity uncertainty (corrected for Local Group motion) of input galaxies in km/s.
    galaxymag : iterable
       M_r absolute magnitudes of input galaxies, or galaxy stellar/baryonic masses (the code will be able to differentiate.)
    rprojboundary : callable
       Search boundary to apply in projection on the sky for grouping input galaxies, function of group-integrated M_r, in units Mpc/h.
    vprojboundary : callable
       Search boundary to apply in velocity  on the sky for grouping input galaxies, function of group-integrated M_r or mass, in units km/s.
    pthresh : float
        threshold probability
    cosmo : astropy.cosmology
        cosmology object
    starting_id : int, default 1
       Base ID number to assign to identified groups (all group IDs will be >= starting_id).
    
    Returns
    -----------
    itassocid: Group ID numbers for every input galaxy. Shape matches `galaxyra`.
    """
    print("Beginning iterative combination...")
    # Check user input
    assert (callable(rprojboundary) and callable(vprojboundary)),"Inputs `rprojboundary` and `vprojboundary` must be callable."
    assert (len(galaxyra)==len(galaxydec) and len(galaxydec)==len(galaxyz) and len(galaxyzerr)==len(galaxyz)),"RA/Dec/z/zerr inputs must have same shape."
    # Convert everything to numpy + create ID array (assuming for now all galaxies are isolated)
    galaxyra = np.array(galaxyra)
    galaxydec = np.array(galaxydec)
    galaxyz = np.array(galaxyz)
    galaxyzerr = np.array(galaxyzerr)
    galaxymag = np.array(galaxymag)
    itassocid = np.arange(starting_id, starting_id+len(galaxyra))
    # Begin algorithm. 
    converged=False
    niter=0
    while (not converged):
        print("iteration {} in progress...".format(niter))
        # Compute based on updated ID number
        olditassocid = itassocid
        itassocid = dwarf_nearest_neighbor_assign(galaxyra, galaxydec, galaxyz, galaxyzerr, galaxymag,\
             olditassocid, rprojboundary, vprojboundary, pthresh, cosmo)
        # check for convergence
        converged = np.array_equal(olditassocid, itassocid)
        niter+=1
    print("Iterative combination complete.")
    return itassocid

def dwarf_nearest_neighbor_assign(galaxyra, galaxydec, galaxyz, galaxyzerr,  galaxymag, grpid, rprojboundary, vprojboundary, pthresh, cosmo):
    """
    For a list of galaxies defined by groups, refine group ID numbers using a nearest-neighbor
    search and applying the search boundaries.

    Parameters
    ------------
    galaxyra, galaxydec, galaxycz, gaaxyczerr : iterable
        Input coordinates of galaxies (RA/Dec in decimal degrees, cz and czerr in km/s)
    galaxymag : iterable
        M_r magnitudes or stellar/baryonic masses of input galaxies. (note code refers to 'mags' throughout, but
        underlying `fit_in_group` function will distinguish the two.)
    grpid : iterable
        Group ID number for every input galaxy, at current iteration (potential group).
    rprojboundary, vprojboundary : callable
        Input functions to assess the search boundaries around potential groups, function of group-integrated luminosity, units Mpc/h and km/s.
    pthresh : float
        probability threshold for merging groups

    Returns
    ------------
    associd : iterable
        Refined group ID numbers based on NN "stitching" of groups.
    """
    # Prepare output array
    associd = deepcopy(grpid)
    # Get the group RA/Dec/cz for every galaxy
    groupra, groupdec, groupz, _ = prob_group_skycoords(galaxyra, galaxydec, galaxyz, galaxyzerr, grpid)
    # Get unique potential groups
    uniqgrpid, uniqind = np.unique(grpid, return_index=True)
    potra, potdec, potz = groupra[uniqind], groupdec[uniqind], groupz[uniqind] 
    # Build & query the K-D Tree
    potphi = potra*np.pi/180.
    pottheta = np.pi/2. - potdec*np.pi/180.
    dist = cosmo.comoving_distance(potz).to_value() # Mpc
    zmpc = dist * np.cos(pottheta) 
    xmpc = dist * np.sin(pottheta)*np.cos(potphi)
    ympc = dist * np.sin(pottheta)*np.sin(potphi)
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
        combinedra,combineddec,combinedz,combinedzerr = np.hstack((galaxyra[Gpgalsel],galaxyra[GNNgalsel])),np.hstack((galaxydec[Gpgalsel],galaxydec[GNNgalsel])),\
            np.hstack((galaxyz[Gpgalsel],galaxyz[GNNgalsel])), np.hstack((galaxyzerr[Gpgalsel],galaxyzerr[GNNgalsel]))
        combinedmag = np.hstack((galaxymag[Gpgalsel], galaxymag[GNNgalsel]))
        combinedgalgrpid = np.hstack((grpid[Gpgalsel],grpid[GNNgalsel]))
        condition = dwarfic_fit_in_group(combinedra, combineddec, combinedz, combinedzerr, combinedgalgrpid, combinedmag, rprojboundary,vprojboundary, pthresh, cosmo)
        if condition and (not alreadydone[idx]) and (not alreadydone[nbridx]):
            # check for reciprocity: is the nearest-neighbor of GNN Gp? If not, leave them both as they are and let it be handled during the next iteration.
            nbrnnidx = nnind[nbridx]
            if idx==nbrnnidx:
                # change group ID of NN galaxies
                associd[GNNgalsel]=int(grpid[Gpgalsel][0])
                alreadydone[idx]=1
                alreadydone[nbridx]=1
            else:
                alreadydone[idx]=1
        else:
            alreadydone[idx]=1
    return associd  


def dwarfic_fit_in_group(galra, galdec, galz, galzerr, galgrpid, galmag, rprojboundary, vprojboundary, pthresh, cosmo):
    """
    Check whether two potential groups can be merged based on the integrated luminosity of the 
    potential members, given limiting input group sizes.
    
    Parameters
    ----------------
    galra, galdec, galz, galzerr : iterable
        Coordinates of input galaxies -- all galaxies belonging to the pair of groups that are being assessed. 
    galgrpid : iterable
        Seed group ID number for each galaxy.
    galmag : iterable
        M_r absolute magnitudes of all input galaxies, or galaxy stellar masses - the function can distinguish the two.
    zpdfdict : dict
        Dictionary containing group z pdfs (output of prob_group_skycoords)
    rprojboundary, vprojboundary : callable
        Limiting projected- and velocity-space group sizes as function of group-integrated luminosity or stellar mass.
    pthresh: float
        Threshold probability for combining seed groups in LOS.

    Returns
    ----------------
    fitingroup : bool
        Bool indicating whether the series of input galaxies can be merged into a single group of the specified size.
    """
    if (galmag<=0).all():
        memberintmag = get_int_mag(galmag, np.full(len(galmag), 1))
    elif (galmag>=0).all():
        memberintmag = get_int_mass(galmag, np.full(len(galmag), 1))
    uniqIDnums = np.unique(galgrpid)
    assert len(uniqIDnums)==2, "galgrpid must have two unique entries (two seed groups)."
    seed1sel = (galgrpid==uniqIDnums[0])
    seed1grpra,seed1grpdec,seed1grpz, seed1pdf = prob_group_skycoords(galra[seed1sel],galdec[seed1sel],galz[seed1sel], galzerr[seed1sel], galgrpid[seed1sel], True)
    seed2sel = (galgrpid==uniqIDnums[1])
    seed2grpra,seed2grpdec,seed2grpz,seed2pdf = prob_group_skycoords(galra[seed2sel],galdec[seed2sel],galz[seed2sel],galzerr[seed2sel], galgrpid[seed2sel], True)
    allgrpra,allgrpdec,allgrpz,_ = prob_group_skycoords(galra, galdec, galz, galzerr, np.zeros(len(galra)), False)
    seed1radialsep = (cosmo.comoving_transverse_distance(seed1grpz[0]).to_value()+cosmo.comoving_transverse_distance(allgrpz[0]).to_value())*(fof.angular_separation(allgrpra[0],allgrpdec[0],seed1grpra[0],seed1grpdec[0])/2.)
    seed2radialsep = (cosmo.comoving_transverse_distance(seed2grpz[0]).to_value()+cosmo.comoving_transverse_distance(allgrpz[0]).to_value())*(fof.angular_separation(allgrpra[0],allgrpdec[0],seed2grpra[0],seed2grpdec[0])/2.)
    
    fitingroup1 = (seed1radialsep<rprojboundary(memberintmag)).all()
    fitingroup2 = (seed2radialsep<rprojboundary(memberintmag)).all()
    if fitingroup1 and fitingroup2:
        gamma_z = (1+seed1grpz[0])/SPEED_OF_LIGHT * vprojboundary(memberintmag)[0]
        interpkwargs = {'bounds_error' : False, 'fill_value' : 0}
        D1 = interp1d(seed1pdf['zmesh'], seed1pdf['pdf'][0], **interpkwargs)
        D2 = interp1d(seed2pdf['zmesh'], seed2pdf['pdf'][0], **interpkwargs)
        integrand = lambda zprime, z: D1(z)*D2(zprime)
        pcombine, pcombine_err = dblquad(integrand, 0.01, 0.02, lambda z: z-gamma_z, lambda z: z+gamma_z, epsrel=0.0001)
        fitingroup = (pcombine > pthresh)
    else:
        fitingroup = False
    return fitingroup


def plot_rproj_vproj_1(uniqgiantgrpn, giantgrpn, relprojdist, wavg_relprojdist, wavg_relprojdist_err, rproj_bestfit, relvel, wavg_relvel, wavg_relvel_err, vproj_bestfit, keepcalsel, saveplotspdf):
    fig,axs=plt.subplots(figsize=doublecolsize, ncols=2)
    tx = np.linspace(1,30,500)
    sel = np.where(giantgrpn>1)
    axs[0].plot(giantgrpn[sel], relprojdist[sel], 'r.', markersize=2, alpha=0.5, label='ECO Giant Galaxies',zorder=0, rasterized=True)
    axs[0].errorbar(uniqgiantgrpn[keepcalsel], wavg_relprojdist, wavg_relprojdist_err, fmt='^', color='k', label=r'$R_{\rm proj}$',zorder=0)
    axs[0].plot(tx, giantmodel(tx,*rproj_bestfit), color='blue', label=r'$1R_{\rm proj}^{\rm fit}$',zorder=2)
    axs[0].plot(tx, 3*giantmodel(tx,*rproj_bestfit), color='green', label=r'$3R_{\rm proj}^{\rm fit}$', linestyle='dashed',zorder=3)
    axs[0].plot(tx, giantmodel(tx,*(3.06e-1,4.16e-1)), color='gray', label=r'$1R_{\rm proj}^{\rm fit}$ from H23',zorder=2)
    
    axs[1].plot(giantgrpn[sel], relvel[sel], 'r.', markersize=2, alpha=0.5, label='ECO Giant Galaxies',zorder=0, rasterized=True)
    axs[1].errorbar(uniqgiantgrpn[keepcalsel], wavg_relvel, wavg_relvel_err, fmt='^', color='k', label=r'$\Delta v_{\rm proj}$',zorder=0)
    axs[1].plot(tx, giantmodel(tx,*vproj_bestfit), color='blue', label=r'$1\Delta v_{\rm proj}^{\rm fit}$',zorder=2)
    axs[1].plot(tx, 4*giantmodel(tx,*vproj_bestfit)+200, color='green', label=r'$4\Delta v_{\rm proj}^{\rm fit} + 200$ km s$^{-1}$',zorder=3, linestyle='dashed')
    axs[1].plot(tx, giantmodel(tx,*(3.45e2,0.17)), color='gray', label=r'$1\Delta v_{\rm proj}^{\rm fit}$ from H23',zorder=3)
    
    for ii in range(0,2):
        axs[ii].set_xlim(0,20)
        axs[ii].set_xlabel("Number of Giant Members in Initial Group")
        axs[ii].legend(loc='upper right',fontsize=8)
        tks = np.arange(0,22,2)
        axs[ii].set_xticks(tks)
    axs[0].set_ylabel("Projected Distance from Giant to Group Center [Mpc]")
    axs[1].set_ylabel(r"Relative Velocity from Giant to Group Center [km s$^{-1}$]")
    axs[0].set_ylim(0,1.0)
    axs[1].set_ylim(0,1000)
    plt.tight_layout()
    if saveplotspdf: plt.savefig("../figures/rproj_vproj_cal.pdf",dpi=300)
    plt.close()
    return fig

def plot_rproj_vproj_2(ecog3grp, ecoabsrmag, ecogdsel, ecogdtotalmag, ecogdrelprojdist, ecogdrelvel, magbincenters, binsel, gdmedianrproj, gdmedianrelvel, poptr, poptv,\
    saveplotspdf):
    tx = np.linspace(-27,-17,100)
    fig, (ax,ax1) = plt.subplots(ncols=2, figsize=doublecolsize)
    giantgrpn = np.array([np.sum((ecoabsrmag[ecogdsel][ecog3grp[ecogdsel]==gg]<-19.5)) for gg in ecog3grp[ecogdsel]])
    sel_ = np.where(np.logical_and(giantgrpn==1,ecogdtotalmag>-24))
    ax.plot(ecogdtotalmag[sel_], ecogdrelprojdist[sel_], '.', color='mediumorchid', alpha=0.6, label=r'ECO $N_{\rm giants}=1$ Group Galaxies', rasterized=True)
    sel_ = np.where(np.logical_and(giantgrpn==2,ecogdtotalmag>-24))
    ax.plot(ecogdtotalmag[sel_], ecogdrelprojdist[sel_], '.', color='darkorange', alpha=0.6, label=r'ECO $N_{\rm giants}=2$ Group Galaxies', rasterized=True)
    sel_ = np.where(np.logical_and(giantgrpn>2,ecogdtotalmag>-24))
    ax.plot(ecogdtotalmag[sel_], ecogdrelprojdist[sel_], '.', color='slategrey', alpha=0.6, label=r'ECO $N_{\rm giants}\geq3$ Group Galaxies', rasterized=True)
    #ax.errorbar(magbincenters, gdmedianrproj, yerr=gdmedianrproj_err, fmt='k^', label=r'Medians ($R_{\rm proj}^{\rm gi,\,dw}$)', rasterized=True, zorder=15)
    ax.errorbar(magbincenters, gdmedianrproj, yerr=None, fmt='k^', label=r'Medians ($R_{\rm proj}^{\rm gi,\,dw}$)', rasterized=True, zorder=15)
    ax.plot(tx, 1*decayexp(tx,*poptr), color='red', label=r'$R_{\rm proj,\,fit}^{\rm gi,\, dw}$', rasterized=True)
    ax.plot(tx, 2*decayexp(tx,*poptr), color='blue', label=r'$2R_{\rm proj,\,fit}^{\rm gi,\, dw}$', rasterized=True,linestyle='--')
    #ax.plot(tx, 3*decayexp(tx,*poptr), label=r'$3R_{\rm proj,\,fit}^{\rm gi,\, dw}$', rasterized=True)
    ax.set_xlabel(r"Integrated $M_r$ of Giant + Dwarf Members")
    ax.set_ylabel(r"Projected Distance from Galaxy to Group Center [Mpc]")
    ax.legend(loc='best',fontsize=8,framealpha=0.5)
    ax.set_xlim(-24.1,-17)
    xrange=[-24,-16]
    xticks=np.arange(xrange[0],xrange[1],1)
    ax.set_xticks(xticks)
    ax.set_ylim(0,0.8)
    ax.invert_xaxis()

    #ax1.plot(ecogdtotalmag[binsel], ecogdrelvel[binsel], '.', alpha=0.6, label='ECO Giant-Hosting Group Galaxies', rasterized=True, color='palegreen')
    #ax1.errorbar(magbincenters, gdmedianrelvel, yerr=gdmedianrelvel_err, fmt='k^',label=r'Medians ($\Delta v_{\rm proj}^{\rm gi,\,dw}$)', rasterized=True, zorder=15)
    ax1.errorbar(magbincenters, gdmedianrelvel, yerr=None, fmt='k^',label=r'Medians ($\Delta v_{\rm proj}^{\rm gi,\,dw}$)', rasterized=True, zorder=15)
    sel_ = np.where(np.logical_and(giantgrpn==1,ecogdtotalmag>-24))
    ax1.plot(ecogdtotalmag[sel_], ecogdrelvel[sel_], '.', color='mediumorchid', alpha=0.6, label=r'ECO $N_{\rm giants}=1$ Group Galaxies', rasterized=True)
    sel_ = np.where(np.logical_and(giantgrpn==2,ecogdtotalmag>-24))
    ax1.plot(ecogdtotalmag[sel_], ecogdrelvel[sel_], '.', color='darkorange', alpha=0.6, label=r'ECO $N_{\rm giants}=2$ Group Galaxies', rasterized=True)
    sel_ = np.where(np.logical_and(giantgrpn>2,ecogdtotalmag>-24))
    ax1.plot(ecogdtotalmag[sel_], ecogdrelvel[sel_], '.', color='slategrey', alpha=0.6, label=r'ECO $N_{\rm giants}\geq3$ Group Galaxies', rasterized=True)
    ax1.plot(tx, decayexp(tx, *poptv), color='red', label=r'$\Delta v_{\rm proj,\, fit}^{\rm gi,\, dw}$', rasterized=True)
    ax1.plot(tx, 4*decayexp(tx, *poptv)+100, color='blue', label=r'$4\Delta v_{\rm proj,\, fit}^{\rm gi,\, dw}$+100 km s$^{-1}$', rasterized=True, linestyle='--')
    ax1.set_ylabel(r"Relative Velocity from Galaxy to Group Center [km s$^{-1}]$")
    ax1.set_xlabel(r"Integrated $M_r$ of Giant + Dwarf Members")
    ax1.set_xlim(-24.1,-17)
    ax1.set_ylim(0,800)
    ax1.invert_xaxis()
    ax1.set_xticks(xticks)
    ax1.legend(loc='best',fontsize=8, framealpha=1)
    plt.tight_layout()
    if saveplotspdf: plt.savefig("../figures/itercombboundaries.pdf")
    plt.close()
    return fig

def sigmarange(x):
    q84, q16 = np.percentile(x, [84 ,16])
    return (q84-q16)/2.

def giantmodel(x, a, b):
    return np.abs(a)*np.log10(np.abs(b)*x+1)

def decayexp(x, a, b):
    #return np.abs(a)*np.exp(-1*np.abs(b)*x)#+np.abs(d)
    return np.abs(a)*np.exp(-1*np.abs(b)*(x+19.5))#+np.abs(d)

def get_int_mag(galmags, grpid):
    """
    Given a list of galaxy absolute magnitudes and group ID numbers,
    compute group-integrated total magnitudes.

    Parameters
    ------------
    galmags : iterable
       List of absolute magnitudes for every galaxy (SDSS r-band).
    grpid : iterable
       List of group ID numbers for every galaxy.

    Returns
    ------------
    grpmags : np array
       Array containing group-integrated magnitudes for each galaxy. Length matches `galmags`.
    """
    galmags=np.asarray(galmags)
    grpid=np.asarray(grpid)
    grpmags = np.zeros(len(galmags))
    uniqgrpid=np.unique(grpid)
    for uid in uniqgrpid:
        sel=np.where(grpid==uid)
        totalmag = -2.5*np.log10(np.sum(10**(-0.4*galmags[sel])))
        grpmags[sel]=totalmag
    return grpmags

def get_extra_biopage_plots(grpid,radeg,dedeg,zz,zzerr,absrmag,dwarfgiantdivide,volume,H0):
    # text
    firstpage = plt.figure(figsize=(11.69,8.27))
    firstpage.clf()
    lines = [
        f'Date generated: {datetime.now()}',
        f'{len(grpid)} Galaxies in Sample '+r'($M_r$ Floor'+f'={np.max(absrmag):0.2f})',
        f'Redshift range: {np.min(zz):0.5f} to {np.max(zz):0.5f}',
        f'Comoving Volume = {volume:0.4E} '+r'Mpc$^3$ (assumes $H_0$='+f'{H0:0.1f} km/s/Mpc)',
        f'{len(np.unique(grpid))} Unique Groups'+r' (including $N_{\rm galaxies}=1$ Groups)',
    ]
    text='\n'.join(lines)
    firstpage.text(0.1,0.5,text,transform=firstpage.transFigure, size=14, ha='left')
    plt.close()
    
    # mult func
    fig1 = plt.figure()
    grpn = np.unique(grpid,return_counts=True)[1]
    hbins = np.arange(0.5,40.5,1)
    plt.hist(grpn, bins=hbins)
    plt.xlim(0.5,40.5)
    plt.yscale('log')
    plt.xlabel(r'Group $N_{\rm galaxies}$')
    plt.ylabel(r'Number of Groups')
    n_high_N = len(grpn[grpn>np.max(hbins)])
    if n_high_N>0:
        plt.annotate(xy=(0.5,0.6), xycoords='axes fraction', text=f'+{n_high_N} N>{np.max(hbins)+0.5} Groups')
    plt.tight_layout()
    plt.close()

    # ra/dec
    fig2,axs = plt.subplots(ncols=2)
    axs[0].scatter(radeg, dedeg, color='k', s=2, alpha=0.2)
    axs[0].set_xlabel('RA [deg]')
    axs[0].set_ylabel('DEC [deg]')
    axs[1].hist(zz, bins='fd')
    axs[1].set_yscale('log')
    axs[1].set_xlabel('$z$')
    axs[1].set_ylabel('Number of Galaxies')
    plt.tight_layout()
    plt.close()

    # lum func
    fig3=plt.figure()
    plt.hist(absrmag, bins='fd')
    plt.xlabel(r'$M_r$')
    plt.ylabel('Number of Galaxies')
    plt.yscale('log')
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.close()

    return firstpage, fig1, fig2, fig3

# =============================================================================== #
# =============================================================================== #
if __name__=='__main__':
    import time
    t1 = time.time()
    eco = pd.read_csv("/srv/one/zhutchen/g3groupfinder/resolve_and_eco/ECOdata_G3catalog_luminosity.csv")
    eco = eco[eco.absrmag<-17.33] # just to test
    eco.loc[:,'czerr'] = eco.cz*0 + 20
    hubble_const = 70.
    omega_m = 0.3
    omega_de = 0.7
    cosmo=LambdaCDM(hubble_const, omega_m, omega_de)
    ecovolume = 191958.08 / (hubble_const/100.)**3.

    gfargseco = dict({'volume':ecovolume,'rproj_fit_multiplier':3,'vproj_fit_multiplier':4,'vproj_fit_offset':200,'summary_page_savepath':'eco.pdf','saveplotspdf':False,
           'gd_rproj_fit_multiplier':2, 'gd_vproj_fit_multiplier':4, 'gd_vproj_fit_offset':100,\
           'gd_fit_bins':np.arange(-24,-19,0.25), 'gd_rproj_fit_guess':[1e-5, 0.4],\
           'pfof_Pth' : 0.99, \
           'gd_vproj_fit_guess':[3e-5,4e-1], 'H0':hubble_const, 'Om0':omega_m, 'Ode0':omega_de,  'iterative_giant_only_groups':True})

    pg3out=prob_g3groupfinder_luminosity(eco.radeg, eco.dedeg, eco.cz, eco.czerr, eco.absrmag,-19.5,fof_bperp=0.07,fof_blos=1.1,**gfargseco)
    pg3grp=pg3out[0]
    eco.loc[:,'pg3grp'] = pg3grp
    print('elapsed time was ', time.time()-t1)

    bins = np.arange(0.5,300.5,1)
    plt.figure()
    plt.hist(fof.multiplicity_function(eco.g3grp_l.to_numpy(), return_by_galaxy=False), bins=bins, color='gray', histtype='stepfilled', label='G3 Groups', alpha=0.7)
    plt.hist(fof.multiplicity_function(eco.pg3grp, return_by_galaxy=False), bins=bins, color='blue', histtype='step', label='PG3 Groups', linewidth=3)
    plt.yscale('log')
    plt.xlabel(r"Group $N_{\rm galaxies}$")
    plt.xlim(0,50)
    plt.legend(loc='best')
    plt.show()
