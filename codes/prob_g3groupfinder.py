import math
import matplotlib
#matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import astropy.units as uu
from astropy.cosmology import LambdaCDM
from scipy.integrate import simpson
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree
#from scipy.sparse import csr_array
#from scipy.sparse.csgraph import connected_components
from scipy.interpolate import interp1d
from scipy.special import erf as scipy_erf
from smoothedbootstrap import smoothedbootstrap as sbs
from center_binned_stats import center_binned_stats
from robustats import weighted_median
from copy import deepcopy
from datetime import datetime
from joblib import Parallel, delayed
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
doublecolsize = (7.500005949910059, 4.3880449973709)
SPEED_OF_LIGHT = 2.998e5

class pg3(object):
    def __init__(self,radeg,dedeg,cz,czerr,absrmag,dwarfgiantdivide,fof_bperp=0.07,fof_blos=1.1,fof_sep=None, volume=None,pfof_Pth=0.01, center_mode='average',\
                     iterative_giant_only_groups=False, n_bootstraps=1000, rproj_fit_guess=None, rproj_fit_params = None, rproj_fit_multiplier=None,\
                     vproj_fit_guess = None, vproj_fit_params = None, vproj_fit_multiplier=None, vproj_fit_offset=0, gd_rproj_fit_guess=None, gd_rproj_fit_params = None,\
                     gd_rproj_fit_multiplier=None, gd_vproj_fit_guess=None, gd_vproj_fit_params = None, gd_vproj_fit_multiplier=None,gd_vproj_fit_offset=None,
                     gd_fit_bins=None,H0=100., Om0=0.3, Ode0=0.7, ncores=None, saveplotspdf=False, summary_page_savepath=None):
        """
        Identify galaxy groups in redshift space using the probabilistic G3 algorithm (Hutchens+2025, in prep.)
        A modified and extended version of the algorithm presented in Hutchens + 2023.

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
        """
        self.radeg=np.float32(radeg)
        self.dedeg=np.float32(dedeg)
        self.cz=np.float32(cz)
        self.czerr=np.float32(czerr)
        self.absrmag=np.float32(absrmag)
        assert (not np.isnan(self.radeg).any()), "RA values must not contain NaNs."
        assert (not np.isnan(self.dedeg).any()), "DEC values must not contain NaNs."
        assert (not np.isnan(self.cz).any()), "cz values must not contain NaNs."
        assert (not np.isnan(self.czerr).any()), "czerr values must not contain NaNs."
        assert (not np.isnan(self.absrmag).any()), "absrmag values must not contain NaNs."
        if (cz<20).all():
            print("WARNING: all input cz's are <20 km/s. Intepreting as z (not cz)...")
            self.cz = self.cz*SPEED_OF_LIGHT
            self.czerr = self.czerr*SPEED_OF_LIGHT
        self.g3grpid = np.zeros_like(radeg)-99.
        self.g3ssid = np.zeros_like(radeg)-99.
        self.H0 = H0
        self.Om0 = Om0
        self.Ode0 = Ode0
        self.cosmo = LambdaCDM(H0=H0,Om0=Om0,Ode0=Ode0)
        self.summary_page_savepath = summary_page_savepath
        self.saveplotspdf = saveplotspdf
        if self.summary_page_savepath is not None:
            self.make_summary_page = True
            self.PDF = PdfPages(self.summary_page_savepath)
        else:
            self.make_summary_page = False
        self.dwarfgiantdivide=dwarfgiantdivide
        self.fof_bperp = fof_bperp
        self.fof_blos = fof_blos
        self.fof_sep = fof_sep
        self.volume = volume
        self.pfof_Pth = pfof_Pth
        self.center_mode = center_mode
        self.iterative_giant_only_groups = iterative_giant_only_groups
        self.n_bootstraps = n_bootstraps
        self.rproj_fit_guess = rproj_fit_guess
        self.rproj_fit_params = rproj_fit_params
        self.rproj_fit_multiplier = rproj_fit_multiplier
        self.vproj_fit_guess = vproj_fit_guess
        self.vproj_fit_params = vproj_fit_params
        self.vproj_fit_multiplier = vproj_fit_multiplier
        self.vproj_fit_offset = vproj_fit_offset
        self.gd_rproj_fit_guess = gd_rproj_fit_guess
        self.gd_rproj_fit_params = gd_rproj_fit_params
        self.gd_rproj_fit_multiplier = gd_rproj_fit_multiplier
        self.gd_vproj_fit_guess = gd_vproj_fit_guess
        self.gd_vproj_fit_params = gd_vproj_fit_params
        self.gd_vproj_fit_multiplier = gd_vproj_fit_multiplier
        self.gd_vproj_fit_offset = gd_vproj_fit_offset
        self.gd_fit_bins = gd_fit_bins
        self.ncores = ncores
        self.giantcalbounds = (array32([0.05,0.05]), array32([1e4,1e4]))

    def find_groups(self):
        """
        `find_groups`: PG3 class method
        main function for performing group-finding

        Arguments:
            None (initiated by obj definition)
        Returns:
            g3grpid : array
                group ID number for each input galaxy`
            g3ssid : array
                group substructure ID for each input galaxy
            fof_sep : float
                Separation value (Mpc) used in FoF
            rproj_bestfit : array
                Best-fitting parameters for giant only PFoF calibration, projected radii
            rproj_bestfit_err : array
                Best-fitting parameters uncertainty for giant only PFoF calibration, projected radii
            vproj_bestfit : array
                Best-fitting parameters for giant only PFoF calibration, velocity
            vproj_bestfit_err : array
                Best-fitting parameters uncertainty for giant only PFoF calibration, velocity
            gdrproj_bestfit : array
                Best-fitting parameters for dwarf only IC calibration, projected radii
            gdrproj_bestfit_err : array
                Best-fitting parameters uncertainty for dwarf only IC calibration, projected radii
            gdvproj_bestfit : array
                Best-fitting parameters for dwarf only IC calibration, velocity
            gdvproj_bestfit_err : array
                Best-fitting parameters uncertainty for dwarf only IC calibration, velocity
        """
        self.giantonlyPFOF()
        self.get_giantFoF_calibrations()
        self.giantonlymerging()
        self.associateDwarfs()
        self.getBoundariesDwarfOnlyGroups()
        self.dwarfonlyIC()
        if self.make_summary_page:
            self.make_summary()
        return self.g3grpid, self.g3ssid, self.fof_sep, self.rproj_bestfit, self.rproj_bestfit_err, self.vproj_bestfit,\
                 self.vproj_bestfit_err, self.gd_rproj_bestfit,self.gd_rproj_bestfit_err, self.gd_vproj_bestfit, self.gd_vproj_bestfit_err 
        
    def giantonlyPFOF(self):
        """
        "giantonlyPFOF" : pg3 class method
        performs probabilstic fof for giant galaxies
        calculates `fof_sep` and assigns it to self is not pre-defined in __init__
        """
        self.giantsel = (self.absrmag<=self.dwarfgiantdivide)
        if self.fof_sep is not None:
            giantfofid = pfof_comoving(self.radeg[self.giantsel], self.dedeg[self.giantsel], self.cz[self.giantsel], self.czerr[self.giantsel],\
                         self.fof_bperp*self.fof_sep, self.fof_blos*self.fof_sep, self.pfof_Pth, H0=self.H0, Om0=self.Om0, Ode0=self.Ode0, ncores=self.ncores)
        else:
            self.fof_sep = (self.volume/np.sum(self.giantsel))**(1/3.)
            giantfofid = pfof_comoving(self.radeg[self.giantsel], self.dedeg[self.giantsel], self.cz[self.giantsel], self.czerr[self.giantsel], \
                         self.fof_bperp*self.fof_sep, self.fof_blos*self.fof_sep, self.pfof_Pth, H0=self.H0, Om0=self.Om0, Ode0=self.Ode0, ncores=self.ncores)
        self.g3grpid[self.giantsel] = giantfofid
        return None

    def get_giantFoF_calibrations(self):
        """
        get_giantFoF_calibrations : pg3 class method
        Gets calibrations from giant PFOF groups, to be used for
        giant-only merging and dwarf association (see H23).
        """
        if (self.rproj_fit_params is None) or (self.vproj_fit_params is None):
            if self.center_mode=='average' or self.center_mode=='giantaverage':
                giantgrpra, giantgrpdec, giantgrpz, _, _, zpdfs = prob_group_skycoords(self.radeg[self.giantsel], self.dedeg[self.giantsel], self.cz[self.giantsel]/SPEED_OF_LIGHT,\
                     self.czerr[self.giantsel]/SPEED_OF_LIGHT, self.g3grpid[self.giantsel])
                giantgrpcz = giantgrpz * SPEED_OF_LIGHT
            else:
                raise ValueError('Check group center definition (`center_mode`), only `average` or `giantaverage` (equivalent) currently supported')
            relvel = np.abs(giantgrpcz - self.cz[self.giantsel])/(1+giantgrpz) # from https://academic.oup.com/mnras/article/442/2/1117/983284#30931438
            grp_ctd = self.cosmo.comoving_transverse_distance(giantgrpz).value
            relprojdist = (grp_ctd+grp_ctd)*np.sin(angular_separation(giantgrpra, giantgrpdec, self.radeg[self.giantsel],self.dedeg[self.giantsel])/2.0)
            giantgrpn = multiplicity_function(self.g3grpid[self.giantsel], return_by_galaxy=True)
            uniqgiantgrpn, uniqindex = np.unique(giantgrpn, return_index=True)
            keepcalsel = np.where(uniqgiantgrpn>1)
            wavg_relprojdist = array32([weighted_median(relprojdist[np.where(giantgrpn==sz)], 1/self.czerr[np.where(giantgrpn==sz)]) for sz in uniqgiantgrpn[keepcalsel]])
            wavg_relvel = array32([weighted_median(relvel[np.where(giantgrpn==sz)], 1/self.czerr[np.where(giantgrpn==sz)]) for sz in uniqgiantgrpn[keepcalsel]])
            wavg_relprojdist_err = np.zeros_like(wavg_relprojdist)
            wavg_relvel_err = np.zeros_like(wavg_relvel)
            for ii,nn in enumerate(uniqgiantgrpn[keepcalsel]):
                df_ = pd.DataFrame({'czerr':self.czerr[np.where(giantgrpn==nn)], 'rpdist':relprojdist[np.where(giantgrpn==nn)], 'dv':relvel[np.where(giantgrpn==nn)]})
                resamples = [df_.sample(frac=1, replace=True) for ii in range(0,self.n_bootstraps)]
                wavg_relprojdist_err[ii] = np.std([weighted_median(resamp.rpdist, 1/resamp.czerr) for resamp in resamples])
                wavg_relvel_err[ii] = np.std([weighted_median(resamp.dv, 1/resamp.czerr) for resamp in resamples])
            self.rproj_bestfit, rproj_bestfit_cov = curve_fit(giantmodel, uniqgiantgrpn[keepcalsel], wavg_relprojdist,  p0=self.rproj_fit_guess, maxfev=5000,sigma=wavg_relprojdist_err, bounds = self.giantcalbounds, absolute_sigma=True)
            self.rproj_bestfit_err = np.sqrt(np.diag(rproj_bestfit_cov))
        else:
            self.rproj_bestfit = array32(self.rproj_fit_params)
            self.rproj_bestfit_err = np.zeros(2)*1.
        if self.vproj_fit_params is None:
            try:
                self.vproj_bestfit, vproj_bestfit_cov  = curve_fit(giantmodel, uniqgiantgrpn[keepcalsel], wavg_relvel,  p0=self.vproj_fit_guess, maxfev=5000,sigma=0.1*wavg_relvel_err, bounds = self.giantcalbounds, absolute_sigma=True)
                self.vproj_bestfit_err = np.sqrt(np.diag(vproj_bestfit_cov))
            except RuntimeError:
                print("Code failed at `get_giantFoF_calibrations` -- likely no Ngiants>1 groups.")
                exit()
        else:
            self.vproj_bestfit = array32(self.vproj_fit_params)
            self.vproj_bestfit_err = np.zeros(2)*1.
        
        self.rproj_boundary = lambda Ngiants:self.rproj_fit_multiplier*giantmodel(Ngiants, *self.rproj_bestfit)
        self.vproj_boundary = lambda Ngiants:self.vproj_fit_multiplier*giantmodel(Ngiants, *self.vproj_bestfit) + self.vproj_fit_offset
        if self.make_summary_page or self.saveplotspdf:
            self.fig1 = plot_rproj_vproj_1(uniqgiantgrpn, giantgrpn, relprojdist, wavg_relprojdist, wavg_relprojdist_err, self.rproj_bestfit, relvel,\
                wavg_relvel, wavg_relvel_err, self.vproj_bestfit, keepcalsel, self.rproj_fit_multiplier, self.vproj_fit_multiplier, self.vproj_fit_offset, self.saveplotspdf)
        return None

    def giantonlymerging(self):
        """
        giantonlymerging : pg3 class method
        merges giant-only groups according to calibrations
        in "get_giantonlyFoF_calibrations".
        """
        if self.iterative_giant_only_groups:
            revisedgiantgrpid = prob_iterative_combination_giants(self.radeg[self.giantsel],self.dedeg[self.giantsel],self.cz[self.giantsel]/SPEED_OF_LIGHT,\
                self.czerr[self.giantsel]/SPEED_OF_LIGHT,self.g3grpid[self.giantsel], self.rproj_boundary,self.vproj_boundary,self.pfof_Pth,self.cosmo)
            self.g3ssid[self.giantsel] = self.g3grpid[self.giantsel]
            self.g3grpid[self.giantsel] = revisedgiantgrpid
        else:
            pass
        return None

    def associateDwarfs(self):
        """
        associateDwarfs : pg3 class method
        Associates dwarfs to giant-only groups, using calibrations
        from "get_giantFoF_calibrations".
        """
        self.dwarfsel = (self.absrmag>self.dwarfgiantdivide)
        if self.center_mode=='average' or self.center_mode=='giantaverage':
            giantgrpra, giantgrpdec, giantgrpz, grpz16, grpz84, pdfdict = prob_group_skycoords(self.radeg[self.giantsel], self.dedeg[self.giantsel], self.cz[self.giantsel]/SPEED_OF_LIGHT,\
                    self.czerr[self.giantsel]/SPEED_OF_LIGHT, self.g3grpid[self.giantsel], True)
            giantgrpzerr = 0.5*(grpz84 - grpz16)
        else:
            raise ValueError("center_mode must be `average` or `giantaverage`")
        
        giantgrpn = multiplicity_function(self.g3grpid[self.giantsel],return_by_galaxy=True)
        dwarfassocid, _ = prob_faint_assoc(self.radeg[self.dwarfsel],self.dedeg[self.dwarfsel],self.cz[self.dwarfsel]/SPEED_OF_LIGHT,self.czerr[self.dwarfsel]/SPEED_OF_LIGHT,\
                            giantgrpra,giantgrpdec,giantgrpz,giantgrpzerr,pdfdict,self.g3grpid[self.giantsel],self.rproj_boundary(giantgrpn),self.vproj_boundary(giantgrpn),\
                             self.pfof_Pth, H0=self.H0,Om0=self.Om0,Ode0=self.Ode0)
        self.g3grpid[self.dwarfsel]=dwarfassocid
        print('Finished associating dwarfs to giant-only groups.')
        return None

    def getBoundariesDwarfOnlyGroups(self):
        """
        getBoundariesDwarfOnlyGroups : pg3 class method
        derives boundaries needed for dwarf-only iterative combiation
        see H23.
        """
        if (self.gd_rproj_fit_params is None) or (self.gd_vproj_fit_params is None):
            gdgrpn = multiplicity_function(self.g3grpid, return_by_galaxy=True)
            gdsel = np.logical_not(np.logical_or(self.g3grpid==-99., ((gdgrpn==1) & (self.absrmag>self.dwarfgiantdivide))))
            gdgrpra, gdgrpdec, gdgrpz, _, _, _ = prob_group_skycoords(self.radeg[gdsel],self.dedeg[gdsel],self.cz[gdsel]/SPEED_OF_LIGHT,self.czerr[gdsel]/SPEED_OF_LIGHT,\
                 self.g3grpid[gdsel])
            gdrelvel = SPEED_OF_LIGHT*np.abs(self.cz[gdsel]/SPEED_OF_LIGHT - gdgrpz)/(1+gdgrpz)
            ctd1 = self.cosmo.comoving_transverse_distance(gdgrpz).value
            ctd2 = self.cosmo.comoving_transverse_distance(self.cz[gdsel]/SPEED_OF_LIGHT).value
            gdrelprojdist = (ctd1 + ctd2) * np.sin(angular_separation(gdgrpra, gdgrpdec, self.radeg[gdsel], self.dedeg[gdsel])/2.0)
            gdn = gdgrpn[gdsel]
            gdtotalmag = get_int_mag(self.absrmag[gdsel], self.g3grpid[gdsel])
            binsel = np.where(np.logical_and(gdn>1, gdtotalmag>-24))
            gdmedianrproj, self.magbincenters,self.magbinedges,jk = center_binned_stats(gdtotalmag[binsel], gdrelprojdist[binsel], np.median, bins=self.gd_fit_bins)
            gdmedianrproj_err, jk, jk, jk = center_binned_stats(gdtotalmag[binsel], gdrelprojdist[binsel], sigmarange, bins=self.gd_fit_bins)
            gdmedianrelvel, jk, jk, jk = center_binned_stats(gdtotalmag[binsel], gdrelvel[binsel], np.median, bins=self.gd_fit_bins)
            gdmedianrelvel_err, jk, jk, jk = center_binned_stats(gdtotalmag[binsel], gdrelvel[binsel], sigmarange, bins=self.gd_fit_bins)
            nansel = np.isnan(gdmedianrproj)
        if (self.gd_rproj_fit_params is None):
            self.gd_rproj_bestfit, gd_rproj_cov=curve_fit(decayexp, self.magbincenters[~nansel], gdmedianrproj[~nansel], p0=self.gd_rproj_fit_guess)
            self.gd_rproj_bestfit_err = np.sqrt(np.diag(gd_rproj_cov))
        else:
            self.gd_rproj_bestfit = array32(gd_rproj_fit_params)
            self.gd_rproj_bestfit_err = np.zeros(len(gd_rproj_fit_params))*1.
        if (self.gd_vproj_fit_params is None):
            self.gd_vproj_bestfit, gd_vproj_cov=curve_fit(decayexp, self.magbincenters[~nansel], gdmedianrelvel[~nansel], p0=self.gd_vproj_fit_guess)
            self.gd_vproj_bestfit_err = np.sqrt(np.diag(gd_vproj_cov))
        else:
            self.gd_vproj_bestfit = array32(gd_vproj_fit_params)
            self.gd_vproj_bestfit_err = np.zeros(len(gd_vproj_fit_params))*1.
        self.rproj_for_iteration = lambda M: self.gd_rproj_fit_multiplier*decayexp(M, *self.gd_rproj_bestfit)
        self.vproj_for_iteration = lambda M: self.gd_vproj_fit_multiplier*decayexp(M, *self.gd_vproj_bestfit) + self.gd_vproj_fit_offset
        if self.make_summary_page or self.saveplotspdf: 
            self.fig2=plot_rproj_vproj_2(self.g3grpid, self.absrmag, gdsel, gdtotalmag, gdrelprojdist, gdrelvel, self.magbincenters, binsel, gdmedianrproj,\
                 gdmedianrelvel, self.gd_rproj_bestfit, self.gd_vproj_bestfit, self.gd_rproj_fit_multiplier, self.gd_vproj_fit_multiplier, self.gd_vproj_fit_offset, self.saveplotspdf)
        return None
        

    def dwarfonlyIC(self):
        """
        dwarfonlyIC : pg3 class method
        performs dwarf-only iterative combination to find dwarf-only groups.
        """
        assert (self.g3grpid[(self.absrmag<=self.dwarfgiantdivide)]!=-99.).all(), "Not all giants are grouped." 
        grpnafterassoc = multiplicity_function(self.g3grpid, return_by_galaxy=True)
        _ungroupeddwarf_sel = (self.absrmag>self.dwarfgiantdivide) & (grpnafterassoc==1)
        itassocid = dwarf_iterative_combination(self.radeg[_ungroupeddwarf_sel], self.dedeg[_ungroupeddwarf_sel], self.cz[_ungroupeddwarf_sel]/SPEED_OF_LIGHT, \
                    self.czerr[_ungroupeddwarf_sel]/SPEED_OF_LIGHT, self.absrmag[_ungroupeddwarf_sel],self.rproj_for_iteration, self.vproj_for_iteration,\
                     self.pfof_Pth, self.cosmo, starting_id=np.max(self.g3grpid)+1)
        self.g3grpid[_ungroupeddwarf_sel]=itassocid
        return None

    def make_summary(self):
        """
        make_summary : pg3 class method
        Makes summary PDF with group-finding-related plots/statistics.
        """
        if self.make_summary_page: 
            figs = get_extra_biopage_plots(self.g3grpid,self.radeg,self.dedeg,self.cz/SPEED_OF_LIGHT,self.czerr/SPEED_OF_LIGHT,self.absrmag,\
                    self.dwarfgiantdivide,self.volume,self.H0)
            for fig in figs:
                self.PDF.savefig(fig)
            self.PDF.savefig(self.fig1)
            self.PDF.savefig(self.fig2)
            self.PDF.close()
            plt.close(self.fig1)
            plt.close(self.fig2)
        return None

    def get_grpn(self, return_by_galaxy):
        """
        get_grpn : pg3 class method
        Get Group N for each group, listed by galaxy or by group.
        see multiplicity_function
        """
        return multiplicity_function(self.g3grpid, return_by_galaxy=return_by_galaxy)

    def get_dwarf_grpn_by_group(self):
        assert (self.absrmag<0).all(), 'M* not currently supported for `get_dwarf_grpn_by_group`.'
        df = pd.DataFrame({'grp':self.g3grpid, 'absrmag':self.absrmag})
        dogrps = df.groupby('grp').filter(lambda g: (g.absrmag>self.dwarfgiantdivide).all())
        return multiplicity_function(dogrps.grp.to_numpy(), return_by_galaxy=False)

    def get_group_centers(self, return_z_pdfs):
        """
        get_group_centers : pg3 class method
        Get group center RA/Dec/z/z_pdf for PG3 groups
        """
        return prob_group_skycoords(self.radeg, self.dedeg, self.cz/SPEED_OF_LIGHT, self.czerr/SPEED_OF_LIGHT, self.g3grpid, return_z_pdfs)


#################################################
#################################################
#################################################
def numeric_integration_pfof_vectorized(zmesh, z1, sig1, z2, sig2, VL_lower, VL_upper):
    zmesh = zmesh[None, :]
    g1pdf = np.exp(-0.5 * ((zmesh - z1[:, None]) / sig1[:, None])**2) / (sig1[:, None] * np.sqrt(2 * np.pi))
    den = 1.4142 * sig2[:, None]
    erf_term = scipy_erf((z2[:, None] - zmesh + VL_lower[:, None]) / den) - scipy_erf((z2[:, None] - zmesh - VL_upper[:, None]) / den)
    P12 = np.trapz(0.5 * g1pdf * erf_term, zmesh, axis=1)
    return P12

def pfof_comoving(ra, dec, cz, czerr, perpll, losll, Pth, H0=100., Om0=0.3, Ode0=0.7, printConf=True, ncores=None):
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
        ncores (int, # of cores for multiprocessing, default None = no multiprocessing)
    Returns:
        grpid (np.array): list containing unique group ID numbers for each target in the input coordinates.
                The list will have shape len(ra).
    -----
    """
    cosmo = LambdaCDM(H0=H0, Om0=Om0, Ode0=Ode0) # this puts everything in "per h" units.
    c=SPEED_OF_LIGHT
    Ngalaxies = len(ra)
    ra = np.float32(ra)
    dec = np.float32(dec)
    zz = np.float32(cz/c)
    zzerr = np.float32(czerr/c)
    assert (len(ra)==len(dec) and len(dec)==len(cz)),"RA/Dec/cz arrays must equivalent length."

    phi = (ra * np.pi/180.)
    theta = (np.pi/2. - dec*(np.pi/180.))
    transv_cmvgdist = (cosmo.comoving_transverse_distance(zz).value)
    los_cmvgdist = (cosmo.comoving_distance(zz).value)
    dc_upper = los_cmvgdist + losll
    dc_lower = los_cmvgdist - losll

    meshZ = np.arange(0.5*np.min(zz),1.5*np.max(zz),np.min(zzerr)/3, dtype=np.float32) # resolution adapts to dataset
    z_dc_interp = interp1d(cosmo.comoving_distance(meshZ).value, meshZ)  
    VL_lower = np.float32(zz - z_dc_interp(dc_lower))
    VL_upper = np.float32(z_dc_interp(dc_upper) - zz)
    friendship = np.zeros((Ngalaxies, Ngalaxies),dtype=np.int8)

    # Compute on-sky perpendicular distance
    half_angle = np.arcsin((np.sin((theta[:,None]-theta)/2.0)**2.0 + np.sin(theta[:,None])*np.sin(theta)*np.sin((phi[:,None]-phi)/2.0)**2.0)**0.5)
    column_transv_cmvgdist = transv_cmvgdist[:, None]
    dperp = (column_transv_cmvgdist + transv_cmvgdist) * half_angle # In Mpc

    # Compute line-of-sight probabilities
    prob_dlos=np.zeros((Ngalaxies, Ngalaxies),dtype=np.float32)
    np.fill_diagonal(prob_dlos,1)
    
    i_idx, j_idx = np.triu_indices(Ngalaxies, k=1)
    mask = dperp[i_idx, j_idx] <= perpll
    i_idx, j_idx = i_idx[mask], j_idx[mask]

    zz_i, zz_j = zz[i_idx], zz[j_idx]
    zzerr_i, zzerr_j = zzerr[i_idx], zzerr[j_idx]
    VL_lower_i, VL_upper_i = VL_lower[i_idx], VL_upper[i_idx]

    vals = numeric_integration_pfof_vectorized(meshZ, zz_i, zzerr_i, zz_j, zzerr_j, VL_lower_i, VL_upper_i)

    prob_dlos[i_idx, j_idx] = vals
    prob_dlos[j_idx, i_idx] = vals

    # Produce friendship matrix and return groups
    index = np.where(np.logical_and(prob_dlos>Pth, dperp<=perpll))
    friendship[index]=1
    assert np.all(np.abs(friendship-friendship.T) < 1e-8), "Friendship matrix must be symmetric."

    if printConf:
        print('PFoF complete!')
    return collapse_friendship_matrix(friendship)
    #return connected_components(csr_array(friendship))[1]

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
    friendship_matrix=array32(friendship_matrix)
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

def get_median_eCDF(xx,aa,percentiles=[0.5]):
    """
    xx : values in distribution
    aa : PDF of data (arbitrary units)
    """
    #xx=np.array(xx)
    #aa=np.array(aa)
    cs = np.cumsum(aa)
    values = [xx[np.argmin(np.abs(cs-perc*cs[-1]))] for perc in percentiles]
    if len(values)==1:
        return values[0]
    else:
        return values

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
       groupz16 : 16th percentile of galaxy i's group redshift distribution.
       groupz84 : 84th percentile of galaxy i's group redshift distribution.
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
    galaxyxx = np.float32(np.expand_dims((np.sin(galaxytheta)*np.cos(galaxyphi)),axis=1)) # equivalent to [:,np.newaxis]
    galaxyyy = np.float32(np.expand_dims((np.sin(galaxytheta)*np.sin(galaxyphi)),axis=1))
    galaxyzz = np.float32(np.expand_dims(((np.cos(galaxytheta))),axis=1))
    # Prepare output arrays
    uniqidnumbers = np.int32(np.unique(galaxygrpid))
    groupra = np.zeros(ngalaxies, dtype=np.float32)
    groupdec = np.zeros(ngalaxies, dtype=np.float32)
    groupz = np.zeros(ngalaxies, dtype=np.float32)
    groupz16 = np.zeros(ngalaxies, dtype=np.float32)
    groupz84 = np.zeros(ngalaxies, dtype=np.float32)
    cspeed=SPEED_OF_LIGHT
    galaxyz = np.expand_dims(galaxyz,axis=1)
    galaxyzerr = np.expand_dims(galaxyzerr,axis=1) 
    for i,uid in enumerate(uniqidnumbers):
        sel=np.where(galaxygrpid==uid)
        if len(sel[0])==1:
            groupra[sel] = galaxyra[sel]
            groupdec[sel] = galaxydec[sel]
            groupz[sel] = galaxyz[sel]#*cspeed
            groupz16[sel] = galaxyz[sel]-galaxyzerr[sel]
            groupz84[sel] = galaxyz[sel]+galaxyzerr[sel]
        else:
            mesh_spacing = np.min(galaxyzerr[sel])/10. # 10 points per standard deviation
            gx,gy,gz = galaxyz[sel]*galaxyxx[sel], galaxyz[sel]*galaxyyy[sel], galaxyz[sel]*galaxyzz[sel]
            gxerr,gyerr,gzerr = galaxyzerr[sel]*galaxyxx[sel], galaxyzerr[sel]*galaxyyy[sel], galaxyzerr[sel]*galaxyzz[sel]
            xmesh = np.arange(np.min(gx)-5*np.max(np.abs(gxerr)), np.max(gx)+5*np.max(np.abs(gxerr)), mesh_spacing, dtype=np.float32)
            ymesh = np.arange(np.min(gy)-5*np.max(np.abs(gyerr)), np.max(gy)+5*np.max(np.abs(gyerr)), mesh_spacing, dtype=np.float32) 
            zmesh = np.arange(np.min(gz)-5*np.max(np.abs(gzerr)), np.max(gz)+5*np.max(np.abs(gzerr)), mesh_spacing, dtype=np.float32)
            x16,xcen,x84 = get_median_eCDF(xmesh, np.sum(gauss_vectorized(xmesh, gx, gxerr),axis=0), percentiles=[0.16,0.5,0.84])
            y16,ycen,y84 = get_median_eCDF(ymesh, np.sum(gauss_vectorized(ymesh, gy, gyerr),axis=0), percentiles=[0.16,0.5,0.84])
            z16,zcen,z84 = get_median_eCDF(zmesh, np.sum(gauss_vectorized(zmesh, gz, gzerr),axis=0),  percentiles=[0.16,0.5,0.84])
            redshiftcen = np.sqrt(xcen*xcen + ycen*ycen + zcen*zcen)
            redshift16 = np.sqrt(x16*x16 + y16*y16 + z16*z16)
            redshift84 = np.sqrt(x84*x84 + y84*y84 + z84*z84)
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
            groupz16[sel] = redshift16
            groupz84[sel] = redshift84
    if return_z_pdfs:
        try:
            zmesh = np.arange(0, np.max(galaxyz)+0.1, 1/cspeed, dtype=np.float32)
        except MemoryError:
            print("MemoryWarning: zmesh is too fine at line 523 in prob_g3groupfinder; trying at 20 km/s resolution")
            zmesh = np.arange(0, np.max(galaxyz)+0.1, 20/cspeed, dtype=np.float32)
        z_pdfs = np.zeros((len(uniqidnumbers), len(zmesh)), dtype=np.float32)
        for i,uid in enumerate(uniqidnumbers):
            sel=(galaxygrpid==uid)
            z_pdfs[i]=np.sum(gauss_vectorized(zmesh, galaxyz[sel], galaxyzerr[sel]), axis=0, dtype=np.float32)
            z_pdfs[i]=np.float32(z_pdfs[i]/simpson(z_pdfs[i],zmesh))
        pdfoutput = {'zmesh':zmesh, 'pdf':z_pdfs, 'grpid':uniqidnumbers}
    else:
        pdfoutput=None
    return groupra, groupdec, groupz, groupz16, groupz84, pdfoutput


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
    galaxyra=array32(galaxyra)
    galaxydec=array32(galaxydec)
    galaxyz=array32(galaxyz)
    galaxyzerr=array32(galaxyzerr)
    giantfofid=array32(giantfofid)
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
    groupra, groupdec, groupz, _, _, _ = prob_group_skycoords(galaxyra, galaxydec, galaxyz, galaxyzerr, grpid)
    groupN = multiplicity_function(grpid,return_by_galaxy=True)
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
    coords = array32([xmpc, ympc, zmpc]).T
    kdt = cKDTree(coords)
    _, nnind = kdt.query(coords,k=2)
    nnind=nnind[:,1] # ignore self-match
    # go through potential groups and adjust membership for input galaxies
    reciprocal = np.arange(len(uniqgrpid))==nnind[nnind]
    alreadydone=np.zeros(len(uniqgrpid), dtype=int)
    for idx in np.where(reciprocal & ~alreadydone)[0]:
        nbridx = nnind[idx]
        if alreadydone[nbridx]:
            continue
        uid = uniqgrpid[idx]
        nbr_uid = uniqgrpid[nbridx]
        Gpgalsel = (grpid==uid)
        GNNgalsel = (grpid==nbr_uid)

        combinedra = np.hstack((galaxyra[Gpgalsel], galaxyra[GNNgalsel]))
        combineddec = np.hstack((galaxydec[Gpgalsel], galaxydec[GNNgalsel]))
        combinedz = np.hstack((galaxyz[Gpgalsel], galaxyz[GNNgalsel]))
        combinedzerr = np.hstack((galaxyzerr[Gpgalsel], galaxyzerr[GNNgalsel]))
        combinedgalgrpid = np.hstack((grpid[Gpgalsel], grpid[GNNgalsel]))
        if prob_giants_fit_in_group(combinedra, combineddec, combinedz, combinedzerr, combinedgalgrpid, rprojboundary, vprojboundary, pthresh, cosmo):
            refinedgrpid[GNNgalsel] = grpid[Gpgalsel][0]
            alreadydone[idx] = 1
            alreadydone[nbridx] = 1
        else:
            alreadydone[idx] = 1
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
    seed2sel = (combinedgalgrpid==uniqIDnums[1])
    seed1grpra,seed1grpdec,seed1grpz,_,_,seed1pdf = prob_group_skycoords(combinedra[seed1sel],combineddec[seed1sel],combinedz[seed1sel], combinedzerr[seed1sel] ,combinedgalgrpid[seed1sel],True)
    seed2grpra,seed2grpdec,seed2grpz,_,_,seed2pdf = prob_group_skycoords(combinedra[seed2sel],combineddec[seed2sel],combinedz[seed2sel], combinedzerr[seed2sel] ,combinedgalgrpid[seed2sel],True)
    allgrpra,allgrpdec,allgrpz,_,_,_ = prob_group_skycoords(combinedra, combineddec, combinedz, combinedzerr, np.zeros(len(combinedra)), False)
    totalgrpN = len(seed1grpra)+len(seed2grpra)

    # Precompute comoving distances
    allgrpz0_dist = cosmo.comoving_transverse_distance(allgrpz[0]).to_value()
    seed1grpz0_dist = cosmo.comoving_transverse_distance(seed1grpz[0]).to_value()
    seed2grpz0_dist = cosmo.comoving_transverse_distance(seed2grpz[0]).to_value()

    # Compute radial separations
    angsep1 = angular_separation(allgrpra[0], allgrpdec[0], seed1grpra[0], seed1grpdec[0])
    angsep2 = angular_separation(allgrpra[0], allgrpdec[0], seed2grpra[0], seed2grpdec[0])
    seed1radialsep = (seed1grpz0_dist + allgrpz0_dist) * (angsep1 / 2.)
    seed2radialsep = (seed2grpz0_dist + allgrpz0_dist) * (angsep2 / 2.)
    Rp = rprojboundary(totalgrpN)
    fitingroup1 = seed1radialsep < Rp
    fitingroup2 = seed2radialsep < Rp 

    if fitingroup1 and fitingroup2:
        eps_z = (1+seed1grpz[0])/SPEED_OF_LIGHT * vprojboundary(totalgrpN)
        pcombine = dbint_D1_D2(seed1pdf['zmesh'],seed1pdf['pdf'][0],seed2pdf['zmesh'],seed2pdf['pdf'][0],eps_z)
        fitingroup = (pcombine > pthresh)
    else:
        fitingroup = False
    return fitingroup

#######################################################################
#######################################################################
#######################################################################
## Dwarf galaxy association code

def prob_faint_assoc(faintra, faintdec, faintz, faintzerr, grpra, grpdec, grpz, grpzerr, grpzpdf, grpid, radius_boundary, velocity_boundary, Pth, H0=100., Om0=0.3, Ode0=0.7):
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
        Redshift of group center.  Length matches `grpra`.
    grpzerr : iterable
        Redshift uncertainty on group center.
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
    velocity_boundary=np.asarray(velocity_boundary).astype(np.float32)
    radius_boundary=np.asarray(radius_boundary).astype(np.float32)
    Nfaint = len(faintra)
    assoc_grpid = np.zeros(Nfaint,dtype=int)
    assoc_flag = np.zeros(Nfaint,dtype=int)
    prob_values=np.zeros(Nfaint,dtype=np.float32)
    
    # resize group coordinates to be the # of groups, not # galaxies
    grpid, uniqind, counts = np.unique(grpid, return_index=True, return_counts=True)
    grpra = grpra[uniqind]
    grpdec = grpdec[uniqind]
    grpz = grpz[uniqind]
    grpzerr = grpzerr[uniqind]
    velocity_boundary=velocity_boundary[uniqind]
    radius_boundary=radius_boundary[uniqind]
    zrange = (1+grpz) * velocity_boundary/SPEED_OF_LIGHT

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
    grpzpdf_dict = {gid: pdf for gid, pdf in zip(grpzpdf['grpid'], grpzpdf['pdf'])}

    for gg in range(len(grpid)):
        cc = counts[gg]
        gid = grpid[gg]
        mask = (Rp[:, gg] < radius_boundary[gg])
        indices = np.where(mask)[0]
        if len(indices) == 0:
            continue
       
        for fg in indices:
            if cc==1:
                Poverlap = dbint_doublegauss(grpzpdf['zmesh'], grpz[gg], grpzerr[gg], faintz[fg], faintzerr[gg], zrange[gg])
            else:    
                Poverlap = dbint_D1_D2(grpzpdf['zmesh'], grpzpdf_dict.get(gid,None), grpzpdf['zmesh'], gauss_vectorized(grpzpdf['zmesh'], faintz[fg], faintzerr[fg]), zrange[gg])
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
    # assign group ID numbers to galaxies that didn't associate
    still_isolated = np.where(assoc_grpid==0)
    assoc_grpid[still_isolated]=np.arange(np.max(grpid)+1, np.max(grpid)+1+len(still_isolated[0]), 1)
    assoc_flag[still_isolated]=-1
    return assoc_grpid, assoc_flag

def dbint_doublegauss(zmesh, z1, sig1, z2, sig2, g_or_e):
    """
    Numerically integrate to find PFoF linking probability (sec 3, eq 6, Hutchens et al.).
    This is for the simpler case of two Gaussian PDFs.

    Parameters
    ------------------------------------------
    zmesh - array of redshifts to integrate over
    z1 - redshift of galaxy 1
    sig1 - redshift uncertainty of galaxy 2
    z2 - redshift of galaxy 2
    sig2 - redshift uncertainty of galaxy 2
    VL_lower - lower bound linking length, redshift units
    VL_upper - upper bound linking length, redshift units

    Returns 
    ------------------------------------------
    P12 - linking probability
    """
    g1pdf = gauss_vectorized(zmesh, z1, sig1)
    den = 1.4142*sig2
    erf_term = scipy_erf((z2 - zmesh + g_or_e)/den) - scipy_erf((z2 - zmesh - g_or_e)/den)
    P12 = np.trapz(0.5 * g1pdf * erf_term, zmesh)
    return P12 

def dbint_D1_D2(z1, s1pdf, z2, s2pdf, g_or_e):
    """
    Numerically calculate double integrals of D1(z)*D2(zprime) for
    probabilistic calculations. D1 and D2 are redshift PDFs.

    Parameters
    ----------------
    z1 : array
        Redshift (z) mesh corresponding to D1.
    s1pdf : array
        PDF values for D1.
    z2 : array
        Redshift (z) mesh corresponding to D2.
    s2pdf : array
        PDF values for D2.
    g_or_e: float
        Redshift range to search around, called gamma or epsilon
        in this code.

    Returns
    -----------------
    P12: float
        Integration result P_12 for D1 and D2 given g_or_e.
    """
    cum_D2 = np.zeros(len(z2))
    cum_D2[1:] = np.cumsum((s2pdf[1:] + s2pdf[:-1]) / 2 * np.diff(z2))
    lower = np.searchsorted(z2, z1 - g_or_e, side='left')
    upper = np.searchsorted(z2, z1 + g_or_e, side='right')
    lower = np.clip(lower, 0, len(z2)-1)
    upper = np.clip(upper, 0, len(z2)-1)
    f_of_z = cum_D2[upper] - cum_D2[lower]
    P12 = np.trapz(s1pdf * f_of_z, z1)
    return P12

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
    galaxyra = array32(galaxyra)
    galaxydec = array32(galaxydec)
    galaxyz = array32(galaxyz)
    galaxyzerr = array32(galaxyzerr)
    galaxymag = array32(galaxymag)
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
    groupra, groupdec, groupz, _, _, _ = prob_group_skycoords(galaxyra, galaxydec, galaxyz, galaxyzerr, grpid)
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
    coords = array32([xmpc, ympc, zmpc]).T
    kdt = cKDTree(coords)
    nndist, nnind = kdt.query(coords,k=2)
    nndist=nndist[:,1] # ignore self match
    nnind=nnind[:,1]
    reciprocal = np.arange(len(uniqgrpid))==nnind[nnind]
    alreadydone=np.zeros(len(uniqgrpid), dtype=int)
    for idx in np.where(reciprocal & ~alreadydone)[0]:
        nbridx = nnind[idx]
        if alreadydone[nbridx]:
            continue
        uid = uniqgrpid[idx]
        nbr_uid = uniqgrpid[nbridx]
        Gpgalsel = (grpid==uid)
        GNNgalsel = (grpid==nbr_uid)

        combinedra = np.hstack((galaxyra[Gpgalsel], galaxyra[GNNgalsel]))
        combineddec = np.hstack((galaxydec[Gpgalsel], galaxydec[GNNgalsel]))
        combinedz = np.hstack((galaxyz[Gpgalsel], galaxyz[GNNgalsel]))
        combinedzerr = np.hstack((galaxyzerr[Gpgalsel], galaxyzerr[GNNgalsel]))
        combinedgalgrpid = np.hstack((grpid[Gpgalsel], grpid[GNNgalsel]))
        combinedmag = np.hstack((galaxymag[Gpgalsel], galaxymag[GNNgalsel]))
        if dwarfic_fit_in_group(combinedra, combineddec, combinedz, combinedzerr, combinedgalgrpid, combinedmag, rprojboundary,vprojboundary, pthresh, cosmo):
            associd[GNNgalsel] = grpid[Gpgalsel][0]
            alreadydone[idx] = 1
            alreadydone[nbridx] = 1
        else:
            alreadydone[idx] = 1
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
    seed2sel = (galgrpid==uniqIDnums[1])
    seed1grpra,seed1grpdec,seed1grpz,_,_,seed1pdf = prob_group_skycoords(galra[seed1sel],galdec[seed1sel],galz[seed1sel],galzerr[seed1sel],galgrpid[seed1sel], True)
    seed2grpra,seed2grpdec,seed2grpz,_,_,seed2pdf = prob_group_skycoords(galra[seed2sel],galdec[seed2sel],galz[seed2sel],galzerr[seed2sel],galgrpid[seed2sel], True)
    allgrpra,allgrpdec,allgrpz,_,_,_ = prob_group_skycoords(galra, galdec, galz, galzerr, np.zeros(len(galra)), False)

    debug_condition = ((allgrpra[0]>185.14) and (allgrpra[0]<185.17) and (allgrpdec[0]>20.65) and (allgrpdec[0]<20.68))

    # Compute radial separations
    angsep1 = angular_separation(allgrpra[0], allgrpdec[0], seed1grpra[0], seed1grpdec[0])
    angsep2 = angular_separation(allgrpra[0], allgrpdec[0], seed2grpra[0], seed2grpdec[0])
    allgrpz0_dist = cosmo.comoving_transverse_distance(allgrpz[0]).to_value()
    seed1grpz0_dist = cosmo.comoving_transverse_distance(seed1grpz[0]).to_value()
    seed2grpz0_dist = cosmo.comoving_transverse_distance(seed2grpz[0]).to_value()
    seed1radialsep = (seed1grpz0_dist + allgrpz0_dist) * (angsep1 / 2.)
    seed2radialsep = (seed2grpz0_dist + allgrpz0_dist) * (angsep2 / 2.)
    Rp = rprojboundary(memberintmag)[0]
    fitingroup1 = seed1radialsep < Rp
    fitingroup2 = seed2radialsep < Rp 
    if fitingroup1 and fitingroup2:
        gamma_z = (1+seed1grpz[0])/SPEED_OF_LIGHT * vprojboundary(memberintmag)[0]
        pcombine = dbint_D1_D2(seed1pdf['zmesh'],seed1pdf['pdf'][0],seed2pdf['zmesh'],seed2pdf['pdf'][0],gamma_z)
        fitingroup = (pcombine > pthresh)
    else:
        fitingroup = False
    return fitingroup

def plot_rproj_vproj_1(uniqgiantgrpn, giantgrpn, relprojdist, wavg_relprojdist, wavg_relprojdist_err, rproj_bestfit, relvel, wavg_relvel, wavg_relvel_err, vproj_bestfit, keepcalsel, rproj_mult, vproj_mult, vproj_offs, saveplotspdf):
    fig,axs=plt.subplots(figsize=doublecolsize, ncols=2)
    tx = np.linspace(1,30,500)
    sel = np.where(giantgrpn>1)
    axs[0].plot(giantgrpn[sel], relprojdist[sel], 'r.', markersize=2, alpha=0.5, label='ECO Giant Galaxies',zorder=0, rasterized=False)
    axs[0].errorbar(uniqgiantgrpn[keepcalsel], wavg_relprojdist, wavg_relprojdist_err, fmt='^', color='k', label=r'$R_{\rm proj}$',zorder=0)
    axs[0].plot(tx, giantmodel(tx,*rproj_bestfit), color='blue', label=r'$1R_{\rm proj}^{\rm fit}$',zorder=2)
    axs[0].plot(tx, rproj_mult*giantmodel(tx,*rproj_bestfit), color='green', label=str(rproj_mult)+r'$R_{\rm proj}^{\rm fit}$', linestyle='dashed',zorder=3)
    axs[0].plot(tx, giantmodel(tx,*(3.06e-1,4.16e-1)), color='gray', label=r'$1R_{\rm proj}^{\rm fit}$ from H23',zorder=2)
    
    axs[1].plot(giantgrpn[sel], relvel[sel], 'r.', markersize=2, alpha=0.5, label='ECO Giant Galaxies',zorder=0, rasterized=False)
    axs[1].errorbar(uniqgiantgrpn[keepcalsel], wavg_relvel, wavg_relvel_err, fmt='^', color='k', label=r'$\Delta v_{\rm proj}$',zorder=0)
    axs[1].plot(tx, giantmodel(tx,*vproj_bestfit), color='blue', label=r'$1\Delta v_{\rm proj}^{\rm fit}$',zorder=2)
    axs[1].plot(tx, vproj_mult*giantmodel(tx,*vproj_bestfit)+vproj_offs, color='green', label=str(vproj_mult)+r'$\Delta v_{\rm proj}^{\rm fit} +$' + str(vproj_offs) + r' km s$^{-1}$',zorder=3, linestyle='dashed')
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
    axs[1].set_ylim(0,2000)
    plt.tight_layout()
    if saveplotspdf: plt.savefig("../figures/rproj_vproj_cal.pdf",dpi=300)
    return fig

def plot_rproj_vproj_2(ecog3grp, ecoabsrmag, ecogdsel, ecogdtotalmag, ecogdrelprojdist, ecogdrelvel, magbincenters, binsel, gdmedianrproj, gdmedianrelvel, poptr, poptv,\
    gdrprojfit_mult, gdvprojfit_mult, gdvprojfit_offs, saveplotspdf):
    tx = np.linspace(-27,-17,100)
    fig, (ax,ax1) = plt.subplots(ncols=2, figsize=doublecolsize)
    giantgrpn = array32([np.sum((ecoabsrmag[ecogdsel][ecog3grp[ecogdsel]==gg]<-19.5)) for gg in ecog3grp[ecogdsel]])
    sel_ = np.where(np.logical_and(giantgrpn==1,ecogdtotalmag>-24))
    ax.plot(ecogdtotalmag[sel_], ecogdrelprojdist[sel_], '.', color='mediumorchid', alpha=0.6, label=r'ECO $N_{\rm giants}=1$ Group Galaxies', rasterized=True)
    sel_ = np.where(np.logical_and(giantgrpn==2,ecogdtotalmag>-24))
    ax.plot(ecogdtotalmag[sel_], ecogdrelprojdist[sel_], '.', color='darkorange', alpha=0.6, label=r'ECO $N_{\rm giants}=2$ Group Galaxies', rasterized=True)
    sel_ = np.where(np.logical_and(giantgrpn>2,ecogdtotalmag>-24))
    ax.plot(ecogdtotalmag[sel_], ecogdrelprojdist[sel_], '.', color='slategrey', alpha=0.6, label=r'ECO $N_{\rm giants}\geq3$ Group Galaxies', rasterized=True)
    #ax.errorbar(magbincenters, gdmedianrproj, yerr=gdmedianrproj_err, fmt='k^', label=r'Medians ($R_{\rm proj}^{\rm gi,\,dw}$)', rasterized=False, zorder=15)
    ax.errorbar(magbincenters, gdmedianrproj, yerr=None, fmt='k^', label=r'Medians ($R_{\rm proj}^{\rm gi,\,dw}$)', rasterized=True, zorder=15)
    ax.plot(tx, decayexp(tx, *(3.42e-2,5.1e-1)), color='k',label=r'$1R_{\rm proj,\, fit}^{\rm gi,\, dw}$ from H23')
    ax.plot(tx, 1*decayexp(tx,*poptr), color='red', label=r'$1R_{\rm proj,\,fit}^{\rm gi,\, dw}$', rasterized=True)
    ax.plot(tx, gdrprojfit_mult*decayexp(tx,*poptr), color='blue', label=str(gdrprojfit_mult)+r'$R_{\rm proj,\,fit}^{\rm gi,\, dw}$', rasterized=True,linestyle='--')
    #ax.plot(tx, 3*decayexp(tx,*poptr), label=r'$3R_{\rm proj,\,fit}^{\rm gi,\, dw}$', rasterized=False)
    ax.set_xlabel(r"Integrated $M_r$ of Giant + Dwarf Members")
    ax.set_ylabel(r"Projected Distance from Galaxy to Group Center [Mpc]")
    ax.legend(loc='best',fontsize=8,framealpha=0.5)
    ax.set_xlim(-24.1,-17)
    xrange=[-24,-16]
    xticks=np.arange(xrange[0],xrange[1],1)
    ax.set_xticks(xticks)
    ax.set_ylim(0,0.8)
    ax.invert_xaxis()

    #ax1.plot(ecogdtotalmag[binsel], ecogdrelvel[binsel], '.', alpha=0.6, label='ECO Giant-Hosting Group Galaxies', rasterized=False, color='palegreen')
    #ax1.errorbar(magbincenters, gdmedianrelvel, yerr=gdmedianrelvel_err, fmt='k^',label=r'Medians ($\Delta v_{\rm proj}^{\rm gi,\,dw}$)', rasterized=False, zorder=15)
    ax1.errorbar(magbincenters, gdmedianrelvel, yerr=None, fmt='k^',label=r'Medians ($\Delta v_{\rm proj}^{\rm gi,\,dw}$)', rasterized=True, zorder=15)
    sel_ = np.where(np.logical_and(giantgrpn==1,ecogdtotalmag>-24))
    ax1.plot(ecogdtotalmag[sel_], ecogdrelvel[sel_], '.', color='mediumorchid', alpha=0.6, label=r'ECO $N_{\rm giants}=1$ Group Galaxies', rasterized=True)
    sel_ = np.where(np.logical_and(giantgrpn==2,ecogdtotalmag>-24))
    ax1.plot(ecogdtotalmag[sel_], ecogdrelvel[sel_], '.', color='darkorange', alpha=0.6, label=r'ECO $N_{\rm giants}=2$ Group Galaxies', rasterized=True)
    sel_ = np.where(np.logical_and(giantgrpn>2,ecogdtotalmag>-24))
    ax1.plot(ecogdtotalmag[sel_], ecogdrelvel[sel_], '.', color='slategrey', alpha=0.6, label=r'ECO $N_{\rm giants}\geq3$ Group Galaxies', rasterized=True)
    ax1.plot(tx, decayexp(tx, *(1.97e1,4.16e-1)), color='k',label=r'1$\Delta v_{\rm proj,\, fit}^{\rm gi,\, dw}$ from H23')
    ax1.plot(tx, decayexp(tx, *poptv), color='red', label=r'$1\Delta v_{\rm proj,\, fit}^{\rm gi,\, dw}$', rasterized=True)
    ax1.plot(tx, gdvprojfit_mult*decayexp(tx, *poptv)+gdvprojfit_offs, color='blue', label=str(gdvprojfit_mult)+r'$\Delta v_{\rm proj,\, fit}^{\rm gi,\, dw}$+' + str(gdvprojfit_offs) + r' km s$^{-1}$', rasterized=True, linestyle='--')
    ax1.set_ylabel(r"Relative Velocity from Galaxy to Group Center [km s$^{-1}]$")
    ax1.set_xlabel(r"Integrated $M_r$ of Giant + Dwarf Members")
    ax1.set_xlim(-24.1,-17)
    ax1.set_ylim(0,2000)
    ax1.invert_xaxis()
    ax1.set_xticks(xticks)
    ax1.legend(loc='best',fontsize=8, framealpha=0.8)
    plt.tight_layout()
    if saveplotspdf: plt.savefig("../figures/itercombboundaries.pdf")
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
    plt.hist(grpn, bins=hbins, weights=1/volume+np.zeros_like(grpn))
    nn = np.linspace(0,45,50)
    plt.plot(nn, (10**(3.9*np.exp(-0.13*nn)))/(191958.08 / (H0/100.)**3), color='k',label=r'ECO (H23), $z=0$, $M_r \leq -17.33$')
    plt.xlim(0.5,40.5)
    plt.yscale('log')
    plt.xlabel(r'Group $N_{\rm galaxies}$')
    plt.ylabel(r'Comoving Number Density of Groups [Mpc$^{-3}$]')
    n_high_N = len(grpn[grpn>np.max(hbins)])
    if n_high_N>0:
        plt.annotate(xy=(0.5,0.6), xycoords='axes fraction', text=f'+{n_high_N} N>{np.max(hbins)+0.5} Groups')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.close()

    # ra/dec
    fig2,axs = plt.subplots(ncols=2,figsize=(11,5))
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
    eco_lf=(array32([0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       8.93424203e-06, 7.14739417e-06, 5.36054358e-06, 1.78684804e-05,
       4.46712438e-05, 7.50476102e-05, 1.50095453e-04, 1.94767577e-04,
       2.93044839e-04, 4.10977518e-04, 5.16402186e-04, 6.43242267e-04,
       7.57578760e-04, 8.62996560e-04, 9.95215494e-04, 1.07561890e-03,
       1.16495602e-03, 1.22229708e-03, 1.28857698e-03, 1.36542693e-03,
       1.40295830e-03, 1.55487098e-03, 1.68348849e-03, 1.76841393e-03,
       2.06136331e-03, 2.06850842e-03, 1.23431720e-03]), array32([-26.,
       -25.75, -25.5 , -25.25, -25.  , -24.75, -24.5 , -24.25,
       -24.  , -23.75, -23.5 , -23.25, -23.  , -22.75, -22.5 , -22.25,
       -22.  , -21.75, -21.5 , -21.25, -21.  , -20.75, -20.5 , -20.25,
       -20.  , -19.75, -19.5 , -19.25, -19.  , -18.75, -18.5 , -18.25,
       -18.  , -17.75, -17.5 , -17.25]))
    eco_lf = (0.5*(eco_lf[1][:-1]+eco_lf[1][1:]), eco_lf[0])
    lmax = int(np.max(absrmag))
    bv = np.arange(-26, lmax, 0.25)
    hist=plt.hist(absrmag, bins=bv, weights=1/volume+np.zeros_like(absrmag))
    plt.plot(eco_lf[0], eco_lf[1], color='k', marker='D', label='ECO (z=0)')
    plt.xlabel(r'$M_r$')
    plt.ylabel(r'Comoving Number Density [Mpc$^{-3}$]')
    plt.yscale('log')
    plt.xlim(-26,lmax)
    plt.legend(loc='best')
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.close()

    return firstpage, fig1, fig2, fig3

def multiplicity_function(grpids, return_by_galaxy=False):
    """
    Return counts for binning based on group ID numbers.

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
    uniqid, occurences = np.unique(grpids, return_counts=True)
    if return_by_galaxy:
        grpn_by_gal=np.zeros(len(grpids)).astype(int)
        for idv in grpids:
            sel = np.where(grpids==idv)
            grpn_by_gal[sel]=len(sel[0])
        return grpn_by_gal
    else:
        return occurences

def angular_separation(ra1,dec1,ra2,dec2):
    """
    Compute the angular separation bewteen two lists of galaxies using the Haversine formula.
    
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
    phi1 = ra1*np.pi/180.
    phi2 = ra2*np.pi/180.
    theta1 = np.pi/2. - dec1*np.pi/180.
    theta2 = np.pi/2. - dec2*np.pi/180.
    return 2*np.arcsin(np.sqrt(np.sin((theta2-theta1)/2.0)**2.0 + np.sin(theta1)*np.sin(theta2)*np.sin((phi2 - phi1)/2.0)**2.0))

def fwqm_relative_radius_laduma(ra, dec, zz):
    """
    Get the beam-relative radius of a galaxy or group to the LADUMA
    pointing center (listed) below. Expressed as a fraction of the
    full-width-quarter-max beamwidth. Returns negative values for
    objects outside the allowable redshift ranges.

    pointing center: 3:32:30.40  -28:07:57.0 hrs/deg
    in degrees: 53.126666, -27.8675
    """
    ra = array32(ra)
    dec = array32(dec)
    zz = array32(zz)
    laduma_pointing = (53.126666, -27.8675)
    angsep = angular_separation(ra, dec, laduma_pointing[0], laduma_pointing[1]) * 180/np.pi
    fwqm = LADUMA_FWQM(zz)
    output = angsep/fwqm
    output[(fwqm < 0)] = -999.
    return output 

def LADUMA_FWQM(zz):
    """
    Compute the full-width-quarter max beamwidth for LADUMA as f(z).
    If z is beyond the high-z spectral window, or otherwise outside
    LADUMA z ranges, -999 is returned.
   
    Frequency -> redshift for SPWs:
    880-933 MHz --> (0.52240707, 0.6140975)
    960-1159 MHz -> (0.22519413, 0.479589375)
    1304-1420 MHz >  (0.0002858, 0.0892683)
 
    Parameters
    ------------------
    zz : np.array
        Redshift for each point to evaluate FWQM beamwidth.
    Returns
    ------------------
    fwqm : np.array
        FWQM beamwidths in units of degrees.
    """
    zz = array32(zz)
    fwqm = np.zeros_like(zz) - 999.
    lowz = (zz < 0.0892683)
    midz = (zz > 0.22519413) & (zz < 0.47959)
    higz = (zz > 0.52240707) & (zz < 0.6140975)
    sqrt2_over_2=math.sqrt(2)/2 
    fwqm[lowz] = sqrt2_over_2 * 0.963 * (1+zz[lowz])**(1.016)
    fwqm[midz] = sqrt2_over_2 * 0.986 * (1+zz[midz])**(1.143)
    fwqm[higz] = sqrt2_over_2 * 0.999 * (1+zz[higz])**(1.077)
    return fwqm

def get_grprproj_e17(galra, galdec, galz, galgrpid, groupra, groupdec, groupz, cosmo):
    """
    Credit: Ella Castelloe for original python code 

    Compute the observational group projected radius from Eckert et al. (2017). Adapted from
    Katie's IDL code, which was used to calculate grprproj for FoF groups in RESOLVE.

    Now corrected for cosmology

    75% radius of all members
 
    Parameters
    --------------------
    galra : iterable
       RA of input galaxies in decimal degrees
    galdec : iterable
       Declination of input galaxies in decimal degrees
    galz : iterable
       Observed local group-corrected radial velocities of input galaxies (km/s)
    galgrpid : iterable
       Group ID numbers for input galaxies, length matches `galra`.
    
    Returns
    --------------------
    grprproj : np.array
        Group projected radii in Mpc/h, length matches `galra`.
 
    """
    galra = array32(galra)
    galdec = array32(galdec)
    galz = array32(galz)
    galgrpid = array32(galgrpid)
    groupra = array32(groupra)
    groupdec = array32(groupdec)
    groupz = array32(groupz)
    rproj75dist = np.zeros(len(galgrpid))
    uniqgrpid = np.unique(galgrpid)
    for uid in uniqgrpid:
        galsel = np.where(galgrpid==uid)
        if len(galsel[0]) > 2:
            ras = array32(galra[galsel])
            decs = array32(galdec[galsel])
            zs = array32(galz[galsel])
            grpn = len(galsel[0])
            grpra =  groupra[galsel][0]
            grpdec = groupdec[galsel][0]
            grpz =   groupz[galsel][0]
            theta = angular_separation(ras, decs, grpra, grpdec)
            d_theta = cosmo.comoving_transverse_distance(grpz).value
            rproj = theta * d_theta # Mpc
            sortorder = np.argsort(rproj)
            rprojsrt_y = rproj[sortorder]
            rprojval_x = np.arange(0 , grpn)/(grpn-1.) #array from 0 to 1, use for interpolation
            f = interp1d(rprojval_x, rprojsrt_y)
            rproj75val = f(0.75)
        else:
            rproj75val = 0.0
        rproj75dist[galsel]=rproj75val
    return rproj75dist

def group_z_demographics(galra, galdec, galz, galzflag, galgrpid, cosmo, zphotflag_value=0, zspecflag_value=1):
    galra = array32(galra)
    galdec = array32(galdec)
    galz = array32(galz)
    galzflag = array32(galzflag)
    galgrpid = array32(galgrpid)
    uniqgrpid = np.unique(galgrpid)
    speczfrac = np.zeros_like(galra)
    largestdzspec = np.zeros_like(galra)-99.
    largestonskydist = np.zeros_like(galra)-99.
    for uid in uniqgrpid:
        galsel = (galgrpid == uid)
        grp_specz_sel = (galsel & (galzflag==zspecflag_value))
        speczfrac[galsel] = np.sum(grp_specz_sel) / np.sum(galsel)

        Nspecz = np.sum(grp_specz_sel)
        if Nspecz>1:
            largestdzspec[galsel] = np.max(galz[grp_specz_sel]) - np.min(galz[grp_specz_sel])
        else:
            largestdzspec[galsel] = -99

        if np.sum(galsel)>1:
            angsep = angular_separation(galra[galsel], galdec[galsel], galra[galsel][:,None], galdec[galsel][:,None])
            onskydist = 0.5*angsep*(cosmo.comoving_transverse_distance(galz[galsel]).value+cosmo.comoving_transverse_distance(galz[galsel][:,None]).value)
            largestonskydist[galsel] = np.max(onskydist) # in Mpc
    return speczfrac, largestdzspec, largestonskydist

def get_int_mass(galmass, grpid):
    """
    Given a list of galaxy stellar or baryonic masses and group ID numbers,
    compute the group-integrated stellar or baryonic mass, galaxy-wise.
    
    Parameters
    ---------------
    galmass : iterable
        List of galaxy log(mass).
    grpid : iterable
        List of group ID numbers for every galaxy.


    Returns
    ---------------
    grpmstar : np.array
         Array containing group-integrated stellar masses for each galaxy; length matches `galmstar`.
    """
    galmass=np.asarray(galmass)
    grpid=np.asarray(grpid)
    grpmass = np.zeros(len(galmass))
    uniqgrpid=np.unique(grpid)
    for uid in uniqgrpid:
        sel=np.where(grpid==uid)
        totalmass = np.log10(np.sum(10**galmass[sel]))
        grpmass[sel]=totalmass
    return grpmass

def get_central_flag(galquantity, galgrpid):
    """
    Produce 1/0 flag indicating central/satellite for a galaxy
    group dataset.

    Parameters
    -------------------
    galquantity : np.array
       Quantity by which to select centrals. If (galquantity>0).all(), the central is taken as the maximum galquantity in each group.
       If (galquantity<0).all(), the quantity is assumed to be an abs. magnitude, and the central is the minimum quanity in each group.
    galgrpid : np.array
       Group ID number for each galaxy.

    Returns
    -------------------
    cflag : np.array
       1/0 flag indicating central/satellite status.
    """
    cflag = np.zeros(len(galquantity))
    uniqgrpid = np.unique(galgrpid)
    if ((galquantity>-1).all()):
       centralfn = np.max
    if ((galquantity<0).all()):
       centralfn = np.min
    for uid in uniqgrpid:
        galsel = np.where(galgrpid==uid)
        centralmass = centralfn(galquantity[galsel])
        centralsel = np.where((galgrpid==uid) & (galquantity==centralmass))
        cflag[centralsel]=1.
        satsel = np.where((galgrpid==uid) & (galquantity!=centralmass))
        cflag[satsel]=0.
    return cflag


def array32(x):
    return np.float32(x)

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
           'pfof_Pth' : 0.9, \
           'gd_vproj_fit_guess':[3e-5,4e-1], 'H0':hubble_const, 'Om0':omega_m, 'Ode0':omega_de,  'iterative_giant_only_groups':True,\
            'ncores' : None,
            })

    pg3ob=pg3(eco.radeg, eco.dedeg, eco.cz, eco.czerr, eco.absrmag,-19.5,fof_bperp=0.07,fof_blos=1.1,**gfargseco)
    pg3grp=pg3ob.find_groups()[0]
    eco.loc[:,'pg3grp'] = pg3grp
    print('elapsed time was ', time.time()-t1)
    eco.to_csv("../analysis/ECO_pg3_allspecz.csv")

    bins = np.arange(0.5,300.5,1)
    plt.figure()
    plt.hist(multiplicity_function(eco.g3grp_l.to_numpy(), return_by_galaxy=False), bins=bins, color='gray', histtype='stepfilled', label='G3 Groups', alpha=0.7)
    plt.hist(pg3ob.get_grpn(return_by_galaxy=False), bins=bins, color='blue', histtype='step', label='PG3 Groups', linewidth=3)
    plt.hist(pg3ob.get_dwarf_grpn_by_group(), bins=bins, color='blue', histtype='stepfilled', alpha=0.3, label='PG3 Dwarf-only Groups', linewidth=3)
    ecodwarfonly = eco.groupby('g3grp_l').filter(lambda g: (g.absrmag>-19.5).all())
    plt.hist(multiplicity_function(ecodwarfonly.g3grp_l.to_numpy(), return_by_galaxy=False), bins=bins, color='k', histtype='step', label='G3 Dwarf-Only Groups')
    plt.yscale('log')
    plt.xlabel(r"Group $N_{\rm galaxies}$")
    plt.xlim(0,50)
    plt.legend(loc='best')
    plt.show()
