import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit
from astropy.cosmology import LambdaCDM

from pg3tools import *
from kdpfof import kdPFOF
from prob_giantonlyic import prob_giantOnlyICRoutine
from prob_dwarfassoc import prob_dwarfAssocRoutine
from prob_dwarfonlyic import prob_dwarfOnlyICRoutine
#from giantonlyic import giantOnlyICRoutine
#from dwarfassoc import dwarfAssocRoutine
#from dwarfonlyic import dwarfOnlyICRoutine
#from g3misc import *

SPEED_OF_LIGHT = 3e5 # km s-1

class pg3groupfinder:
    """
    A Python class to perform galaxy group-finding using the Probabilistic
    G3 Group Finder [Hutchens et al. 2026, in prep.]

    Parameters
    -------------------
    radeg : array_like
        Right ascension of input galaxies in decimal degrees.
    dedeg : array_like
        Declination of input galaxies in decimal degrees.
    z : array_like
        Redshift of input galaxies.
    zerr : array_like
        Redshift uncertainties.
    absrmag : array_like
        Absolute magnitude for input galaxies.
    dwarfgiantdivide : float
        Value that will divide giants and dwarfs.
    H0 : float
        z=0 Hubble constant in units of (km/s)/Mpc, default 70.0. Return parameters will
        be consistent with this choice.
    Om0 : float
        Omega Matter at z=0, default 0.3
    Ode0 : float
        Omega DE at z=0, default 0.7.
    precision : callable
        Default Numpy precision to use throughout the code. Default is np.float32
        for memory efficiency. 
    """
    def __init__(self, radeg, dedeg, z, zerr, absrmag, dwarfgiantdivide, H0=70., Om0=0.3, Ode0=0.7, precision=np.float32):
        self.radeg = precision(radeg)
        self.dedeg = precision(dedeg)
        self.z = precision(z)
        self.zerr = precision(zerr)
        self.absrmag = precision(absrmag)
        self.dwarfgiantdivide = dwarfgiantdivide
        assert (not np.isnan(self.radeg).any()), "RA values must not contain NaNs."
        assert (not np.isnan(self.dedeg).any()), "DEC values must not contain NaNs."
        assert (not np.isnan(self.z).any()), "z values must not contain NaNs."
        assert (not np.isnan(self.absrmag).any()), "absrmag values must not contain NaNs."
        if (z>20).all():
            print("WARNING: all input z's are >20. Intepreting as cz (not z)...")
            self.z /= SPEED_OF_LIGHT
        self.g3grpid = np.zeros_like(radeg, dtype=np.int32)-99
        self.g3ssid = np.zeros_like(radeg, dtype=np.int32)-99
        self.H0 = H0
        self.Om0 = Om0
        self.Ode0 = Ode0
        self.cosmo = LambdaCDM(H0=H0,Om0=Om0,Ode0=Ode0)
        self.giantsel = (self.absrmag<=self.dwarfgiantdivide)
        self.precision = precision

    def giantOnlyPFOF(self, Pth, fof_bperp, fof_blos, fof_sep=None, volume=None, n_pts_per_sigma=5):
        """
        Construct initial giant-only groups using FoF.
        
        Parameters
        ---------------
        Pth : float
            PFoF threshold probability
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
        n_pts_per_sigma : int
            Number of points per standard deviation in p(z) integration.
        """
        self.pfof_Pth = Pth
        self.fof_bperp = fof_bperp
        self.fof_blos = fof_blos
        self.fof_sep = fof_sep
        self.volume = volume
        if self.fof_sep is not None:
            pass
        else:
            self.fof_sep = (self.volume/np.sum(self.giantsel))**(1/3.)

        self.lperp = self.fof_bperp * self.fof_sep
        self.llos = self.fof_blos * self.fof_sep
        self.giantfofid = kdPFOF(self.radeg[self.giantsel],self.dedeg[self.giantsel],self.z[self.giantsel],self.zerr[self.giantsel],\
            self.lperp,self.llos,self.pfof_Pth,self.cosmo,n_pts_per_sigma)
        self.g3grpid[self.giantsel] = self.giantfofid
        self.g3ssid[self.giantsel] = self.giantfofid
        return self.giantfofid

    def deriveGiantCalibrations(self, rproj_fit_guess=None, rproj_fit_params=None, rproj_fit_multiplier=None, vproj_fit_guess=None,\
                            vproj_fit_params = None, vproj_fit_multiplier = None, vproj_fit_offset = 0,  n_bootstraps=5000, n_pts_per_sigma=5):
        """
        Derive and/or set calibrations for giant-only merging and dwarf association,
        corresponding to Equations 4-5 in H23.

        If you want to derive the calibrations from scratch, you should leave rproj_fit_params
        and vproj_fit_params as None, and pass guesses for the best fitting values.

        If you want to set the calibrations based on prior values, instead leave rproj_fit_guess
        and vproj_fit_guess as None, and pass the known values to rproj_fit_params and vproj_fit_params.

        In either case, you must pass the values for the multipliers.

        Parameters
        ---------------------------------------------------------------
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
        n_bootstraps : int
            Number of bootstraps to use when calculating uncertainties on median values.
        n_pts_per_sigma : int
            Number of points per standard deviation in p(z) integration.
        """
        self.rproj_fit_guess = rproj_fit_guess
        self.rproj_fit_params = rproj_fit_params
        self.rproj_fit_multiplier = rproj_fit_multiplier
        self.vproj_fit_guess = vproj_fit_guess
        self.vproj_fit_params = vproj_fit_params
        self.vproj_fit_multiplier = vproj_fit_multiplier
        self.vproj_fit_offset = vproj_fit_offset

        if (rproj_fit_params is None) or (vproj_fit_params is None):
            self.giantgrpra, self.giantgrpdec, self.giantgrpz = prob_group_skycoords(self.radeg[self.giantsel], self.dedeg[self.giantsel],\
                 self.z[self.giantsel], self.zerr[self.giantsel], self.g3grpid[self.giantsel], n_pts_per_sigma)
            giantgrpcz = SPEED_OF_LIGHT*self.giantgrpz
            self.relvel = np.abs(giantgrpcz - SPEED_OF_LIGHT*self.z[self.giantsel])/(1+self.giantgrpz) # from https://academic.oup.com/mnras/article/442/2/1117/983284#30931438
            grp_ctd = self.cosmo.comoving_transverse_distance(self.giantgrpz).value
            gia_ctd = self.cosmo.comoving_transverse_distance(self.z[self.giantsel]).value
            self.relprojdist = (grp_ctd+gia_ctd)*np.sin(angular_separation(self.giantgrpra, self.giantgrpdec, self.radeg[self.giantsel],self.dedeg[self.giantsel])/2.0)
            self.giantgrpn = multiplicity_function(self.g3grpid[self.giantsel], return_by_galaxy=True)
            self.uniqgiantgrpn, uniqindex = np.unique(self.giantgrpn, return_index=True)
            keepcalsel = np.where(self.uniqgiantgrpn>1)
            self.median_relprojdist = np.array([np.median(self.relprojdist[np.where(self.giantgrpn==sz)]) for sz in self.uniqgiantgrpn[keepcalsel]])
            self.median_relvel = np.array([np.median(self.relvel[np.where(self.giantgrpn==sz)]) for sz in self.uniqgiantgrpn[keepcalsel]])
            self.rproj_median_error = np.std(np.array([smoothedbootstrap(self.relprojdist[np.where(self.giantgrpn==sz)], n_bootstraps, np.median,\
                 kwargs=dict({'axis':1 })) for sz in self.uniqgiantgrpn[keepcalsel]]), axis=1) 
            self.dvproj_median_error = np.std(np.array([smoothedbootstrap(self.relvel[np.where(self.giantgrpn==sz)], n_bootstraps, np.median,\
                 kwargs=dict({'axis':1})) for sz in self.uniqgiantgrpn[keepcalsel]]), axis=1) 
        if rproj_fit_params is None:    
            self.rproj_bestfit, rproj_bestfit_cov = curve_fit(giantmodel, self.uniqgiantgrpn[keepcalsel], self.median_relprojdist,\
                     sigma=self.rproj_median_error, p0=rproj_fit_guess)
            self.rproj_bestfit_err = np.sqrt(np.diag(rproj_bestfit_cov))
        else:
            self.rproj_bestfit = np.array(self.rproj_fit_params)
            self.rproj_bestfit_err = np.zeros(2)*1.
        if vproj_fit_params is None:
            self.vproj_bestfit, vproj_bestfit_cov  = curve_fit(giantmodel, self.uniqgiantgrpn[keepcalsel], self.median_relvel,\
                 sigma=self.dvproj_median_error, p0=vproj_fit_guess)
            self.vproj_bestfit_err = np.sqrt(np.diag(vproj_bestfit_cov))
        else:
            self.vproj_bestfit = np.array(self.vproj_fit_params)
            self.vproj_bestfit_err = np.zeros(2)*1.
        self.rproj_boundary = lambda Ngiants: self.rproj_fit_multiplier*giantmodel(Ngiants, *self.rproj_bestfit)
        self.vproj_boundary = lambda Ngiants: self.vproj_fit_multiplier*giantmodel(Ngiants, *self.vproj_bestfit) + self.vproj_fit_offset

    def plotGiantGroupBoundaries(self, show=False, savepath=None, rproj_xlim=20, rproj_ylim=1, vproj_xlim=20, vproj_ylim=1000, legend_alpha=0.5, dpi=300):
        """
        Plot the underlying data and boundaries following from `deriveGiantCalibrations.`
        Matches the form of Figure 5 in H23.

        Parameters
        -------------------
        show : bool, default False
            Display the plot in real time?
        savepath : str
            Path on disk to save the plot. If None, figure is not saved (default).

        """
        fig,axs=plt.subplots(ncols=2, figsize=(7,4))
        tx=np.linspace(1,max([rproj_xlim, vproj_xlim])+1)
        axs[0].scatter(self.giantgrpn, self.relprojdist, color='red', s=2, rasterized=True, label=r'Giant Galaxies ($R_{\rm proj,\, gal}$)')
        axs[0].errorbar(self.uniqgiantgrpn[1:], self.median_relprojdist, yerr=self.rproj_median_error, fmt='^', color='k', label=r'$R_{\rm proj}$')
        axs[0].plot(tx, giantmodel(tx, *self.rproj_bestfit), color='blue', label=r'$1R_{\rm proj}^{\rm fit}$')
        axs[0].plot(tx, self.rproj_boundary(tx), color='green', linestyle='dashed', label=str(self.rproj_fit_multiplier)+r'$R_{\rm proj}^{\rm fit}$')
        axs[1].scatter(self.giantgrpn, self.relvel, color='red', s=2, rasterized=True, label=r'Giant Galaxies ($\Delta v_{\rm proj,\, gal}$)')
        axs[1].errorbar(self.uniqgiantgrpn[1:], self.median_relvel, yerr=self.dvproj_median_error, fmt='^', color='k', label=r'$\Delta v_{\rm proj}$')
        axs[1].plot(tx, giantmodel(tx, *self.vproj_bestfit), color='blue', label=r'$1\Delta v_{\rm proj}^{\rm fit}$')
        axs[1].plot(tx, self.vproj_boundary(tx), color='green', linestyle='dashed', label=str(self.vproj_fit_multiplier)+r'$\Delta v_{\rm proj}^{\rm fit}$'+\
            f' + {self.vproj_fit_offset} km/s')
        axs[0].set_xlim(0, rproj_xlim)
        axs[1].set_xlim(0, vproj_xlim)
        axs[0].set_ylim(0, rproj_ylim)
        axs[1].set_ylim(0, vproj_ylim)
        
        axs[0].set_xlabel('Number of Giant Members')
        axs[0].set_ylabel('Projected Distance [Mpc]')
        axs[1].set_xlabel('Number of Giant Members')
        axs[1].set_ylabel('Relative Velocity [km/s]')
        axs[0].legend(loc='upper left', framealpha=legend_alpha, fontsize=9)
        axs[1].legend(loc='upper left', framealpha=legend_alpha, fontsize=9)
        plt.tight_layout()
        if savepath is not None:
            plt.savefig(savepath, dpi=dpi)
        if show:
            plt.show()
        else:
            plt.close()

    def giantOnlyMerging(self, Pth, n_pts_per_sigma=5):
        """
        Perform giant-only merging (Step 2 of H23) based
        on boundaries calibrated in deriveGiantBoundaries.

        Parameters
        ------------------
        Pth : float
            Threshold proability.
        n_pts_per_sigma : int
            Number of points per standard deviation in p(z) integration.
        """
        revisedgiantgrpid = prob_giantOnlyICRoutine(self.radeg[self.giantsel],self.dedeg[self.giantsel],self.z[self.giantsel],self.zerr[self.giantsel],\
                            self.giantfofid,self.rproj_boundary, self.vproj_boundary, Pth, self.cosmo, n_pts_per_sigma)
        self.g3grpid[self.giantsel] = revisedgiantgrpid

    def dwarfAssoc(self, Pth, n_pts_per_sigma=5):
        """
        Perform dwarf association (Step 3 of H23) based
        on boundaries calibrated in deriveGiantBoundaries.

        Parameters
        ------------------
        Pth : float
            Threshold proability.
        n_pts_per_sigma : int
            Number of points per standard deviation in p(z) integration.
        """
        print('Associating dwarfs to giant-only groups...')
        self.dwarfsel = ~self.giantsel
        dwarfassocid, self.dwarfassocflag = prob_dwarfAssocRoutine(self.radeg[self.dwarfsel], self.dedeg[self.dwarfsel], self.z[self.dwarfsel], \
                self.zerr[self.dwarfsel], self.radeg[self.giantsel], self.dedeg[self.giantsel], self.z[self.giantsel], self.zerr[self.giantsel],\
                self.g3grpid[self.giantsel], self.rproj_boundary, self.vproj_boundary, Pth, self.cosmo, n_pts_per_sigma)
        self.g3grpid[self.dwarfsel] = dwarfassocid
        print(f'Dwarf association complete.')

    def deriveDwarfBoundaries(self, gd_rproj_fit_guess = None, gd_rproj_fit_params = None, gd_rproj_fit_multiplier=None, gd_vproj_fit_guess=None,\
                              gd_vproj_fit_params = None, gd_vproj_fit_multiplier=None, gd_vproj_fit_offs = None, gd_fit_bins=None):
        """
        Derive the boundaries for dwarf-only iterative combination
        (Eqns. 7+8 of H23).

        If you want to derive the calibrations from scratch, you should leave gd_rproj_fit_params
        and gd_vproj_fit_params as None, and pass guesses for the best fitting values.

        If you want to set the calibrations based on prior values, instead leave gd_rproj_fit_guess
        and gd_vproj_fit_guess as None, and pass the known values to gd_rproj_fit_params and gd_vproj_fit_params.

        In either case, you must pass the values for the multipliers.

        Parameters
        -----------------------------------------
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
        """
        self.gd_rproj_fit_guess = gd_rproj_fit_guess
        self.gd_rproj_fit_params = gd_rproj_fit_params
        self.gd_rproj_fit_multiplier = gd_rproj_fit_multiplier
        self.gd_vproj_fit_guess = gd_vproj_fit_guess
        self.gd_vproj_fit_params = gd_vproj_fit_params
        self.gd_vproj_fit_multiplier = gd_vproj_fit_multiplier
        self.gd_vproj_fit_offs = gd_vproj_fit_offs
        self.gd_fit_bins = gd_fit_bins

        gdsel = np.full(len(self.z), True)
        gdsel[np.where(self.dwarfsel[np.where(~self.dwarfassocflag)])] = False
        if (self.gd_rproj_fit_params is None) or (self.gd_vproj_fit_params is None):
            gdgrpra, gdgrpdec, gdgrpz = prob_group_skycoords(self.radeg[gdsel], self.dedeg[gdsel], self.z[gdsel], self.zerr[gdsel], self.g3grpid[gdsel])
            self.gdgrpn = multiplicity_function(self.g3grpid[gdsel], return_by_galaxy=True)
            
            dMi = self.cosmo.comoving_transverse_distance(gdgrpz).value
            dMj = self.cosmo.comoving_transverse_distance(self.z[gdsel]).value
            alpha_ij = angular_separation(gdgrpra, gdgrpdec, self.radeg[gdsel], self.dedeg[gdsel])
            self.gd_rproj = 0.5 * (dMi + dMj) * alpha_ij
            self.gd_vpec = SPEED_OF_LIGHT * np.abs(gdgrpz - self.z[gdsel]) / (1 + gdgrpz)
            self.gd_Mint = get_int_mag(self.absrmag[gdsel], self.g3grpid[gdsel])

            self.gdbinsel = np.where(np.logical_and(self.gdgrpn>1, self.gd_Mint>-24))
            self.gdmedianrproj, self.gd_magbinc, _, _ = center_binned_stats(self.gd_Mint[self.gdbinsel], self.gd_rproj[self.gdbinsel], np.median, bins=self.gd_fit_bins)
            self.gdmedianrproj_err, _, _, _ = center_binned_stats(self.gd_Mint[self.gdbinsel], self.gd_rproj[self.gdbinsel], sigmarange, bins=self.gd_fit_bins)
            self.gdmedianrelvel, _, _, _ = center_binned_stats(self.gd_Mint[self.gdbinsel], self.gd_vpec[self.gdbinsel], np.median, bins=self.gd_fit_bins)
            self.gdmedianrelvel_err, _, _, _ = center_binned_stats(self.gd_Mint[self.gdbinsel], self.gd_vpec[self.gdbinsel], sigmarange, bins=self.gd_fit_bins)
            nansel = np.isnan(self.gdmedianrproj)
        if (self.gd_rproj_fit_params is None):
            self.gd_rproj_bestfit, gd_rproj_cov=curve_fit(decayexp, self.gd_magbinc[~nansel], self.gdmedianrproj[~nansel], p0=self.gd_rproj_fit_guess)
            self.gd_rproj_bestfit_err = np.sqrt(np.diag(gd_rproj_cov))
        else:
            self.gd_rproj_bestfit = np.array(self.gd_rproj_fit_params)
            self.gd_rproj_bestfit_err = np.zeros(len(self.gd_rproj_fit_params))
        if (gd_vproj_fit_params is None):
            self.gd_vproj_bestfit, gd_vproj_cov=curve_fit(decayexp, self.gd_magbinc[~nansel], self.gdmedianrelvel[~nansel], p0=self.gd_vproj_fit_guess)
            self.gd_vproj_bestfit_err = np.sqrt(np.diag(gd_vproj_cov))
        else:
            self.gd_vproj_bestfit = np.array(self.gd_vproj_fit_params)
            self.gd_vproj_bestfit_err = np.zeros(len(self.gd_vproj_fit_params))
        self.gd_rproj_boundary = lambda M: self.gd_rproj_fit_multiplier*decayexp(M, *self.gd_rproj_bestfit)
        self.gd_vproj_boundary = lambda M: self.gd_vproj_fit_multiplier*decayexp(M, *self.gd_vproj_bestfit) + self.gd_vproj_fit_offs

    def plotDwarfBoundaries(self, show=False, savepath=None, rproj_xlim=None, rproj_ylim=None, vproj_xlim=None, vproj_ylim=None, legend_alpha=0.5, dpi=300):
            """
            Plot the underlying data and boundaries following from `deriveDwarfCalibrations.`
            Matches the form of Figure 7 in H23.

            Parameters
            -------------------
            show : bool, default False
                Display the plot in real time?
            savepath : str
                Path on disk to save the plot. If None, figure is not saved (default).
            """
            Marr = np.linspace(-24,-17,1000)
            fig,axs=plt.subplots(ncols=2, figsize=(7,4))
            axs[0].scatter(self.gd_Mint[self.gdbinsel], self.gd_rproj[self.gdbinsel], s=1, alpha=0.5, color='gray', rasterized=True, label='Galaxies')
            axs[0].plot(Marr, self.gd_rproj_boundary(Marr)/self.gd_rproj_fit_multiplier, color='red', label=r'$R_{\rm proj,\, fit}^{\rm gi,\, dw}$')
            axs[0].plot(Marr, self.gd_rproj_boundary(Marr), color='blue', linestyle='dashed', \
                    label=str(self.gd_rproj_fit_multiplier)+r'$R_{\rm proj,\, fit}^{\rm gi,\, dw}$')
            axs[0].plot(self.gd_magbinc, self.gdmedianrproj, 'k^', label='Medians')
            axs[0].legend(loc='upper left', framealpha=legend_alpha)
            axs[0].set_xlabel(r"Integrated Giant+Dwarf $M_r$",fontsize=10)
            axs[0].set_ylabel(r"$R_{\rm proj}^{\rm gi,dw}$",fontsize=10)
            if rproj_xlim is not None: axs[0].set_xlim(rproj_xlim)
            if rproj_ylim is not None: axs[0].set_ylim(rproj_ylim)

            axs[1].scatter(self.gd_Mint[self.gdbinsel], self.gd_vpec[self.gdbinsel], s=1, alpha=0.5, color='gray', rasterized=True, label='Galaxies')
            axs[1].plot(Marr, (self.gd_vproj_boundary(Marr)-self.gd_vproj_fit_offs)/self.gd_vproj_fit_multiplier, color='red',\
                        label=r'$\Delta v_{\rm proj,\, fit}^{\rm gi,\, dw}$')
            axs[1].plot(Marr, self.gd_vproj_boundary(Marr), color='blue', linestyle='dashed', \
                   label=str(self.gd_vproj_fit_multiplier)+r'$\Delta v_{\rm proj,\, fit}^{\rm gi,\, dw}$'+f'+{self.gd_vproj_fit_offs} km/s')
            axs[1].plot(self.gd_magbinc, self.gdmedianrelvel, 'k^', label='Medians')
            axs[1].legend(loc='upper left', framealpha=legend_alpha)
            axs[1].set_xlabel(r"Integrated Giant+Dwarf $M_r$",fontsize=10)
            axs[1].set_ylabel(r"$\Delta v_{\rm proj}^{\rm gi,dw}$",fontsize=10)
            if vproj_xlim is not None: axs[1].set_xlim(vproj_xlim)
            if vproj_ylim is not None: axs[1].set_ylim(vproj_ylim)

            plt.tight_layout()
            if savepath is not None:
                plt.savefig(savepath, dpi=dpi)
            if show:
                plt.show()
            else:
                plt.close()

    def findDwarfOnlyGroups(self, Pth, n_pts_per_sigma=5):
        """
        From the remaining ungrouped dwarfs, construct dwarf-only groups.
        (Step 4 of H23.)

        Parameters
        ------------------
        Pth : float
            Threshold proability.
        n_pts_per_sigma : int
            Number of points per standard deviation in p(z) integration.
        """
        startID = np.max(self.g3grpid)+1
        grpn_after_assoc = multiplicity_function(self.g3grpid, True)
        self.dwarfonlysel = (grpn_after_assoc == 1) & self.dwarfsel
        dwarf_only_grpid = prob_dwarfOnlyICRoutine(self.radeg[self.dwarfonlysel], self.dedeg[self.dwarfonlysel], self.z[self.dwarfonlysel], \
                           self.zerr[self.dwarfonlysel], self.absrmag[self.dwarfonlysel], self.gd_rproj_boundary, self.gd_vproj_boundary, \
                           Pth, self.cosmo, startID, n_pts_per_sigma)
        self.g3grpid[self.dwarfonlysel] = dwarf_only_grpid

    def getCatalog(self, by='galaxy'):
        """
        Return a group catalog containing group IDs, group centers,
        integrated luminosities, and group N.

        Parameters
        ---------------------
        by : str
            If 'galaxy' (default), the group catalog info is returned on a galaxy-by-
            galaxy basis matching the order of the input galaxies.
            If 'group', it is returned group-by-group.
        
        Returns
        ---------------------
        catalog : numpy array
            Group catalog with columns [grpID, grpRA, grpDEC, grpz, grpN, grpAbsMag].
        """
        grpra, grpdec, grpz = prob_group_skycoords(self.radeg, self.dedeg, self.z, self.zerr, self.g3grpid)
        grpn = multiplicity_function(self.g3grpid, True)
        grpM = get_int_mag(self.absrmag, self.g3grpid)
        catalog = np.array([self.g3grpid, grpra, grpdec, grpz, grpn, grpM]).T
        if by == 'galaxy':
            return catalog
        elif by == 'group':
            _, uniqind = np.unique(self.g3grpid, return_index=True)
            return catalog[uniqind]
        else:
            raise ValueError(f"Parameter `by` must be 'galaxy' or 'group', not '{by}'.")
