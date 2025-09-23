import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import median_abs_deviation, binned_statistic
from scipy.optimize import curve_fit, OptimizeWarning
from astropy.stats import sigma_clip
from copy import deepcopy
import warnings

warnings.simplefilter("ignore",OptimizeWarning)

class corrector(object):
    def __init__(self, df : pd.DataFrame,
                 zphot_key : str,
                 zphot_err_key : str,
                 zspec_key : str,
                 apmag_key : str,
                 goodzflag_key : str = None,
                 imag_key : str = None,
                 imag_cut : float = None,
                 bic_diff_thresh : float = 5.0,
                 view_intermediate_plots : bool = False,
                 save_final_fig_path : str = None,
                 log_file_path: str = None,
                 force_linear_fit1: bool = False,
                 force_quad_fit2: bool = False,
                 force_linear_fit2: bool = False,
                ) -> None:
        """
        `corrector` - class object for correcting photometric z
        systematics.

        Parameters
        -----------------------
        df : pd.DataFrame
            Input pandas dataframe containing redshift/flag/mag data.
        zphot_key : str
            Key in df indicating zphot.
        zphot_err_key : str
            Key in df indicating zphot_err.
        zspec_key : str
            Key in df indicating zspec.
        apmag_key : str
            Key in df indicating apparent magnitude column.
            Used to calibrate against.
        goodzflag_key : str
            Column in df indicating redshift quality (1=keep, 0=ignore)
            For photo-z's based on EAZY, one can use (Qz<1) to indicate
            high-quality redshifts.
        imag_key : str
            Key in dataframe for i-mag, used in selection for XMM-SERVS.
            Default None (not used).
        imag_cut : str
            Cut on i-mag set by imag_key, used in selection for XMM-SERVS.
            Default None (not used).
        bic_diff_thresh : float
            Threshold for deciding between competing models using BIC, default 5.
        view_intermediate_plots : bool
            If True (default is False), intermediate plots will be displayed
            using matplotlib (intermediate meaning iteration-by-iteration).
        save_final_fig_path : str
            If not None (default), the final figure will be saved to disk
            at the specified path.
        log_file_path : str 
            Path at which to save an ASCII log file containing information 
            relveant to fitting. If None (default), a log is not created.
        """
        self.converged = False
        self.df = df
        self.mag = self.df[apmag_key].to_numpy()
        self.zspec = self.df[zspec_key].to_numpy()
        self.zphot = self.df[zphot_key].to_numpy()
        self.zphoterr = self.df[zphot_err_key].to_numpy()
        self.imag_cut = imag_cut
        if imag_key is not None:
            self.imag = self.df[imag_key].to_numpy()
        if goodzflag_key is not None:
            self.goodz = self.df[goodzflag_key].to_numpy(dtype=int)
        else:
            self.goodz = np.ones_like(self.zphot)
        self.bic_diff_thresh = bic_diff_thresh
        self.save_final_fig_path = save_final_fig_path
        self.view_intermediate_plots = view_intermediate_plots
        if log_file_path is not None:
            self.logFile = open(log_file_path,'w+')
        else:
            self.logFile = None
        self.force_linear_fit1 = force_linear_fit1
        self.force_quad_fit2 = force_quad_fit2
        self.force_linear_fit2 = force_linear_fit2
        
    def run(self) -> (np.array, np.array, np.array):
        self.zphotcorr = deepcopy(self.zphot)
        self.zphoterrcorr = deepcopy(self.zphoterr)
        counter = 0
        while not self.converged:
            _ = self.getCalibrationSample()
            self.getCorrections()
            self.converged = self.checkConverged()
            self.updateLog(counter)
            counter+=1
            if self.view_intermediate_plots: self.makePlots()

        print(f'Done after {counter} iteration(s).')
        if self.save_final_fig_path is not None:
            self.makePlots(self.save_final_fig_path)
        if self.logFile is not None:
            self.logFile.close()
        return self.zphotcorr, self.zphoterrcorr, self.corrected_flag

    def getCalibrationSample(self, nsigma=4):
        dz_over_ee = (self.zphotcorr - self.zspec)/self.zphoterrcorr
        not_outlier = (sigma_clip(dz_over_ee, nsigma).mask == False)
        if self.imag_cut is not None:
            imag_sel = (self.imag < self.imag_cut)
        else:
            imag_sel = np.full(len(self.mag),True)
        self.selection = (
            ~np.isnan(self.zphot) & \
            ~np.isnan(self.zphoterr) & \
            ~np.isnan(self.mag) & \
            ~np.isnan(self.zspec) & \
            (self.zspec >= 0) & \
            (self.zphot >= 0) & \
            (self.mag > 0)  & \
            (self.goodz == 1) &  \
            not_outlier & \
            imag_sel \
        )
        self.mag_cal = self.mag[self.selection]
        self.zspec_cal = self.zspec[self.selection]
        self.zphot_cal = self.zphotcorr[self.selection]
        self.zphoterr_cal = self.zphoterrcorr[self.selection]
        perc_selected=np.sum(self.selection)/len(dz_over_ee)*100
        if perc_selected<5:
            print(f"WARNING: only {perc_selected:0.2f}% of the sample was selected for calibration.")
        return self.selection

    def getCorrections(self):
        dz = self.zphot_cal - self.zspec_cal
        etab = self.zphoterr_cal
        mag = self.mag_cal
        self.magbins = np.linspace(np.percentile(mag,5), np.percentile(mag,95), 10)
        magbinc, mediandz, bedg, _ = bstat(mag, dz, 'median', bins=self.magbins)
        _, mediandzerr, _, _ = bstat(mag, dz, median_bootstrap, bins=bedg)
        model1,model1p,model1e = self.fit_dz_vs_mr(magbinc, mediandz, mediandzerr, self.bic_diff_thresh, self.force_linear_fit1)
        dzcorr = model1(mag)
        zpcorr = self.zphot_cal - dzcorr
        dzcorr_over_etab = (zpcorr - self.zspec_cal)/etab
        magbinc, nmad_dzcorr_over_etab, bedg, _ = bstat(mag, dzcorr_over_etab, nmad, bins=self.magbins)
        _,nmad_dzcorr_over_etab_err,_,_ = bstat(mag, dzcorr_over_etab, nmad_bootstrap, bins=self.magbins)
        model2,model2p,model2e = self.fit_dz_over_etab(magbinc,nmad_dzcorr_over_etab,nmad_dzcorr_over_etab_err,\
            self.bic_diff_thresh, self.force_quad_fit2, self.force_linear_fit2)
        etabcorr = etab * model2(mag)

        self.corrected_flag = np.ones_like(self.zphot)
        self.zphotcorr  -= model1(self.mag)
        sel = (self.zphot<=0) | np.isnan(self.zphot) | (self.mag<=0) | np.isnan(self.mag) | (self.zphotcorr<0)
        self.zphotcorr[sel]=self.zphot[sel]
        self.corrected_flag[sel] = 0

        print(f'% extrapolating: {100*np.sum(((self.mag<np.min(self.mag_cal)) | (self.mag>np.max(self.mag_cal))))/len(self.mag)}')

        self.zphoterrcorr *= model2(self.mag)
        sel = (self.zphoterr<=0) | np.isnan(self.zphoterr) | (self.mag<=0) | np.isnan(self.mag) | (self.zphotcorr<0)
        self.zphoterrcorr[sel]=self.zphoterr[sel]

        self.model1 = model1
        self.model1p = model1p
        self.model1e = model1e
        self.model2 = model2
        self.model2p = model2p
        self.model2e = model2e
        self.magbinc = magbinc
        self.mediandz = mediandz
        self.mediandzerr = mediandzerr
        self.nmad_dzcorr_over_etab = nmad_dzcorr_over_etab
        self.nmad_dzcorr_over_etab_err = nmad_dzcorr_over_etab_err
        
    def checkConverged(self, threshold=0.1):
        dzcorr = (self.zphotcorr - self.zspec)[self.selection]
        etabcorr = self.zphoterrcorr[self.selection]
        loc = np.median(dzcorr)
        scale = nmad(dzcorr/etabcorr)
        scale_dev = np.abs(scale-1)
        converged = (np.abs(loc)<threshold) & (scale_dev<threshold)
        return converged
   
    @staticmethod
    def fit_dz_vs_mr(magbinc,mediandz,mediandzerr,bic_diff_thresh,force_linear_fit1):
        linearmodel, linearparams, linearparamerr = fit_poly(magbinc, mediandz, 1, mediandzerr)
        quadmodel, quadparams, quadparamerr = fit_poly(magbinc, mediandz, 2, mediandzerr)
        pwisemodel, pwiseparams, pwiseparamerr = fit_piecewise_linear(magbinc, mediandz, mediandzerr)
        sigmasquared = (mediandzerr*mediandzerr)
        chi2_linear = np.sum((mediandz - linearmodel(magbinc))**2./sigmasquared)
        chi2_quad = np.sum((mediandz - quadmodel(magbinc))**2./sigmasquared)
        chi2_pwise = np.sum((mediandz - pwisemodel(magbinc))**2./sigmasquared)
        
        n_sample = len(magbinc)
        bic_linear = compute_bic(chi2_linear, len(linearparams), n_sample)
        bic_quad = compute_bic(chi2_quad, len(quadparams), n_sample)
        bic_pwise = compute_bic(chi2_pwise, len(pwiseparams), n_sample)
        if force_linear_fit1:
            return linearmodel, linearparams, linearparamerr
        else:
            if (bic_linear-bic_quad)>=bic_diff_thresh and (bic_linear-bic_pwise)<bic_diff_thresh:
                return quadmodel, quadparams, quadparamerr
            elif (bic_linear-bic_quad)<bic_diff_thresh and (bic_linear-bic_pwise)>=bic_diff_thresh:
                return pwisemodel, pwiseparams, pwiseparamerr
            else:
                return linearmodel, linearparams, linearparamerr

    @staticmethod
    def fit_dz_over_etab(magbinc, nmad_ratio, nmad_ratioerr,bic_diff_thresh,force_quad_fit2,force_linear_fit2):
        linearmodel, linearparams, linearparamerr = fit_poly(magbinc, nmad_ratio, 1, nmad_ratioerr)
        quadmodel, quadparams, quadparamerr = fit_poly(magbinc, nmad_ratio, 2, nmad_ratioerr)
        pwisemodel, pwiseparams, pwiseparamerr = fit_piecewise_linear(magbinc, nmad_ratio, nmad_ratioerr)
        sigmasquared = (nmad_ratioerr*nmad_ratioerr)
        chi2_linear = np.sum((nmad_ratio - linearmodel(magbinc))**2./sigmasquared)
        chi2_quad = np.sum((nmad_ratio - quadmodel(magbinc))**2./sigmasquared)
        chi2_pwise = np.sum((nmad_ratio - pwisemodel(magbinc))**2./sigmasquared)

        n_sample = len(magbinc)
        bic_linear = compute_bic(chi2_linear, len(linearparams), n_sample)
        bic_quad = compute_bic(chi2_quad, len(quadparams), n_sample)
        bic_pwise = compute_bic(chi2_pwise, len(pwiseparams), n_sample)
        if force_quad_fit2 and (not force_linear_fit2):
            quadratic = lambda x,a,b,c: (a*x*x + b*x + c)
            quadparams, pcov = curve_fit(quadratic, magbinc, nmad_ratio,\
                sigma=nmad_ratioerr, absolute_sigma=True)
            quadparamerr = np.sqrt(np.diagonal(pcov))
            quadmodel = lambda x: quadratic(x,*quadparams) 
            return quadmodel, quadparams, quadparamerr
        elif force_linear_fit2 and (not force_quad_fit2):
            return linearmodel, linearparams, linearparamerr
        elif force_linear_fit2 and force_quad_fit2:
            raise ValueError("force_lienar_fit2 and force_quad_fit2 cannot be both be True")   
        else:
            if (bic_linear-bic_quad)>=bic_diff_thresh and (bic_linear-bic_pwise)<bic_diff_thresh:
                return quadmodel, quadparams, quadparamerr
            elif (bic_linear-bic_quad)<bic_diff_thresh and (bic_linear-bic_pwise)>=bic_diff_thresh:
                return pwisemodel, pwiseparams, pwiseparamerr
            else:
                return linearmodel, linearparams, linearparamerr


    def makePlots(self, savepath=None):
        histkwargs = dict(bins=np.arange(-5,5,0.1),density=True,alpha=0.6)
        errkwargs = dict(fmt='o', color='k', capsize=3)
        mx = np.linspace(10,35,1000)
        mrxlim = (int(np.min(self.mag[self.selection]))-2, int(np.max(self.mag[self.selection]))+2)
        fig,axs = plt.subplots(ncols=2, nrows=2)
        zphotmin, zphotmax = np.min(self.zphot_cal), np.max(self.zphot_cal)
        fig.suptitle(f'{zphotmin:0.3f} ' + r'$ < z_{\rm phot} < $' + f'{zphotmax:0.3f}') 
        dz_over_etab = (self.zphot - self.zspec)/self.zphoterr
        dz_over_etab = dz_over_etab[self.selection]
        axs[0][0].hist(dz_over_etab, **histkwargs)
        axs[0][0].set_xlim(-4,4)
        txt = f'Med={np.median(dz_over_etab):0.3f}\nNMAD={nmad(dz_over_etab):0.3f}'
        axs[0][0].annotate(text=txt, xy=(0.03,0.8), xycoords='axes fraction')
        axs[0][0].set_xlabel(r'$(z_{\rm phot}-z_{\rm spec})/\sigma_{\rm phot}$')

        dz = (self.zphot-self.zspec)[self.selection]
        axs[0][1].scatter(self.mag[self.selection], dz, alpha=0.6, s=2)
        axs[0][1].plot(mx, self.model1(mx), color='red')
        axs[0][1].errorbar(self.magbinc, self.mediandz, self.mediandzerr, **errkwargs)
        #axs[0][1].axvline(np.max(self.mag[self.mag>0]),color='k',linestyle='dashed')
        #axs[0][1].axvline(np.min(self.mag[self.mag>0]),color='k',linestyle='dashed')
        axs[0][1].set_xlim(*mrxlim)
        axs[0][1].set_ylim(-0.5,0.5)
        axs[0][1].set_xlabel(r'$m_r$')
        axs[0][1].set_ylabel(r'Median($z_{\rm phot}-z_{\rm spec}$)')
        axs[0][1].invert_xaxis()

        axs[1][0].plot(mx, self.model2(mx), color='red')
        axs[1][0].errorbar(self.magbinc, self.nmad_dzcorr_over_etab, self.nmad_dzcorr_over_etab_err, **errkwargs)
        axs[1][0].set_xlim(*mrxlim)
        axs[1][0].set_ylim(0,4)
        axs[1][0].set_ylim(0,1.2*np.max(self.nmad_dzcorr_over_etab))
        axs[1][0].set_xlabel(r'$m_r$')
        axs[1][0].set_ylabel('Scale Factor')
        axs[1][0].invert_xaxis()

        dzcorr_over_etabcorr = (self.zphotcorr - self.zspec)/self.zphoterrcorr
        dzcorr_over_etabcorr = dzcorr_over_etabcorr[self.selection] 
        txt = f'Med={np.median(dzcorr_over_etabcorr):0.3f}\nNMAD={nmad(dzcorr_over_etabcorr):0.3f}'
        axs[1][1].annotate(text=txt, xy=(0.03,0.8), xycoords='axes fraction')
        axs[1][1].hist(dzcorr_over_etabcorr, **histkwargs)
        axs[1][1].set_xlim(-4,4)
        axs[1][1].set_xlabel(r'$(z_{\rm phot,corr}-z_{\rm spec})/\sigma_{\rm phot,corr}$')
        plt.tight_layout()
        if savepath is not None: plt.savefig(savepath, dpi=300)
        plt.show()

    def updateLog(self, counter):
        if self.logFile is None:
            return None
        else:
            self.logFile.write(f'Iteration {counter}.\n')
            self.logFile.write(f'N(calib) = {self.selection.sum()}\n')
            self.logFile.write(f'model1 param/err = {self.model1p} {self.model1e}\n')
            self.logFile.write(f'model2 param/err = {self.model2p} {self.model2e}\n')
            self.logFile.write(f'------------------------------------------------\n')

# ------------------------------------------------------- #
# ------------------------------------------------------- #
#               supporting functions                      #
# ------------------------------------------------------- #
# ------------------------------------------------------- #
def nmad(*args,**kwargs):
    return 1.4826*median_abs_deviation(*args,**kwargs)

def bstat(*args, **kwargs):
    stat, binedges, binnum = binned_statistic(*args,**kwargs)
    binc = 0.5*(binedges[1:]+binedges[:-1])
    return binc, stat, binedges, binnum

def median_bootstrap(x):
    n_resamp = 5000
    d = np.random.choice(x, size=(len(x), n_resamp), replace=True)
    medians = np.median(d,axis=0)
    assert len(medians)==n_resamp
    return np.std(medians)

def nmad_bootstrap(x):
    n_resamp = 5000
    d = np.random.choice(x, size=(len(x), n_resamp), replace=True)
    nmadvals = nmad(d,axis=0)
    assert len(nmadvals)==n_resamp
    return np.std(nmadvals)

def compute_bic(chi2, n_params, n_sample):
    return n_params*np.log(n_sample) + chi2

def fit_poly(xx, yy, degree, yyerr=None):
    if yyerr is not None:
        ww = 1/yyerr
    else:
        ww = None
    params, cov = np.polyfit(xx,yy,deg=degree,w=ww,cov=True)
    model = lambda x: np.polyval(params, x)
    perr = np.sqrt(np.diagonal(cov))
    return model, params, perr

def fit_piecewise_linear(xx, yy, yyerr=None, guess=None):
    popt, pcov = curve_fit(piecewise_linear_function, xx, yy, p0=guess, \
                 sigma=yyerr, absolute_sigma=(False if (yyerr is None) else True))
    perr = np.sqrt(np.diagonal(pcov))
    model = lambda x: piecewise_linear_function(x, *popt)
    return model, popt, perr

def piecewise_linear_function(xx, xx0, s1, i1, s2, i2):
    """
    xx0 is the break
    """
    output = np.zeros_like(xx)
    sel1 = (xx <= xx0)
    sel2 = ~sel1
    output[sel1] = s1*xx[sel1] + i1
    output[sel2] = s2*xx[sel2] + i2
    return output
