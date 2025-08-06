import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,'../codes/')
from photzcorrection import corrector
from survey_volume import solid_angle, integrate_volume
from copy import deepcopy
from astropy.cosmology import LambdaCDM
cosmo = LambdaCDM(70,0.3,0.7)

direc = dict(
    redshift_catalog = './xmmservs_w_absrmag_specz.csv',
    subvolpath = '/srv/one/zhutchen/paper3/data/subvolumes/',
)

def absMag(z, m):
    return m+5-5*np.log10(cosmo.luminosity_distance(z).to('pc').value)

if __name__=='__main__':
    # ----------------------------------------------------------------- #
    # split redshift catalog into bins
    xmms = pd.read_csv(direc["redshift_catalog"])

    rband_voice_depth = 26
    iband_zp_depth = 24
    rband_zp_depth = np.median(xmms.mag_R_VOICE[(xmms.mag_I_VOICE>iband_zp_depth-0.1)&(xmms.mag_I_VOICE<iband_zp_depth+0.1)])
    xmms = xmms[(xmms.mag_R_VOICE<rband_zp_depth) & (xmms.mag_R_VOICE>0) & (xmms.absmag_R_VOICE>-99) & (xmms.mag_I_VOICE<iband_zp_depth) & \
                (xmms.redchi2_gal < 2)]

    fitting_bins = np.arange(0,1.4,0.15)
    xmms.loc[:,'zphotcorr_bin'] = np.digitize(xmms.zphot.to_numpy(), bins=fitting_bins)

    # ----------------------------------------------------------------- #
    # get corrected zphot data
    fit1params = []
    fit2params = []
    groups = xmms.groupby('zphotcorr_bin')
    for _, gg in groups:
        if (gg.zphotcorr_bin==0).all() or (gg.zphotcorr_bin==len(fitting_bins)).all():
            pass
        else:
            print(f'Working on bin {gg.iloc[0].zphotcorr_bin}.')
            figpath = './zp_correction_plots_logs/zphot_corr_{zmin:0.2f}_to_{zmax:0.2f}.png'.format(zmin=np.min(gg.zphot), zmax=np.max(gg.zphot))
            logpath = figpath.replace('png','txt')
            ct = corrector(gg, 'zphot', 'zphoterr', 'zspecbest', 'mag_R_VOICE', 'goodzflag', imag_key='mag_I_VOICE', imag_cut=24, \
                    view_intermediate_plots=False, save_final_fig_path=figpath, log_file_path=logpath, force_linear_fit1=True, force_quad_fit2=True)
            _ = ct.run()
            fit1params.append(ct.model1p)
            fit2params.append(ct.model2p)# if len(ct.model2p)>2 else np.array([0]+list(ct.model2p)))

    zbin = 0.5*(fitting_bins[1:]+fitting_bins[:-1])
    fit1params = np.array(fit1params)
    fit2params = np.array(fit2params)
    
    fig,axs=plt.subplot_mosaic("""
    ab.
    cde
    """)
    axs['a'].plot(zbin, fit1params[:,0], 'bo-') 
    axs['b'].plot(zbin, fit1params[:,1], 'bo-') 
    axs['c'].plot(zbin, fit2params[:,0], 'bo-') 
    axs['d'].plot(zbin, fit2params[:,1], 'bo-') 
    axs['e'].plot(zbin, fit2params[:,2], 'bo-') 
    plt.show()

    exit()
    xmms = pd.concat(processed, axis=0)
    assert len(xmms)==np.sum(counts)

    zbest = deepcopy(xmms.zphotcorr.to_numpy())
    zbesterr = deepcopy(xmms.zphoterrcorr.to_numpy())
    zbestflag = np.zeros_like(zbest)
    spec_avail = (xmms.zspecbest > 0).to_numpy()
    zbest[spec_avail] = xmms.zspecbest.to_numpy()[spec_avail]
    zbesterr[spec_avail] = xmms.zspecbesterr.to_numpy()[spec_avail]
    zbestflag[spec_avail] = 1
    xmms.loc[:,'bestoverallz'] = zbest
    xmms.loc[:,'bestoverallzerr'] = zbesterr
    xmms.loc[:,'bestoverallzflag'] = zbestflag #1=photz 2=specz

    tmp=(xmms[(xmms.zphot>0.6) & (xmms.zphot<0.8) & (xmms.absmag_R_VOICE>-15) & (xmms.absmag_R_VOICE>-99) & (xmms.mag_R_VOICE<rband_zp_depth)])
    tmp = tmp[['bestoverallz','zspecbest','zphot','zspec','mag_R_VOICE','absmag_R_VOICE','Tractor_ID','ngoodband','redchi2_gal']]
    print(tmp)

    selection = (xmms.bestoverallz>0) & (xmms.bestoverallz < np.max(fitting_bins)) & (xmms.mag_R_VOICE<rband_zp_depth) & (xmms.mag_R_VOICE>0)
    magcols = [kk for kk in xmms.columns if (kk.startswith('mag') or kk.startswith('absmag'))]
    zcols = [kk for kk in xmms.columns if (kk.endswith('zspec_cat') or kk.startswith('z') or kk.startswith('bestov'))]
    cols_to_keep = ['RA','DEC','Tractor_ID']+magcols+zcols
    xmms = xmms.loc[selection, cols_to_keep]

    plt.figure()
    sc=plt.scatter(xmms.bestoverallz, xmms.absmag_R_VOICE, c=xmms.mag_R_VOICE, vmin=18, vmax=24.4, alpha=0.6, s=1)
    z_arr = np.linspace(0,1.4,1000)
    plt.plot(z_arr, absMag(z_arr, rband_voice_depth), label=r'VOICE Photometric Depth ($m_r = 26$)', color='k')
    plt.plot(z_arr, absMag(z_arr, rband_zp_depth), label=r'VOICE $z_{\rm phot}$  Depth ($m_r = $' + f'{rband_zp_depth:0.1f}' + r')',\
             linestyle='dashed', color='k')
    plt.colorbar(sc, label=r'$m_r$')
    plt.ylim(-5,-25)
    plt.xlim(0,np.max(fitting_bins))
    plt.xlabel('Best Overall $z$')
    plt.ylabel(r'$M_r$')
    plt.legend(loc='best')
    plt.show()


    # ----------------------------------------------------------------- #
    zwin=0.2
    subvolranges = np.array([np.arange(0,1,zwin), np.arange(0+zwin,1+zwin,zwin)]).T 
    subvolranges_offset = subvolranges+zwin/2
    subvolranges = np.vstack([subvolranges, subvolranges_offset])
    print(subvolranges)

    log = open(direc['subvolpath']+"subvolume_metadata.csv",'w+')
    log.write('File\tN\tzmin\tzmax\tSA_str\tVolMpc3\tMrlim\n')
    for (zmin, zmax) in subvolranges:
        fname = "subvolume_{:0.1f}_to_{:0.1f}.csv".format(zmin,zmax)
        subvolume = xmms[(xmms.bestoverallz>zmin) & (xmms.bestoverallz<=zmax) & ~pd.isna(xmms.mag_R_VOICE)]
        ngal = len(subvolume)
        SA = solid_angle(subvolume.RA, subvolume.DEC, bins=1000) / 3282.8 # converted from deg2 to steradians
        vol = integrate_volume(np.linspace(zmin,zmax,10000), SA, 70., 0.3, 0.7)
        Mrlim = absMag(zmax, rband_zp_depth)
        log.write(f"{fname}\t{int(ngal)}\t{zmin:0.1f}\t{zmax:0.1f}\t{SA:0.6E}\t{vol:0.6E}\t{Mrlim:0.2f}\n")
        subvolume.to_csv(direc["subvolpath"]+fname,index=False)
    log.close()
