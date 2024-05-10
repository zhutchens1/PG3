import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,'../codes/')
import prob_g3groupfinder as pg3
import foftools as fof
import pickle
from scipy.integrate import simpson as simps
######
eco = pd.read_csv("/srv/one/zhutchen/g3groupfinder/resolve_and_eco/ECOdata_G3catalog_luminosity.csv").set_index('name')
eco = eco[(eco.absrmag<=-19.5)]
eco.loc[:,'czerr'] = eco.cz*0 + 50

######
# PFOF
bperp = 0.07
blos = 1.1
s = (1.91936e5/len(eco))**(1/3.)
#pfofid = pg3.pfof_comoving(eco.radeg, eco.dedeg, eco.cz, eco.czerr, bperp*s, blos*s, 0.9)
#pickle.dump(pfofid, open('pfofid.pkl','wb'))
pfofid = pickle.load(open('pfofid.pkl','rb'))
eco.loc[:,'pfofid'] = pfofid

###############
# Group Centers

grpra, grpdec, grpz, zpdfs = pg3.prob_group_skycoords(eco.radeg.to_numpy(), eco.dedeg.to_numpy(), eco.cz.to_numpy()/3e5, eco.czerr.to_numpy()/3e5, eco.pfofid.to_numpy(), True)
grpn = fof.multiplicity_function(eco.pfofid, return_by_galaxy=True)
eco.loc[:,'pfofgrpn'] = grpn
eco.loc[:,'pfofgrpz'] = grpz

groups=eco.groupby('pfofid').first()

############################
# Shift zpdfs to rest frame


#vpecmesh = 3e5*(zpdfs['zmesh'] - grpz[:,np.newaxis])/(1+grpz[:,np.newaxis])

#plt.figure()
#plt.plot(vpecmesh[13], zpdfs['pdf'][13] / simps(zpdfs['pdf'][13],zpdfs['zmesh']))
#plt.xlabel(r"$v_{\rm pec}$ [km/sec]")
#plt.ylabel("Normalized PDF")
#plt.show()

newvpecmesh = np.arange(-3000,3000,1)
for nn in np.unique(groups.pfofgrpn):
    if nn==10:
        grpids_needed = groups[groups.pfofgrpn==nn].index.to_numpy() # note index is pfofid
        grpz_needed = groups[groups.pfofgrpn==nn].pfofgrpz.to_numpy()
        _, idx, _ = np.intersect1d(zpdfs['grpid'], grpids_needed, return_indices=True)
        pdfs_at_fixed_nn = np.array([zpdfs['pdf'][np.where(zpdfs['grpid']==gg)][0] for gg in grpids_needed])
        print(pdfs_at_fixed_nn.shape)
        grpz_for_pdfs = grpz_needed#np.array([zpdfs['pdf'][np.where(zpdfs['grpid']==gg)] for gg in grpids_needed])
        #grpz_for_pdfs = np.array([grpz_needed[np.where(grpids_needed==gg)][0] for gg in zpdfs['grpid'][idx]])
        vpecmesh = 3e5*(zpdfs['zmesh'] - grpz_for_pdfs[:,np.newaxis])/(1+grpz_for_pdfs[:,np.newaxis])
       
        #plt.figure()
        #plt.plot(zpdfs['zmesh'], pdfs_at_fixed_nn[12])
        #plt.show()

        #plt.figure()
        #print(grpz_for_pdfs[12])
        #print(zpdfs['grpid'][idx][12])
        #print(groups[groups.index==73][['pfofgrpn','pfofgrpz']])
        #plt.plot(vpecmesh[12], pdfs_at_fixed_nn[12])
        #plt.show()
        combinedvpecdist = np.array([np.interp(newvpecmesh, vpecmesh[ii], pdfs_at_fixed_nn[ii], left=0, right=0) for ii in range(0,len(grpids_needed))])

        combinedvpecdist = np.hsplit(combinedvpecdist, 2)
        #combinedvpecdist = np.vstack([np.abs(combinedvpecdist[0]),np.abs(combinedvpecdist[1])])
        combinedvpecdist = np.vstack([np.flip(combinedvpecdist[0]), combinedvpecdist[1]])
        combinedvpecdist = np.sum(combinedvpecdist,axis=0)
        print(combinedvpecdist.shape)

        plt.figure()
        finalvpec = np.hsplit(newvpecmesh,2)[1]
        plt.plot(finalvpec, combinedvpecdist)
        plt.axvline(pg3.get_median_eCDF(finalvpec,combinedvpecdist),color='k')
        plt.show()



"""
        break

        sel = np.where(grpn==nn)
        pdfs_at_fixed_nn = zpdfs['pdf'][sel]
        #pdfs_at_fixed_nn = pdfs_at_fixed_nn / simps(pdfs_at_fixed_nn, zpdfs['zmesh'])[:,np.newaxis]
        print(pdfs_at_fixed_nn.shape)
        print(simps(pdfs_at_fixed_nn, zpdfs['zmesh']).shape)
        print(zpdfs['zmesh'])
       
        plt.figure()
        plt.plot(zpdfs['zmesh'], pdfs_at_fixed_nn[-1])
        print((pdfs_at_fixed_nn[-1]==0).all())
        plt.show() 
    
        print(zpdfs['grpid'][sel][-1])


        sel = np.where(zpdfs['grpid']==2485.)
        print(sel)
        print(zpdfs['pdf'][sel])
        plt.figure()
        plt.plot(zpdfs['zmesh'], zpdfs['pdf'][sel])
        plt.show()

        break    
        combinedvpecdist = np.array([np.interp(newvpecmesh, vpecmesh[sel][ii], pdfs_at_fixed_nn[ii], left=0, right=0) for ii in range(0,len(sel[0]))])
        print(combinedvpecdist.shape)
        combinedvpecdist = np.hsplit(combinedvpecdist, 2)
        print(combinedvpecdist[0].shape)
        #combinedvpecdist = np.vstack([np.abs(combinedvpecdist[0]),np.abs(combinedvpecdist[1])])
        combinedvpecdist = np.vstack([np.flip(combinedvpecdist[0]), combinedvpecdist[1]])
        combinedvpecdist = np.nansum(combinedvpecdist,axis=0)
        print(combinedvpecdist.shape)
        plt.figure()
        plt.title(str(nn))
        finalvpec = np.hsplit(newvpecmesh,2)[1]
        plt.plot(finalvpec, combinedvpecdist)
        plt.axvline(pg3.get_median_eCDF(finalvpec, combinedvpecdist),color='k')
        plt.show()
        
        #print(pdfs_at_fixed_nn.shape)
        #print(grpn[grpn==nn].shape)
        #print(nn)
        #print((pdfs_at_fixed_nn / simps(pdfs_at_fixed_nn, zpdfs['zmesh'])[:,np.newaxis]).shape)
        #print(pdfs_at_fixed_nn.shape)
        break
sel = np.where(zpdfs['grpid']==14)
print(sel)

plt.figure()
plt.plot(3e5*zpdfs['zmesh']-grpcz[14], zpdfs['pdf'][14],color= 'k')
plt.show()
plt.figure()
plt.plot(3e5*zpdfs['zmesh']-grpcz[92], zpdfs['pdf'][92],color= 'k')
plt.show()
"""
