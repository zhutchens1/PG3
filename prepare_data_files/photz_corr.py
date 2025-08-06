import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

files = dict(
    redshift_catalog = './xmmservs_w_absrmag_specz.csv',
)


if __name__=='__main__':
    # split redshift catalog into dz=0.2 bins
    xmms = files["redshift_catalog"]
    fitting_bins = np.arange(0,1.4,0.2)
    xmms.loc[:,'zphotcorr_bin'] = np.digitize(xmms.zbest.to_numpy(), bins=fitting_bins)
