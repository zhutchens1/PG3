from scipy.stats import binned_statistic

def center_binned_stats(*args, **kwargs):
    """
     Same as scipy.stats.binned_statistic, but returns
     the bin centers (matching length of `statistic`)
     instead of the binedges.

     See docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binned_statistic.html
    """
    stat, binedges, binnumber = binned_statistic(*args,**kwargs)
    bincenters = (binedges[:-1]+binedges[1:])/2.
    return stat, bincenters, binedges, binnumber
