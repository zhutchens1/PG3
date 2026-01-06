import numpy as np

# from https://gist.github.com/robbibt/c7ec5f0cb3e4e0cee5ed3156bcb666de

def weighted_median(values, weights):
    """
    Compute the weighted median of an array of values.
    
    This implementation sorts values and computes the cumulative
    sum of the weights. The weighted median is the smallest value for
    which the cumulative sum is greater than or equal to half of the
    total sum of weights.
    Parameters
    ----------
    values : array-like
        List or array of values on which to calculate the weighted median.
    weights : array-like
        List or array of weights corresponding to the values.
    Returns
    -------
    float
        The weighted median of the input values.
    """
    # Convert input values and weights to numpy arrays
    values = np.array(values)
    weights = np.array(weights)
    
    # Get the indices that would sort the array
    sort_indices = np.argsort(values)
    
    # Sort values and weights according to the sorted indices
    values_sorted = values[sort_indices]
    weights_sorted = weights[sort_indices]  

    # Compute the cumulative sum of the sorted weights
    cumsum = weights_sorted.cumsum()
    
    # Calculate the cutoff as half of the total weight sum
    cutoff = weights_sorted.sum() / 2.
    
    # Return the smallest value for which the cumulative sum is greater
    # than or equal to the cutoff
    return values_sorted[cumsum >= cutoff][0]
