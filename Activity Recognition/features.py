# -*- coding: utf-8 -*-
"""
This file is used for extracting features over windows of tri-axial accelerometer 
data. We recommend using helper functions like _compute_mean_features(window) to 
extract individual features.

As a side note, the underscore at the beginning of a function is a Python 
convention indicating that the function has private access (although in reality 
it is still publicly accessible).

"""

import numpy as np
from scipy.signal import find_peaks

def _compute_mean_features(window):
    """
    Computes the mean x, y and z acceleration over the given window. 
    """
    return np.mean(window, axis=0)

def _compute_min_features(window):
    """
    Computes the mean x, y and z acceleration over the given window. 
    """
    return np.amin(window, axis=0)

def _compute_median_features(window):
    """
    Computes the median x, y and z acceleration over the given window. 
    """
    return np.median(window, axis=0)

# # TODO: define functions to compute more features

def _compute_var_features(window):
    """
    Computes the variance x, y and z acceleration over the given window. 
    """
    return np.var(window, axis=0)

def _fft(window):
    """
    Computes the discrete Fourier Transform over x, y and z acceleration over the given window. 
    """
    return np.mean(np.fft.rfft(np.sqrt(window[:, 0]**2+window[:, 1]**2+window[:, 2]**2), axis = 0).astype(float)) 


def _compute_peaks(window):
    peaks = find_peaks(list(np.sqrt(window[0]**2+window[1]**2+window[2]**2)))
    return [len(peaks)]
    

def _entropy(window):
    hist = np.histogram(window, bins = 5, density=True)
    data = hist[0]
    data = data / data.sum()
    entropyy = -(data*np.log(np.abs(data)))
    noNan = np.isnan(entropyy)
    index = ~noNan
    entropy = entropyy[index]
    return entropy.sum()

def extract_features(window):
    """
    Here is where you will extract your features from the data over 
    the given window. We have given you an example of computing 
    the mean and appending it to the feature vector.
    
    """
    
    x = []
    feature_names = []

    x.append(_compute_mean_features(window))
    feature_names.append("x_mean")
    feature_names.append("y_mean")
    feature_names.append("z_mean")
    
#     TODO: call functions to compute other features. Append the features to x and the names of these features to feature_names
    x.append(_compute_median_features(window))
    feature_names.append("x_median")
    feature_names.append("y_median")
    feature_names.append("z_median")
    
    x.append(_compute_min_features(window))
    feature_names.append("x_min")
    feature_names.append("y_min")
    feature_names.append("z_min")
    
    x.append([_entropy(window)])
    feature_names.append("entropy")

    x.append([_fft(window)])
    feature_names.append("fft")
    
    x.append(_compute_var_features(window))
    feature_names.append("x_var")
    feature_names.append("y_var")
    feature_names.append("z_var")
    x.append(_compute_peaks(window))
    feature_names.append("peaks")
    
    


    feature_vector = np.concatenate(x, axis=0) # convert the list of features to a single 1-dimensional vector
    return feature_names, feature_vector