# pylint: disable=invalid-name

import numpy as np
from sklearn.neighbors import KernelDensity

def densitymap(coords_peaks, img, bandwdt=6):
    """Get density map of image based on input list of coordinates."""
    # prep training data
    datatrain = np.array(coords_peaks)
    # make a kernel density estimation from the coordinates, to get local cluster density
    kde = KernelDensity(bandwidth=bandwdt)
    kde.fit(datatrain)
    # set up data grid for contour plot
    grid_size = 1
    xmin, xmax, ymin, ymax = 1, np.shape(img)[0], 1, np.shape(img)[0]
    xgrid = np.arange(xmin, xmax, grid_size)
    ygrid = np.arange(ymin, ymax, grid_size)
    X, Y = np.meshgrid(xgrid[::1], ygrid[::1][::-1])
    xy = np.vstack([Y.ravel(), X.ravel()]).T
    Z = np.exp(kde.score_samples(xy))
    Z = np.flip(Z.reshape(X.shape), 0)
    return X, Y, Z
    