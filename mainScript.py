# -*- coding: utf-8 -*-
# pylint: disable=invalid-name
"""
Created on Fri Mar 20 15:53 2020
Try to find a map and measure for the clustering density in an image.
@author: jonatan.alvelid
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import cv2 as cv
from findmaxima import findmaxima
from densitymap import densitymap

#def main():
#"""Does the main job."""
dirpath = 'T:/Jonatan/Collaborations/Immunoreceptors - Trixy (KI)/RedSTED Data/2020-02-27/To share - tif'

filepath = os.path.join(dirpath, 'C-Cell010.tif')

# read image and preprocess
imgraw = cv.imread(filepath)
imgraw = imgraw[400:600, 300:500, 1]
#imgraw = imgraw[:,:,1]
img = ndi.gaussian_filter(imgraw, 0.8)

# img parameters
pxs_nm = 30  # pixel size in nm

# get the peak coordinates
coords_peaks = findmaxima(img, thresh_abs=7, min_dist=2)

# get density map of peaks
bandwidth_nm = 120
X, Y, Z = densitymap(coords_peaks, img, bandwdt=bandwidth_nm/pxs_nm)

# plotting
# plotting of raw image and peak coordinates
fig, axs = plt.subplots(2, 2)
axs[0, 0].imshow(imgraw, interpolation='none', cmap=plt.cm.hot)
axs[0, 1].imshow(img, interpolation='none', cmap=plt.cm.hot)
axs[0, 1].plot(coords_peaks[:, 1], coords_peaks[:, 0], 'g.')
# plot contours of the density estimate
levels = np.linspace(Z.max()/3, Z.max(), 25)
axs[1, 0].contourf(X, Y, Z, levels=levels, cmap=plt.cm.Reds)
axs[1, 0].set_aspect('equal', 'box')

fig.tight_layout()
plt.show()

#main()
