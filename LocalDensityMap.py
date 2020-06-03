# coding: utf-8
# pylint: disable=invalid-unary-operand-type

# # Immunocell receptors analysis
# ## Local density map
# 
# Try to find a map and measure for the clustering density in an image.
# 
# @author: jonatan.alvelid
# Copied jupyter code: 2020-05-28

# Import packages
import os
import glob
import json
import tifffile
import numpy as np
import matplotlib.pyplot as plt
import skimage.filters as skfilt
from scipy import ndimage as ndi
from tkinter.filedialog import askdirectory

from densitymap import density_map
from findmaxima import find_maxima
from binarycellmap import binary_cell_map

# Define parameter constants
allimgs = False  # parameter to check if you want to loop through all imgs or just analyse one
dirpath = askdirectory(title='Choose your folder...',initialdir='E:/PhD/Data analysis/Immunoreceptors - temp copy/RedSTED Data/2020-02-27')  # directory path
print(dirpath)
difgaus_sigmahi_nm = 100  # difference of gaussians high_sigma in nm
sm_size_nm = 15  # smoothing Gaussian size in nm
peakthresh = 2.5  # absolute intensity threshold for peak detection
minpeakdist = 1  # minimum distance between peaks in pixels for peak detection - CONSIDER CHANGING THIS TO NM?
bandwidth_nm = 200  # bandwidth in nm for density map of peaks

if allimgs:
    files = glob.glob(os.path.join(dirpath,'*[0-9].tif'))
    
else:
    files = [os.path.join(dirpath,'C-Cell012.tif')]

print([path.replace(dirpath+'\\','') for path in files])

for filepath in files:
    imgname = filepath.split('\\')[1].split('.')[0]
    print(imgname)
    # Load raw image file and read pixel size from metadata
    #filepath = os.path.join(dirpath, imgname)
    with tifffile.TiffFile(filepath) as tif:
        imgraw = tif.pages[0].asarray()  # image as numpy array
        pxs_nm = 1e9/tif.pages[0].tags['XResolution'].value[0]  # pixel size in nm

    # Get binary mask and multiple the pre-processed image with this (should I do this before pre-processing? does it make a difference?).
    binarymap = binary_cell_map(imgraw, pxs_nm=pxs_nm)
    img = imgraw*binarymap

    # Preprocess image with a difference of gaussians filter and a gaussian blurring
    # take the difference of gaussians to minimize faint out of focus noise
    img = skfilt.difference_of_gaussians(img, low_sigma=0, high_sigma=difgaus_sigmahi_nm/pxs_nm)
    img[img < 0] = 0  # remove any negative values in the image
    # gaussian smoothing of the image
    img = ndi.gaussian_filter(img, sm_size_nm/pxs_nm)

    # Standardize the image by dividing by a factor of mean+std, to standardize all images to ~the same range of values (assuming the intensity distribution is similar)
    imgmean = np.ma.masked_array(img,~binarymap).mean()
    imgstd = np.ma.masked_array(img,~binarymap).std()
    img = np.array(img/(imgmean+imgstd))

    # Get the coordinates of the peaks in the pre-processed image
    coords_peaks = find_maxima(img, thresh_abs=peakthresh, min_dist=minpeakdist)

    # Get density map of peaks
    X, Y, Zraw = density_map(coords_peaks, img, bandwdt=bandwidth_nm/pxs_nm)

    # Gaussian kernel un-normalization, to the number of receptors in kernel
    totnumpeaks = np.size(coords_peaks[:,0])
    peakval = 1/(2*np.pi*np.power(bandwidth_nm/pxs_nm,2))/totnumpeaks  # peak value of a single Gaussian peak in Z
    Z = Zraw/peakval


    # Plot detected receptors map and save
    fig = plt.figure(figsize = (15,15), frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    imgplot0=plt.imshow(imgraw, interpolation='none', cmap=plt.cm.hot)
    plt.plot(coords_peaks[:, 1], coords_peaks[:, 0], 'g.', markersize=6)
    detrecname = imgname+'_detrec.tif'
    save_path_detcor = os.path.join(dirpath, detrecname)
    fig.savefig(save_path_detcor, transparent=True, pad_inches=0)
    plt.close(fig)

    # Save density map to tiff-file
    denmapname = imgname+'_denmap.tif'
    save_path_denmap = os.path.join(dirpath, denmapname)
    tifffile.imwrite(save_path_denmap,Z.astype(np.float32),imagej=True)

    # Save binary cell map to tiff-file
    bincelname = imgname+'_cellmask.tif'
    save_path_bincel = os.path.join(dirpath, bincelname)
    tifffile.imwrite(save_path_bincel,binarymap.astype(np.uint8),imagej=True)

    # Save detector receptor coordinates to .txt-file
    detreccorname = imgname+'_detreccor.txt'
    save_path_detreccor = os.path.join(dirpath, detreccorname)
    np.savetxt(save_path_detreccor, coords_peaks, fmt='%i')

    # Save a dictionary with information about the analysis
    analysis_dict = {
    "Raw image max": int(imgraw.max()),
    "Raw image mean (masked)": np.ma.masked_array(imgraw,~binarymap).mean(),
    "Raw image std (masked)": np.ma.masked_array(imgraw,~binarymap).std(),
    "Number of peaks": len(coords_peaks),
    "KDE max": Z.max(),
    "KDE mean (masked)": np.ma.masked_array(Z,~binarymap).mean(),
    "KDE std (masked)": np.ma.masked_array(Z,~binarymap).std(),
    }
    anaresdictname = imgname+'_anares_KDE.txt'
    with open(os.path.join(dirpath, anaresdictname),'w') as file:
        file.write(json.dumps(analysis_dict))

# Save all parameter constants to file
param_dict = {
    "High_sigma in difference of Gaussians (nm)": difgaus_sigmahi_nm,
    "Gaussian smoothing size (nm)": sm_size_nm,
    "Absolute intensity peak detection threshold (cnts)": peakthresh,
    "Minimum peak distance (pxs)": minpeakdist,
    "Bandwidth for density map of peaks (nm)": bandwidth_nm
}
with open(os.path.join(dirpath, "analysis_params_KDE.txt"),'w') as file:
    file.write(json.dumps(param_dict))
