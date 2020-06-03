# coding: utf-8
# pylint: disable=invalid-unary-operand-type

# # Immunocell receptors analysis
# ## Nearest neighbor analysis
# 
# Find the nearest neighbors of the detected PD-1 peaks.
# Use k=10 and compare the different samples.
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
from scipy import stats
from tkinter.filedialog import askdirectory
from sklearn.neighbors import NearestNeighbors

from densitymap import density_map
from findmaxima import find_maxima
from binarycellmap import binary_cell_map

# Define parameter constants
allimgs = True  # parameter to check if you want to loop through all imgs or just analyse one
dirpath = askdirectory(title='Choose your folder...',initialdir='E:/PhD/Data analysis/Immunoreceptors - temp copy/RedSTED Data/2020-02-27')  # directory path
print(dirpath)
difgaus_sigmahi_nm = 100  # difference of gaussians high_sigma in nm
sm_size_nm = 15  # smoothing Gaussian size in nm
peakthresh = 2.5  # absolute intensity threshold for peak detection
minpeakdist = 1  # minimum distance between peaks in pixels for peak detection - CONSIDER CHANGING THIS TO NM? OR not, since the detection of p2p distances depends on the pixel size.
k_nn = 4  # k for nearest neighbor analysis
samples = ['A','B','C']  # sample names
dist_all = [[],[],[]]  # all lists for the three samples as a nested list
histrange = 800  # range of the distance for the histogram of NN-distances
histbins = 30  # number of bins for the histogram of NN-distances

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

    # Perform nearest neihbor algorithm for k_neighbors=10
    nbrs = NearestNeighbors(n_neighbors=k_nn+1, algorithm='ball_tree').fit(coords_peaks)
    distances, indices = nbrs.kneighbors(coords_peaks)
    dist_all[samples.index(imgname[0])].extend(distances[:,1:].flatten()*pxs_nm)


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
    }
    anaresdictname = imgname+'_anares_NN.txt'
    with open(os.path.join(dirpath, anaresdictname),'w') as file:
        file.write(json.dumps(analysis_dict))

# Plot histogram of nearest neighbors for all three samples
(na, binsa, patchesa) = plt.hist(dist_all[0], bins=histbins, range=(0,histrange), density=True, rwidth=0.3, align='mid')
(nb, binsb, patchesb) = plt.hist(dist_all[1], bins=histbins, range=(0,histrange), density=True, rwidth=0.3, align='mid')
(nc, binsc, patchesc) = plt.hist(dist_all[2], bins=histbins, range=(0,histrange), density=True, rwidth=0.3, align='mid')
fig = plt.figure(figsize = (15,10), frameon=False)
plt.hist(np.array(dist_all[0])-histrange/histbins/4, bins=binsa-histrange/histbins/4, range=(0,histrange), density=True, rwidth=0.22, align='mid')
plt.hist(dist_all[1], bins=binsb, range=(0,800), density=True, rwidth=0.22, align='mid')
plt.hist(np.array(dist_all[2])+histrange/histbins/4, bins=binsc+histrange/histbins/4, range=(0,histrange), density=True, rwidth=0.22, align='mid')
plt.legend(samples,fontsize='xx-large')
plt.xlim(0, histrange)
plt.show()

# Perform KS-tests on the CDFs for the three samples
ks_ab_gr = stats.ks_2samp(dist_all[0],dist_all[1],alternative='greater').pvalue
ks_ac_gr = stats.ks_2samp(dist_all[0],dist_all[2],alternative='greater').pvalue
ks_bc_gr = stats.ks_2samp(dist_all[1],dist_all[2],alternative='greater').pvalue
ks_ab_ls = stats.ks_2samp(dist_all[0],dist_all[1],alternative='less').pvalue
ks_ac_ls = stats.ks_2samp(dist_all[0],dist_all[2],alternative='less').pvalue
ks_bc_ls = stats.ks_2samp(dist_all[1],dist_all[2],alternative='less').pvalue
ks_ab = stats.ks_2samp(dist_all[0],dist_all[1]).pvalue
ks_ac = stats.ks_2samp(dist_all[0],dist_all[2]).pvalue
ks_bc = stats.ks_2samp(dist_all[1],dist_all[2]).pvalue

# Save histogram to tiff-file
save_path_denmap = os.path.join(dirpath, "NN-histogram.svg")
fig.savefig(save_path_denmap, format='svg')

# Save KS-test results and total info to file
param_dict = {
    "Number of peaks (A)": int(len(dist_all[0])/k_nn),
    "Number of peaks (B)": int(len(dist_all[1])/k_nn),
    "Number of peaks (C)": int(len(dist_all[2])/k_nn),
    "KS-test (A-B)": float(format(ks_ab, '.2e')),
    "KS-test (A-C)": float(format(ks_ac, '.2e')),
    "KS-test (B-C)": float(format(ks_bc, '.2e')),
    "KS-test, greater CDF (A-B)": float(format(ks_ab_gr, '.2e')),
    "KS-test, greater CDF (A-C)": float(format(ks_ac_gr, '.2e')),
    "KS-test, greater CDF (B-C)": float(format(ks_bc_gr, '.2e')),
    "KS-test, less CDF (A-B)": float(format(ks_ab_ls, '.2e')),
    "KS-test, less CDF (A-C)": float(format(ks_ac_ls, '.2e')),
    "KS-test, less CDF (B-C)": float(format(ks_bc_ls, '.2e')),
    "K nearest neighbors": k_nn
}
with open(os.path.join(dirpath, "analysis_results_NN.txt"),'w') as file:
    file.write(json.dumps(param_dict))

# Save all parameter constants to file
param_dict = {
    "High_sigma in difference of Gaussians (nm)": difgaus_sigmahi_nm,
    "Gaussian smoothing size (nm)": sm_size_nm,
    "Absolute intensity peak detection threshold (cnts)": peakthresh,
    "Minimum peak distance (pxs)": minpeakdist,
    "K nearest neighbors": k_nn
}
with open(os.path.join(dirpath, "analysis_params_NN.txt"),'w') as file:
    file.write(json.dumps(param_dict))
