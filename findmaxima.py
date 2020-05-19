# pylint: disable=invalid-name

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max

def findmaxima(img, min_dist=3, thresh_abs=3):
    """Finds all local peaks in image and returns them in a list."""
    # find map with all pixels that are local peaks with min_dist and abs._thresh
    is_peak = peak_local_max(img, min_distance=min_dist, threshold_abs=thresh_abs, num_peaks_per_label=1, indices=False)
    # label all peaks with a connectivity of structure (8-connectivity as of now)
    labels_peaks = ndi.measurements.label(is_peak, structure=np.ones((3, 3)))[0]
    # merge coordinates close to each other to a common center of mass (8-connectivity)
    coords_merged = ndi.measurements.center_of_mass(is_peak, labels_peaks, range(1, np.max(labels_peaks)+1))
    coords_merged = np.array(coords_merged)
    return coords_merged
