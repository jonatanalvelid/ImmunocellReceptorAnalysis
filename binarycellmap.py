from skimage import morphology
from scipy import ndimage as ndi
import skimage.filters as skfilt

def binary_cell_map(img, gaussstd_nm=100, pxs_nm=25, num_di=12, num_er=5):
    """Get a binary map of the cell, from the PD-1 image."""
    # gaussian smoothing of the image
    gaussstd_nm = 100  # bandwidth in nm
    gaussstd = gaussstd_nm/pxs_nm
    img = ndi.gaussian_filter(img, gaussstd)
    binary = img > skfilt.threshold_li(img)
    # dilate and erode to smooth out edges
    for i in range(0,num_di):
        binary = ndi.binary_dilation(binary)
    for i in range(0,num_er):
        binary = ndi.binary_erosion(binary)
    # fill holes
    binary = ndi.binary_fill_holes(binary)
    # remove stray background dots
    binary = morphology.remove_small_objects(binary, 200)
    #binary = binary.astype(int)
    return binary