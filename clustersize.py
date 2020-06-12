import numpy as np
from skimage import measure
from scipy.optimize import curve_fit

def cluster_size(img, peak_coords, pxs_nm, fittol=0.7, linelen=400, widmax=200):
    """Take line profiles at the given coordinates, in various directions, and give the size of the cluster back."""
    
    def lorentzian(x, a, b, c, d):
        """Definition of Lorentzian fitting function."""
        return a/(((x-b)/(0.5*c))**2+1)+d

    def lozfit(xdata, ydata, widtol=200):
        """Perform the fitting and return parameters of interest."""
        try:
            popt, pcov = curve_fit(lorentzian, xdata, ydata, bounds=([0, xdata[-1]/2-100, 0, 0], [max(ydata)*1.5, xdata[-1]/2+100, widtol, max(ydata)/2]))
        except:
            return (0,0,0,0)
        fwhm = popt[2]
        # Calculate r-sq value
        residuals = ydata - lorentzian(xdata, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((ydata-np.mean(ydata))**2)
        r_sq = 1 - (ss_res/ss_tot)
        return fwhm, r_sq, popt, pcov
    
    def lineprofile(img, startpoint, endpoint, linewidth=1, pxsize=1):
        """Extracts a line profile from img, according to the parameters."""
        # Make coordinates for the line
        x0, y0 = startpoint[0], startpoint[1]  # pixel coordinates
        x1, y1 = endpoint[0], endpoint[1]  # pixel coordinates
        # Extract the values along the line
        linelength = np.sqrt((x1-x0)**2+(y1-y0)**2)*pxsize
        zi = measure.profile_line(img, (y0, x0), (y1, x1), linewidth=linewidth, mode='constant')
        ci = np.linspace(0, linelength, zi.size)
        # Return the intensity profile and the corresponding coordinate vector
        return zi, ci
    
    def get_line_profile(angle):
        ys, xs = round(xm - linelen/pxs_nm/2*np.cos(angle)), round(ym - linelen/pxs_nm/2*np.sin(angle))
        ye, xe = round(xm + linelen/pxs_nm/2*np.cos(angle)), round(ym + linelen/pxs_nm/2*np.sin(angle))
        zi, ci = lineprofile(img, (xs,ys), (xe,ye), linewidth=1, pxsize=pxs_nm)
        return ci, zi

    xm, ym = peak_coords
    
    # X-profile
    # Get line profile and fit with Lorentzian
    ci, zi = get_line_profile(0)
    fwhm_x, r_sq_x = lozfit(ci,zi,widmax)[0:2]

    # Y-profile
    # Get line profile and fit with Lorentzian
    ci, zi = get_line_profile(np.pi/2)
    fwhm_y, r_sq_y = lozfit(ci,zi,widmax)[0:2]
    
    # 45deg-profile
    # Get line profile and fit with Lorentzian
    ci, zi = get_line_profile(np.pi/4)
    fwhm_45, r_sq_45 = lozfit(ci,zi,widmax)[0:2]
    
    # -45deg-profile
    # Get line profile and fit with Lorentzian
    ci, zi = get_line_profile(-np.pi/4)
    fwhm_n45, r_sq_n45 = lozfit(ci,zi,widmax)[0:2]

    r_sqs = [r_sq_x, r_sq_y, r_sq_45, r_sq_n45]
    fwhms = [fwhm_x, fwhm_y, fwhm_45, fwhm_n45]
    fwhms_goodfit = [fwhm for fwhm in fwhms if r_sqs[fwhms.index(fwhm)]>fittol and round(fwhm)<widmax and round(fwhm)>pxs_nm]
    if len(fwhms_goodfit) != 0:
        fwhm = min(fwhms_goodfit)
    else:
        fwhm = []
    return fwhm
