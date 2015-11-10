import numpy   as np
import spinmob as sm
from spectrum import get_spectrum_chn

pulse_heights_calib = [260, 284, 304, 330, 430] # 380 is all weird-looking
fnames_calib = ["data/calib" + str(height) + ".Chn" for height in pulse_heights_calib]
spectra_calib = [get_spectrum_chn(fname) for fname in fnames_calib]

def get_yerr(dataset):
    """
    Gets the poisson error
    """
    return np.sqrt(dataset.c(1))

def fit(dataset, f, p, xmin=10, xmax=110, g=None):
    """
    Fits a function f with parameters p to dataset
    """
    peak_fit = sm.data.fitter(f=f, p=p, g=g, plot_guess=True)
    yerr     = get_yerr(dataset)

    peak_fit.set_data(xdata=dataset.c(0), ydata=dataset.c(1), eydata=yerr)
    peak_fit.set(xmin=xmin, xmax=xmax)
    peak_fit.fit()

    return peak_fit

def calibrate():
    """
    Automate calibration of the apparatus
    """
    # TODO

