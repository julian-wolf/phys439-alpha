import numpy   as np
import spinmob as sm
from spectrum import get_spectrum_chn

# bins used by Maestro
bins = np.arange(0,2048)

# data used for calibration
pulse_heights_calib = np.array([260, 284, 304, 330, 430]) # 380 is all weird-looking

fnames_pulse_calib  = ["data/calib" + str(height) + ".Chn"
                       for height in pulse_heights_calib]
spectra_pulse_calib = [get_spectrum_chn(fname)
                       for fname in fnames_pulse_calib]

spectrum_Am_calib = get_spectrum_chn("data/calib_Am.Chn")

def _gaussian(x, sigma):
    norm_factor  = 1 / (sigma * np.sqrt(2 * np.pi))
    distribution = np.exp(-x**2 / (2 * sigma**2))
    return norm_factor * distribution

def _lorentzian(x, gamma):
    norm_factor  = gamma / np.pi
    distribution = 1 / (x**2 + gamma**2)
    return norm_factor * distribution

def get_yerr(spectrum):
    """
    Gets the poisson error
    """
    return np.sqrt(spectrum.counts())

def fit_peak(spectrum, f, p, xmin, xmax, g={'G' : _gaussian, 'L' : _lorentzian}):
    """
    Fits a function f with parameters p to dataset
    """
    peak_fit = sm.data.fitter(f=f, p=p, g=g, plot_guess=True)
    yerr     = get_yerr(spectrum)

    peak_fit.set_data(xdata=bins, ydata=spectrum.counts(), eydata=yerr)
    peak_fit.set(xmin=xmin, xmax=xmax)
    peak_fit.fit()

    return peak_fit

def calibrate(f="a*x+b", p="a,b"):
    """
    Automate calibration of the apparatus
    """
    # first-guess locations of peaks
    x0 = np.array([709, 777.3, 837, 914.5, 1190])

    xlim_offset = 22
    xmin = x0 - xlim_offset
    xmax = x0 + xlim_offset

    # parameters to be sent to sm.data.fitter constructor
    fit_func = "norm*((eta)*L(x-x0,gamma)+(1-eta)*G(x-x0,sigma))+bg"
    params   = "norm=7000,eta=0.12,gamma=2,sigma=2,bg=3,x0="
    params   = [params + str(x) for x in x0]

    # location of x0 in parameter list
    peak_loc_ind = 5

    peak_loc  = np.zeros(len(spectra_pulse_calib))
    peak_err  = np.zeros(len(spectra_pulse_calib))
    good_fits = [True] * len(spectra_pulse_calib)
    for i in range(len(spectra_pulse_calib)):
        peak_fit = fit_peak(spectra_pulse_calib[i], fit_func,
                            params[i], xmin[i], xmax[i])

        # ignore peaks whose fits don't converge to simplify debugging
        if peak_fit.results[1] is None:
            good_fits[i] = False
            continue

        peak_loc[i] = peak_fit.results[0][peak_loc_ind]
        peak_err[i] = peak_fit.results[1][peak_loc_ind][peak_loc_ind]
        peak_err[i] = np.sqrt(peak_err[i])

    good_fits = np.array(good_fits, dtype=bool)

    pulse_heights = pulse_heights_calib[good_fits]
    peak_loc = peak_loc[good_fits]
    peak_err = peak_err[good_fits]

    # fit the peak locations to find the calibration curve
    calib_fit = sm.data.fitter(f=f, p=p, plot_guess=False)
    calib_fit.set_data(xdata=pulse_heights, ydata=peak_loc,
                       exdata=2, eydata=peak_err)
    calib_fit.fit()

    offset     = calib_fit.results[0][1]
    offset_err = calib_fit.results[1][1][1]

    # location of the true americium peak, minus gold foil stuff
    peak_loc_true = 5.4856 - 0.033
    peak_err_true = 0.001

    peak_fit = fit_peak(spectrum_Am_calib, "norm*G(x-x0,sigma)+bg",
                        "norm=2400,x0=1042,sigma=18,bg=1", 1025, 1090)

    # location of the observed americium peak
    peak_loc_measured = peak_fit.results[0][1]
    peak_err_measured = peak_fit.results[1][1][1]

    slope     = peak_loc_true / (peak_loc_measured - offset)
    slope_err = np.sqrt((   (peak_err_true) /
                            (peak_loc_measured - offset))**2 +
                        (   (peak_loc_true * peak_err_measured) /
                            (peak_loc_measured - offset)**2)**2 +
                        (   (peak_loc_true * offset_err) /
                            (peak_loc_measured - offset)**2)**2)

    # produce a function which magically turns bins into energies
    def bin_to_energy(bin):
        energy     = slope * (bin - offset);
        energy_err = np.sqrt((bin - offset)**2 *  slope_err**2 +
                             (slope)**2        * offset_err**2)
        return (energy, energy_err)

    return bin_to_energy

