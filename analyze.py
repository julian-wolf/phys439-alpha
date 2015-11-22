# -*- coding: utf-8 -*-

import numpy   as np
import spinmob as sm
from spectrum import get_spectrum_chn

# bins used by Maestro
bins = np.arange(0, 2048)

# data used for calibration
pulse_heights_calib = np.array([260, 284, 304, 330, 430]) # 380 is all weird-looking

fnames_pulse_calib  = ["data/calib/calib" + str(height) + ".Chn"
                       for height in pulse_heights_calib]
spectra_pulse_calib = [get_spectrum_chn(fname)
                       for fname in fnames_pulse_calib]

spectrum_Am_calib = get_spectrum_chn("data/calib/calib_Am_25mTorr.Chn")

fnames_short_charge  = ["data/short_charge/Tr%03d.Chn" % (n,)
                        for n in range(270)]
spectra_short_charge = [get_spectrum_chn(fname)
                        for fname in fnames_short_charge]

fnames_long_charge   = ["data/long_charge/Tr%03d_long.Chn" % (n,)
                        for n in range(264)]
spectra_long_charge  = [get_spectrum_chn(fname)
                        for fname in fnames_long_charge]

def _gaussian(x, sigma):
    norm_factor  = 1 / (sigma * np.sqrt(2 * np.pi))
    distribution = np.exp(-x**2 / (2 * sigma**2))
    return norm_factor * distribution

def _lorentzian(x, gamma):
    norm_factor  = gamma / np.pi
    distribution = 1 / (x**2 + gamma**2)
    return norm_factor * distribution

def get_yerr_counts(spectrum):
    """
    Gets the poisson error in number of counts
    """
    return np.sqrt(spectrum.counts())

def get_yerr_count_rates(spectrum):
    """
    Gets the poisson error in count rates
    """
    return np.sqrt(spectrum.count_rates())

def fit_peak(spectrum, f, p, xmin, xmax, g={'G' : _gaussian, 'L' : _lorentzian}):
    """
    Fits a function f with parameters p to dataset
    """
    peak_fit = sm.data.fitter(f=f, p=p, g=g, plot_guess=True)
    yerr     = get_yerr_counts(spectrum)

    peak_fit.set_data(xdata=bins, ydata=spectrum.counts(), eydata=yerr)
    peak_fit.set(xmin=xmin, xmax=xmax)
    peak_fit.fit()

    return peak_fit

def calibrate(f="a*x+b", p="a,b"):
    """
    Automate calibration of the apparatus;
    returns function which converts bin numbers to energies
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
    offset_err = np.sqrt(offset_err)

    # location of the true americium peak, minus gold foil stuff
    peak_loc_true = 5.4856 - 0.033
    peak_err_true = 0.001

    peak_fit = fit_peak(spectrum_Am_calib, "norm*G(x-x0,sigma)+bg",
                        "norm=2400,x0=1300,sigma=18,bg=1", 1295, 1320)

    # location of the observed americium peak
    peak_loc_measured = peak_fit.results[0][1]
    peak_err_measured = peak_fit.results[1][1][1]
    peak_err_measured = np.sqrt(peak_err_measured)

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

def _get_total_count_rates(spectra, bin_low):
    rates  = [np.sum(spec.count_rates()[bin_low:]) for spec in spectra]
    errors = [np.mean(get_yerr_count_rates(spec)[bin_low:]) for spec in spectra]
    return (rates, errors)

def _get_total_elapsed_times(spectra):
    atime_first = spectra[0].absolute_time()
    return [spec.absolute_time() - atime_first for spec in spectra]

def find_halflife(spectra, f, p, xmin=None, bin_low=500, coarsen=10):
    """
    Finds the half-life of a sample based on the corresponding spectra
    """
    (count_rates, count_errs) = _get_total_count_rates(spectra, bin_low)
    elapsed_times = _get_total_elapsed_times(spectra)

    hl_fit = sm.data.fitter(f=f, p=p)
    hl_fit.set_data(xdata=elapsed_times, ydata=count_rates, eydata=count_errs)
    hl_fit.set(coarsen=coarsen)
    if not xmin is None: hl_fit.set(xmin=xmin)
    hl_fit.fit()

    return hl_fit

def _analyze_hl(spectra, xmin, coarsen):
    """
    Automates analysis of halflife data
    """
    l1_expected = np.log(2) / (10.64 * 60 * 60)
    l2_expected = np.log(2) / (60.60 * 60)

    f_activity = "N0*l1*l2*(exp(-l1*x)-exp(-l2*x))/(l2-l1)"
    p_activity = "N0,l1=%f,l2=%f" % (l1_expected, l2_expected)

    hl_fit = find_halflife(spectra, f_activity, p_activity, xmin=xmin, coarsen=coarsen)

    hl_Pb212 = np.log(2) / hl_fit.results[0][1]
    hl_Bi212 = np.log(2) / hl_fit.results[0][2]

    hl_Pb212_err = np.sqrt(hl_fit.results[1][1][1])*np.log(2) / hl_fit.results[0][1]**2
    hl_Bi212_err = np.sqrt(hl_fit.results[1][2][2])*np.log(2) / hl_fit.results[0][2]**2

    return ((hl_Pb212, hl_Bi212), (hl_Pb212_err, hl_Bi212_err),
            hl_fit.reduced_chi_squareds())

def analyze_hl_long_charge():
    return _analyze_hl(spectra_long_charge, 5000, 10)

def analyze_hl_short_charge():
    return _analyze_hl(spectra_short_charge, 19000, 12)

def print_data_to_columns(sm_fit, fname, residuals=False):
    xmin = sm_fit._settings['xmin']
    xmax = sm_fit._settings['xmax']

    xdata  = sm_fit.get_data()[0][0]
    i_used = (xdata >= xmin) & (xdata <= xmax)
    xdata  = xdata[i_used]

    if not residuals:
        ydata  = sm_fit.get_data()[1][0][i_used]
        eydata = sm_fit.get_data()[2][0][i_used]
    else:
        ydata  = sm_fit.studentized_residuals()[0]
        eydata = (0 * ydata) + 1

    with open(fname, 'w') as f_out:
        n_data = len(xdata);
        for i in range(n_data):
            entry = "%f\t%f\t%f\n" % (xdata[i], ydata[i], eydata[i])
            f_out.write(entry)

    return
