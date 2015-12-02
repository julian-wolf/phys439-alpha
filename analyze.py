# -*- coding: utf-8 -*-

import numpy   as np
import spinmob as sm
from spectrum import get_spectrum_chn, add_spectra

# bins used by Maestro
bins = np.arange(0, 2048)

# data used for calibration
pulse_heights_calib_old = np.array([260, 284, 304, 330, 430]) # 380 is all weird-looking
pulse_heights_calib_new = np.array([110, 180, 250, 360])

# for differential pressure (stopping power) measurement
fnames_pulse_calib_old  = ["data/calib_old/calib" + str(height) + ".Chn"
                           for height in pulse_heights_calib_old]
spectra_pulse_calib_old = [get_spectrum_chn(fname)
                           for fname in fnames_pulse_calib_old]

# for energy, halflife, and branching ratio measurements
fnames_pulse_calib_new  = ["data/calib_new/calib_" + str(height) + ".Chn"
                           for height in pulse_heights_calib_new]
spectra_pulse_calib_new = [get_spectrum_chn(fname)
                           for fname in fnames_pulse_calib_new]

# for differential pressure (stopping power) measurement
spectrum_Am_calib_old = get_spectrum_chn("data/calib_old/calib_Am_25mTorr.Chn")

# for energy, halflife, and branching ratio measurements
spectrum_Am_calib_new = get_spectrum_chn("data/calib_new/calib_Am.Chn")

fnames_Tr  = ["data/long_charge/tr_%03d.Chn" % (n,) for n in range(264)]
spectra_Tr = [get_spectrum_chn(fname) for fname in fnames_Tr]

# in mBar
pressures_stopping = np.arange(750, 150, -50)

fnames_stopping  = ["data/pressures/SP_" + str(p) + "mb.Chn"
                    for p in pressures_stopping]
spectra_stopping = [get_spectrum_chn(fname) for fname in fnames_stopping]

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
    return np.sqrt(spectrum.counts()) / spectrum._etime

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

def calibrate(use_old, f="a*x+b", p="a,b"):
    """
    Automates calibration of the apparatus;
    returns function which converts bin numbers to energies
    """
    # first-guess locations of peaks
    if use_old:
        spectra_pulse = spectra_pulse_calib_old
        spectrum_Am   = spectrum_Am_calib_old
        x0 = np.array([709, 777.3, 837, 914.5, 1190])
        pulse_heights = pulse_heights_calib_old
    else:
        spectra_pulse = spectra_pulse_calib_new
        spectrum_Am   = add_spectra(spectra_pulse)
        x0 = np.array([258, 438, 620, 910, 1033])
        pulse_heights = pulse_heights_calib_new

    xlim_offset = 22
    xmin = x0 - xlim_offset
    xmax = x0 + xlim_offset

    # parameters to be sent to sm.data.fitter constructor
    fit_func = "norm*((eta)*L(x-x0,gamma)+(1-eta)*G(x-x0,sigma))+bg"
    params   = "norm=7000,eta=0.12,gamma=2,sigma=2,bg=3,x0="
    params   = [params + str(x) for x in x0]

    # location of x0 in parameter list
    peak_loc_ind = 5

    n_fits = len(spectra_pulse)

    peak_loc  = np.zeros(n_fits)
    peak_err  = np.zeros(n_fits)
    good_fits = [True] * n_fits
    for i in range(n_fits):
        peak_fit = fit_peak(spectra_pulse[i], fit_func, params[i], xmin[i], xmax[i])

        # ignore peaks whose fits don't converge to simplify debugging
        if peak_fit.results[1] is None:
            good_fits[i] = False
            continue

        peak_loc[i] = peak_fit.results[0][peak_loc_ind]
        peak_err[i] = peak_fit.results[1][peak_loc_ind][peak_loc_ind]
        peak_err[i] = np.sqrt(peak_err[i])

    good_fits = np.array(good_fits, dtype=bool)

    pulse_heights = pulse_heights[good_fits]
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

    if use_old:
        p_peak    = "norm=2400,x0=1300,sigma=18,bg=1"
        xmin_peak = 1295
        xmax_peak = 1320
    else:
        p_peak    = "norm=4100,x0=1181,sigma=2,bg=1"
        xmin_peak = 1177
        xmax_peak = 1205

    peak_fit = fit_peak(spectrum_Am, "norm*G(x-x0,sigma)+bg",
                        p_peak, xmin_peak, xmax_peak)

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
    errors = [np.sqrt(np.sum(spec.counts()[bin_low:])) / spec._etime
              for spec in spectra]
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

# TODO: fudge initial values; figure out fit
def analyze_hl():
    """
    Automates analysis of halflife data
    """
    l1_expected = np.log(2) / (10.64 * 60 * 60) # lambda for Pb212
    l2_expected = np.log(2) / (60.60 * 60)      # lambda for Bi212

    f_activity = "N0*l1*l2*(exp(-l1*x)-exp(-l2*x))/(l2-l1)"
    p_activity = "N0,l1=%f,l2=%f" % (l1_expected, l2_expected)

    hl_fit = find_halflife(spectra_Tr, f_activity, p_activity,
                           xmin=5000, coarsen=8)

    hl_Pb212 = np.log(2) / hl_fit.results[0][1]
    hl_Bi212 = np.log(2) / hl_fit.results[0][2]

    hl_Pb212_err = np.sqrt(hl_fit.results[1][1][1])*np.log(2) / hl_fit.results[0][1]**2
    hl_Bi212_err = np.sqrt(hl_fit.results[1][2][2])*np.log(2) / hl_fit.results[0][2]**2

    return ((hl_Pb212, hl_Bi212), (hl_Pb212_err, hl_Bi212_err),
            hl_fit.reduced_chi_squareds()[0])

# TODO: update fit parameters for new calibration
def analyze_energy(spectra=spectra_Tr, bin_to_energy=None):
    """
    Automates analysis of alpha energy and branching ratio data
    """
    spectrum_sum = add_spectra(spectra)

    # TODO: all these various parameters'll need to be updated for new calib data
    xmin=5.4
    xmax=6.6

    # x0 = [5.607, 5.769, 6.051, 6.090, 8.794]
    x0 = [5.607, 5.769, 6.051, 6.090]
    n0 = [20,    20,    200,   10]
    s0 = [0.003, 0.003, 0.004, 0.001]
    c0 = [1,     1,     1,     1]
    t0 = [100,   100,   100,   1]

    fit_func   = ""
    parameters = ""
    for i in range(len(x0)):
        fit_func   += "n"  + str(i) + "*G(x-x" + str(i) + ",s" + str(i) + ")" + \
                      "*(1+c" + str(i) + "*exp(t" + str(i) + "*(x" + str(i) + "-x)))+"
        parameters += "n"  + str(i) + "=" + str(n0[i]) + \
                      ",x" + str(i) + "=" + str(x0[i]) + \
                      ",s" + str(i) + "=" + str(s0[i]) + \
                      ",c" + str(i) + "=" + str(c0[i]) + \
                      ",t" + str(i) + "=" + str(t0[i]) + ","
    fit_func   += "bg"
    parameters += "bg=1"

    # save having to run calibrate() each time
    if bin_to_energy is None: bin_to_energy = calibrate(use_old=False)

    energy_bins = np.asarray([bin_to_energy(bin) for bin in bins])

    e_fit = sm.data.fitter(f=fit_func, p=parameters, g={'G' : _gaussian})
    e_fit.set_data( xdata=energy_bins[:,0],  ydata=spectrum_sum.counts(),
                   exdata=energy_bins[:,1], eydata=get_yerr_counts(spectrum_sum))
    e_fit.set(xmin=xmin, xmax=xmax)
    e_fit.fit()

    return e_fit

    ndof_per_peak = 5
    n0_inds = ndof_per_peak * np.arange(len(x0))
    x0_inds = n0_inds + 1
    s0_inds = n0_inds + 2

def analyze_stopping(spectra=spectra_stopping, bin_to_energy=None):
    """
    Automates analysis of stopping power (variable pressure) data
    """
    M_air = 28.96 # g / mol
    T_air = 294   # K
    R     = 8.314 # J / mol K    (negligible error)
    dist  = 4.8   # cm

    M_air_err = 0.01
    T_air_err = 2
    dist_err  = 0.1

    def thickness(P_air, P_air_err):
        # first, convert pressures from millibar to N / cm^2
        P_air     = 100 * P_air
        P_air_err = 100 * P_air_err

        t = ((M_air * P_air) / (R * T_air)) * dist
        t_err = np.sqrt((M_air_err * (t / M_air))**2 +
                        (P_air_err * (t / P_air))**2 +
                        (dist_err  * (t /  dist))**2 +
                        (T_air_err * (t / T_air))**2)

        return (t, t_err)

    # get thicknesses corresponding to each pressure
    # units are (g / J) cm (N / cm^2) = g N / cm J = 0.01 * (g / cm^2)
    thicknesses = np.asarray([thickness(P, 5) for P in pressures_stopping])

    # multiply by 100 to get to g / cm^2
    t     = 100 * thicknesses[:,0]
    t_err = 100 * thicknesses[:,1]

    # save having to run calibrate() each time
    if bin_to_energy is None: bin_to_energy = calibrate(use_old=False)

    x0 = [35, 200, 328, 453, 541, 646, 727, 801, 867, 941, 958, 1072]
    x0 = np.array([bin_to_energy(x)[0] for x in x0])

    xlim_offset = bin_to_energy(20)[0]
    xmin = x0 - xlim_offset
    xmax = x0 + xlim_offset

    energy_bins = np.asarray([bin_to_energy(bin) for bin in bins])

    # fit_func = "norm*((eta)*L(x-x0,gamma)+(1-eta)*G(x-x0,sigma))+bg"
    # params   = "norm=10,eta=0.2,gamma=0.05,sigma=0.05,bg=1,x0="
    fit_func = "norm*G(x-x0,sigma)+bg"
    params   = "norm=20,sigma=0.06,bg=1,x0="
    params   = [params + str(x) for x in x0]

    # location of x0 in parameter list
    peak_loc_ind = 3 # 5

    n_fits = len(spectra)

    peak_loc  = np.zeros(n_fits)
    peak_err  = np.zeros(n_fits)
    good_fits = [True] * n_fits
    for i in range(n_fits):
        peak_fit = sm.data.fitter(f=fit_func, p=params[i],
                               g={'G' : _gaussian, 'L' : _lorentzian})
        peak_fit.set_data( xdata=energy_bins[:,0],  ydata=spectra[i].counts(),
                       exdata=energy_bins[:,1], eydata=get_yerr_counts(spectra[i]))
        peak_fit.set(xmin=xmin[i], xmax=xmax[i])
        peak_fit.fit()

        # ignore peaks whose fits don't converge to simplify debugging
        if peak_fit.results[1] is None:
            good_fits[i] = False
            continue

        peak_loc[i] = peak_fit.results[0][peak_loc_ind]
        peak_err[i] = peak_fit.results[1][peak_loc_ind][peak_loc_ind]
        peak_err[i] = np.sqrt(peak_err[i])

    good_fits = np.array(good_fits, dtype=bool)

    t     = t    [good_fits]
    t_err = t_err[good_fits]
    peak_loc = peak_loc[good_fits]
    peak_err = peak_err[good_fits]

    return ((t, peak_loc), (t_err, peak_err))

    # fit the peak locations to find the calibration curve
    stopping_fit = sm.data.fitter(f=f, p=p, plot_guess=False)
    stopping_fit.set_data( xdata=t,      ydata=peak_loc,
                          exdata=t_err, eydata=peak_err)
    stopping_fit.fit()



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
