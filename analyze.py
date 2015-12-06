# -*- coding: utf-8 -*-

import numpy   as np
import spinmob as sm
from matplotlib import pyplot as plt
from spectrum   import get_spectrum_chn, add_spectra

# bins used by Maestro
bins = np.arange(0, 2048)

# data used for calibration
pulse_heights_calib_old = np.array([260, 284, 304, 330, 430]) # 380 is all weird-looking
pulse_heights_calib_new = np.array([110, 180, 250, 360])      # 320 '' ''   ''  -  ''

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

fnames_Tr  = ["data/long_charge/tr_%03d.Chn" % (n,) for n in range(288)]
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

def calibrate_old(f="a*x+b", p="a,b"):
    """
    Automates calibration of the apparatus;
    returns function which converts bin numbers to energies
    """
    # first-guess locations of peaks
    spectra_pulse = spectra_pulse_calib_old
    spectrum_Am   = spectrum_Am_calib_old
    x0 = np.array([709, 777.3, 837, 914.5, 1190])
    pulse_heights = pulse_heights_calib_old

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

    # peak_err = peak_err * 10

    # peak_loc[2] -= 4
    # peak_loc[3] -= 7

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

    p_peak    = "norm=2400,x0=1300,sigma=18,bg=1"
    xmin_peak = 1295
    xmax_peak = 1320

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

def calibrate_new(f="a*x+b", p="a,b"):
    """
    Automates calibration of the apparatus;
    returns function which converts bin numbers to energies
    """
    spectrum_sum = add_spectra(spectra_Tr[78:223], chronological=True)

    # true peak locations
    x0 = [5.607, 5.769, 6.051, 6.090, 8.794]

    x0_init = [1188, 1224, 1285,  1294, 1882]
    n0_init = [450,  450,  12000, 300,  20000]
    s0_init = [3,    3,    4.5,   1,    5]

    fit_func   = ""
    parameters = ""
    for i in range(len(x0_init)):
        fit_func   += "n"  + str(i) + "*G(x-x" + str(i) + ",s" + str(i) + ")+"
        parameters += "n"  + str(i) + "=" + str(n0_init[i]) + \
                      ",x" + str(i) + "=" + str(x0_init[i]) + \
                      ",s" + str(i) + "=" + str(s0_init[i]) + ","
    fit_func   += "bg"
    parameters += "bg=1"

    e_fit_init = sm.data.fitter(f=fit_func, p=parameters, g={'G' : _gaussian})
    e_fit_init.set_data(xdata=bins, ydata=spectrum_sum.counts(),
                        eydata=get_yerr_counts(spectrum_sum))
    e_fit_init.set(xmin=1170, xmax=1900)
    e_fit_init.fit()

    peak_inds = np.arange(1, 3*len(x0_init), 3)

    x0_found = e_fit_init.results[0][peak_inds]

    calib_fit = sm.data.fitter(f="a*x+b", p="a,b")
    calib_fit.set_data(xdata=x0_found, ydata=x0, eydata=0.01)
    calib_fit.fit()

    slope     = calib_fit.results[0][0]
    intercept = calib_fit.results[0][1]
    slope_err     = np.sqrt(calib_fit.results[1][0][0])
    intercept_err = np.sqrt(calib_fit.results[1][1][1])

    def bin_to_energy(bin):
        energy = slope * bin + intercept
        energy_err = np.sqrt(bin**2 * slope_err**2 + intercept_err**2)
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

def analyze_hl(spectra=spectra_Tr):
    """
    Automates analysis of halflife data
    """
    l1_expected = np.log(2) / (10.64 * 60 * 60) # lambda for Pb212
    l2_expected = np.log(2) / (60.60 * 60)      # lambda for Bi212

    f_activity = "N0*l1*l2*(exp(-l1*x)-exp(-l2*x))/(l2-l1)+N2*l2*exp(-l2*x)"
    p_activity = "N0,l1=%f,l2=%f,N2" % (l1_expected, l2_expected)

    (count_rates, count_errs) = _get_total_count_rates(spectra, 500)
    elapsed_times = _get_total_elapsed_times(spectra) # in seconds

    hl_fit = sm.data.fitter(f=f_activity, p=p_activity)
    hl_fit.set_data(xdata=elapsed_times, ydata=count_rates, eydata=count_errs)
    hl_fit.set(coarsen=3)
    hl_fit.fit()

    # print "n0 = ", hl_fit.results[0][0], " ± ", np.sqrt(hl_fit.results[1][0][0])
    # print "n2 = ", hl_fit.results[0][3], " ± ", np.sqrt(hl_fit.results[1][3][3])

    hl_Pb212 = np.log(2) / hl_fit.results[0][1]
    hl_Bi212 = np.log(2) / hl_fit.results[0][2]

    hl_Pb212_err = np.sqrt(hl_fit.results[1][1][1])*np.log(2) / hl_fit.results[0][1]**2
    hl_Bi212_err = np.sqrt(hl_fit.results[1][2][2])*np.log(2) / hl_fit.results[0][2]**2

    print  ((hl_Pb212     / (60 * 60), hl_Bi212     / (60)),
            (hl_Pb212_err / (60 * 60), hl_Bi212_err / (60)),
            hl_fit.reduced_chi_squareds()[0])

    return hl_fit

# TODO: fudge fit conditions
def _analyze_energy_and_branching(spectra, bin_to_energy):
    """
    Automates analysis of alpha energy and branching ratio data
    """
    spectrum_sum = add_spectra(spectra, chronological=True)

    x0 = [5.607, 5.769, 6.051, 6.090, 8.794]
    n0 = [0.4,   0.7,   12,    1.5,   30]
    s0 = [0.005, 0.005, 0.01,  0.008, 0.01]
    c0 = [0.01,  0.01,  1,     0.001, 1]
    t0 = [10,    10,    10,    10,    10]

    fit_func   = ""
    parameters = ""
    for i in range(len(x0)):
        fit_func   += "n" + str(i) + "*G(x-x" + str(i) + ",s" + str(i) + ")*(1+" + \
                      "c" + str(i) + "*exp(abs(t" + str(i) + ")*(x" + str(i) + "-x)))+"
        parameters += "n"  + str(i) + "=" + str(n0[i]) + \
                      ",x" + str(i) + "=" + str(x0[i]) + \
                      ",s" + str(i) + "=" + str(s0[i]) + \
                      ",c" + str(i) + "=" + str(c0[i]) + \
                      ",t" + str(i) + "=" + str(t0[i]) + ","
    fit_func   += "bg"
    parameters += "bg=6"

    # save having to run calibrate() each time
    if bin_to_energy is None: bin_to_energy = calibrate_new()

    energy_bins = np.asarray([bin_to_energy(bin) for bin in bins])

    xmin = bin_to_energy(1170)[0]
    xmax = bin_to_energy(1900)[0]

    e_fit = sm.data.fitter(f=fit_func, p=parameters, g={'G' : _gaussian})
    e_fit.set_data( xdata=energy_bins[:,0],  ydata=spectrum_sum.counts(),
                   exdata=energy_bins[:,1], eydata=get_yerr_counts(spectrum_sum))
    e_fit.set(xmin=xmin, xmax=xmax)
    e_fit.fit()

    ndof_per_peak = 5
    n0_inds = ndof_per_peak * np.arange(len(x0))
    x0_inds = n0_inds + 1
    s0_inds = n0_inds + 2

    result_vals =                     e_fit.results[0]
    result_errs = np.sqrt(np.diagonal(e_fit.results[1]))

    # for finding the peak locations
    x0_vals = result_vals[x0_inds]
    x0_errs = result_errs[x0_inds]

    peak_locs = (x0_vals, x0_errs)

    # for finding the peak areas
    n0_vals = result_vals[n0_inds]
    n0_errs = result_errs[n0_inds]

    peak_areas = (n0_vals, n0_errs)

    # might want peak widths for something
    s0_vals = result_vals[s0_inds]
    s0_errs = result_errs[s0_inds]

    return (peak_locs, peak_areas)

def analyze_energy(spectra=spectra_Tr[78:223], bin_to_energy=None):
    (peak_locs, _) = _analyze_energy_and_branching(spectra, bin_to_energy)
    return peak_locs

# TODO: account for errors
def analyze_branching(spectra=spectra_Tr[78:223], bin_to_energy=None):
    (_, (n0_vals, n0_errs)) = _analyze_energy_and_branching(spectra, bin_to_energy)
    total_areas   = np.sum(n0_vals)
    partial_areas = np.sum(n0_vals[:-1])

    print n0_vals
    print n0_errs

    total_ratios   = np.array([    partial_areas / total_areas,
                               1 - partial_areas / total_areas])
    partial_ratios = np.array(n0_vals[:-1]) / partial_areas

    total_area_errs   = np.sqrt(np.sum(n0_errs**2))
    partial_area_errs = np.sqrt(np.sum(n0_errs[:-1]**2))

    total_errs   = np.sqrt(np.array(
                    [(partial_area_errs / total_areas)**2 +
                     (partial_areas * total_area_errs / total_areas**2)**2] * 2))
    partial_errs = np.sqrt(np.array(
                    (n0_errs[:-1] / partial_areas)**2 +
                    (partial_area_errs * n0_vals[:-1] / partial_areas**2)**2))

    return ((total_ratios, total_errs), (partial_ratios, partial_errs))

# TODO: don't forget to remove indices argument before running actual analysis
def analyze_stopping(indices, spectra=spectra_stopping, bin_to_energy=None):
    """
    Automates analysis of stopping power (variable pressure) data
    """
    M_air = 0.2896 # kg / mol
    T_air = 294    # K
    R     = 8.314  # J / mol K    (negligible error)
    dist  = 0.0485 # m

    M_air_err = 0.0001
    T_air_err = 2
    dist_err  = 0.0003

    def get_thickness(P_air, P_air_err):
        # first, convert pressures from millibar to N / m^2
        P_air     = 100 * P_air
        P_air_err = 100 * P_air_err

        t = ((M_air * P_air) / (R * T_air)) * dist
        t_err = np.sqrt((M_air_err * (t / M_air))**2 +
                        (P_air_err * (t / P_air))**2 +
                        (dist_err  * (t /  dist))**2 +
                        (T_air_err * (t / T_air))**2)

        return (t, t_err)

    # rough error in pressure measurements
    P_air_err = 5

    # get thicknesses corresponding to each pressure
    # units are (kg / J) m (N / m^2) = kg N / m J = kg / m^2
    thicknesses = np.asarray([get_thickness(P, P_air_err) for P in pressures_stopping])

    thickness     = thicknesses[:,0]
    thickness_err = thicknesses[:,1]

    #     0     1     2     3     4     5     6     7     8     9     10    11
    #     c2    x     x     x     x     x     x     x     x     x     x     x
    x0 = [34,   204,  331,  455,  545,  646,  729,  802,  868,  942,  959,  1074]
    n0 = [15,   13,   11,   12,   11,   12,   11,   12,   14,   11,   12,   10]
    s0 = [0.09, 0.09, 0.07, 0.06, 0.05, 0.05, 0.04, 0.04, 0.04, 0.03, 0.03, 0.025]
    # t0 = [2.4,  12,   23,   23,   32,   28,   40,   37,   32,   51,   48,   70]
    t0 = [2.4,  -20,    2,    3,    2,    8,    0,    7,    2,    1,    8,    0]

    # save having to run calibrate() each time
    if bin_to_energy is None: bin_to_energy = calibrate_old()

    x0 = np.array([bin_to_energy(x)[0] for x in x0])

    # xmin_offset = bin_to_energy(18)[0]
    # xmax_offset = bin_to_energy(50)[0]
    # xmin = x0 - xmin_offset
    # xmax = x0 + xmax_offset

    # xmin[0] = bin_to_energy(7)[0] # account for end of spectrum

    # fit_func = "norm*G(x-x0,sigma)*(1+c*exp(abs(t)*(x0-x)))+bg"
    # params   = ["norm=%d,sigma=%f,c=0.008,t=%f,bg=0,x0=%f" % (n, s, t, x)
    #             for (x, n, s, t) in zip(x0, n0, s0, t0)]

    # energy_bins = np.asarray([bin_to_energy(bin) for bin in bins])

    # # location of x0 in parameter list
    # peak_loc_ind = 5

    # n_fits = len(spectra)

    # peak_loc  = np.zeros(n_fits)
    # peak_err  = np.zeros(n_fits)
    # good_fits = [True] * n_fits
    # # for i in range(n_fits):
    # for i in indices:
    #     peak_fit = sm.data.fitter(f=fit_func, p=params[i], g={'G' : _gaussian})
    #     peak_fit.set_data( xdata=energy_bins[:,0],  ydata=spectra[i].counts(),
    #                       exdata=energy_bins[:,1], eydata=get_yerr_counts(spectra[i]))
    #     peak_fit.set(xmin=xmin[i], xmax=xmax[i], coarsen=2)
    #     peak_fit.fit()

    #     # ignore peaks whose fits don't converge to simplify debugging
    #     if peak_fit.results[1] is None:
    #         good_fits[i] = False
    #         continue

    #     print peak_fit

    #     peak_loc[i] = peak_fit.results[0][peak_loc_ind]
    #     peak_err[i] = peak_fit.results[1][peak_loc_ind][peak_loc_ind]
    #     peak_err[i] = np.sqrt(peak_err[i])

    # good_fits = np.array(good_fits, dtype=bool)

    # thickness     = thickness    [good_fits]
    # thickness_err = thickness_err[good_fits]
    # peak_loc = peak_loc[good_fits]
    # peak_err = peak_err[good_fits]

    # temporary guesses
    peak_loc = x0
    peak_err = 0 * x0 + 0.1

    dE =  peak_loc[1:] -  peak_loc[:-1] # in MeV
    dt = thickness[1:] - thickness[:-1] # in kg / m^2

    dEdt = dE / dt
    dEdt_err = np.sqrt((     peak_err[1:]**2 +      peak_err[:-1]**2) * (dEdt / dE)**2 +
                       (thickness_err[1:]**2 + thickness_err[:-1]**2) * (dEdt / dt)**2)

    t_mean = (thickness[1:] + thickness[:-1]) / 2
    t_err = np.sqrt((thickness_err[1:]**2 + thickness_err[:-1]**2)) / 2

    S = -t_mean * dEdt / dist # in MeV / m
    S_err = np.sqrt((   t_err * (S / t_mean))**2 +
                    (dEdt_err * (S /   dEdt))**2 +
                    (dist_err * (S /   dist))**2)

    P_mean = (pressures_stopping[1:] + pressures_stopping[:-1]) / 2;

    # convert to MeV / cm
    return (P_mean, 0.01 * S, 0.01 * S_err / 3)

def print_data_to_columns(sm_fit, fname, residuals=False):
    xmin = sm_fit._settings['xmin']
    xmax = sm_fit._settings['xmax']

    xdata  = sm_fit._xdata_massaged[0]
    i_used = (xdata >= xmin) & (xdata <= xmax)
    xdata  = xdata[i_used]

    if not residuals:
        ydata  = sm_fit._ydata_massaged[0]
        eydata = sm_fit._eydata_massaged[0]
    else:
        ydata  = sm_fit.studentized_residuals()[0]
        eydata = (0 * ydata) + 1

    with open(fname, 'w') as f_out:
        n_data = len(xdata);
        for i in range(n_data):
            entry = "%f\t%f\t%f\n" % (xdata[i], ydata[i], eydata[i])
            f_out.write(entry)

    return
