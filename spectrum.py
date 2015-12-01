import numpy as np

class Spectrum:
    """
    Holds data relating to a spectrum recorded by Maestro
    """
    def __init__(self, data, etime, atime):
        self._data  = data
        self._etime = etime
        self._atime = atime

    def counts(self):
        return self._data

    def count_rates(self):
        return self._data / self._etime

    def absolute_time(self):
        return self._atime

def get_spectrum_chn(fname_in):
    with open (fname_in, "rb") as fin:
        np.fromfile(fin, dtype=np.int16,  count=1) # type
        np.fromfile(fin, dtype=np.uint16, count=2) # unit number, segment number
        ascii_secs = np.fromfile(fin, dtype=np.uint8, count=2)
        np.fromfile(fin, dtype=np.uint32, count=1) # real time (20ms)
        live_time_20ms = np.fromfile(fin, dtype=np.uint32, count=1)
        start_date = np.fromfile(fin, dtype=np.uint8, count=8)
        start_time = np.fromfile(fin, dtype=np.uint8, count=4)
        np.fromfile(fin, dtype=np.uint16, count=1) # channel offset
        n_channels = np.fromfile(fin, dtype=np.uint16, count=1)
        data = np.fromfile(fin, dtype=np.uint32, count=n_channels)

    days  = 10 * start_date[0] + start_date[1]
    hours = 10 * start_time[0] + start_time[1]
    mins  = 10 * start_time[2] + start_time[3]
    secs  = 10 * ascii_secs[0] + ascii_secs[1]

    absolute_time = (24 * 60 * 60 * days) + (60 * 60 * hours) + (60 * mins) + (secs);

    return Spectrum(data, 0.02*live_time_20ms[0], absolute_time)

def add_spectra(spectra, chronological=False):
    counts = [spec.counts() for spec in spectra]
    etimes = [spec._etime   for spec in spectra]

    if chronological: start_time = spectra[0]._atime
    else:             start_time = np.min([spec._atime for spec in spectra], 0)

    return Spectrum(np.sum(counts, 0), np.sum(etimes, 0), start_time)
