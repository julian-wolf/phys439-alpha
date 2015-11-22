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
        start_day  = np.fromfile(fin, dtype=np.uint8, count=2)
        start_date = np.fromfile(fin, dtype=np.uint8, count=6)
        start_time = np.fromfile(fin, dtype=np.uint8, count=4)
        np.fromfile(fin, dtype=np.uint16, count=1) # channel offset
        n_channels = np.fromfile(fin, dtype=np.uint16, count=1)
        data = np.fromfile(fin, dtype=np.uint32, count=n_channels)

    days  = 10 * start_day[0]  + start_day[1]
    hours = 10 * start_time[0] + start_time[1]
    mins  = 10 * start_time[2] + start_time[3]
    secs  = 10 * ascii_secs[0] + ascii_secs[1]

    absolute_time = (24 * 60 * 60 * days) + (60 * 60 * hours) + (60 * mins) + (secs);

    return Spectrum(data, 0.02*live_time_20ms[0], absolute_time)
