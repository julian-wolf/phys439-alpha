import numpy as np

class Spectrum:
    """
    Holds data relating to a spectrum recorded by Maestro
    """
    def __init__(self, data, etime):
        self._data  = data
        seft._etime = etime

    def counts(self):
        return self._data

    def count_rates(self):
        return self._data / self._etime

def get_spectrum_chn(fname_in):
    with open (fname_in, "rb") as fin:
        np.fromfile(fin, dtype="np.uint16", count=2)  # unit number, segment number
        np.fromfile(fin, dtype=np.uint8,  count=2)  # ascii seconds
        np.fromfile(fin, dtype=np.uint32, count=1)  # real time (20ms)
        live_time_20ms = np.fromfile(fin, dtype=np.uint32, count=1)
        np.fromfile(fin, dtype=np.uint8,  count=12) # start date, start time
        np.fromfile(fin, dtype=np.uint16, count=1)  # channel offset
        n_channels = np.fromfile(fin, dtype=np.uint16, count=1)
        data = np.fromfile(fin, dtype=np.uint32, count=n_channels)

    return spectrum(data, 0.02*live_time_20ms)
