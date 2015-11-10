import numpy   as np
import spinmob as sm
from spectrum import get_spectrum_chn

pulse_heights_calib = [260, 284, 304, 330, 380, 430]
fnames_calib = ["data/calib" + str(height) + ".Chn" for height in pulse_heights_calib]
spectra_calib = [get_spectrum_chn(fname) for fname in fnames_calib]
