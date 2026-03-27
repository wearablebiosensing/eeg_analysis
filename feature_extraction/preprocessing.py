import numpy as np
import scipy.signal
# Note: Requires asrpy to be installed.
from asrpy import clean_windows, asr_calibrate, asr_process

def preprocess_eeg(eeg_data: np.ndarray, fs: float, lowcut=1, highcut=45, order=4, asr_cutoff=15) -> np.ndarray:
    """
    Applies the identical bandpass filter and ASR process from the preprocessing script.
    eeg_data is expected to be shape (n_channels, n_samples).
    """
    # ==========================
    # Applico filtro (Bandpass)
    # ==========================
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    eeg_data_filtered = scipy.signal.lfilter(b, a, eeg_data)
    
    # ==========================
    # ASR
    # ==========================
    # (optional) make sure your asr is only fitted to clean parts of the data
    pre_cleaned, _ = clean_windows(eeg_data_filtered, fs, max_bad_chans=0.1)

    # fit the asr-in literature the cutoff is a value between 10 and 30
    M, T = asr_calibrate(pre_cleaned, fs, cutoff=asr_cutoff)
    
    # apply it
    eeg_data_clean = asr_process(eeg_data_filtered, fs, M, T)
    
    return eeg_data_clean
