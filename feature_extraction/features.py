from abc import ABC, abstractmethod
import numpy as np
from scipy.signal import welch
from fooof import FOOOF

class FeatureExtractor(ABC):
    """Base class for all feature extractors."""
    @abstractmethod
    def extract(self, signal: np.ndarray, fs: float) -> dict:
        """
        Extract features from a 1D EEG signal.
        
        Args:
            signal: 1D numpy array representing the EEG signal for a single channel.
            fs: Sampling frequency in Hz.
            
        Returns:
            A dictionary containing the extracted features.
        """
        pass

class FOOOFThetaPeakExtractor(FeatureExtractor):
    def __init__(self, segment_length_sec=4, theta_band=(4.0, 8.0), fallback_freq=6.0, freq_range=[1, 40]):
        self.segment_length_sec = segment_length_sec
        self.theta_band = theta_band
        self.fallback_freq = fallback_freq
        self.freq_range = freq_range

    def extract(self, signal: np.ndarray, fs: float) -> dict:
        """Extracts the theta peak frequency using FOOOF, keeping math identical to the original script."""
        segment_samples = int(fs * self.segment_length_sec)
        
        # ======================
        # SEGMENTAZIONE (4 s)
        # ======================
        n_segments = signal.shape[0] // segment_samples
        if n_segments == 0:
            return {"theta_peak_freq": self.fallback_freq}
            
        segments = signal[:n_segments * segment_samples]
        segments = segments.reshape(n_segments, segment_samples)

        # ======================
        # WELCH PSD
        # ======================
        # Risoluzione in frequenza = fs / nperseg = 0.25 Hz
        nperseg = int(fs / 0.25)
        
        psd_list = []
        for seg in segments:
            freqs, psd = welch(
                seg,
                fs=fs,
                window='hamming',
                nperseg=nperseg,
                noverlap=0,
                detrend='constant',
                scaling='density'
            )
            psd_list.append(psd)

        # PSD media sui segmenti
        mean_psd = np.mean(psd_list, axis=0)

        # Handle valid PSDs so FOOOF does not crash on flat/invalid channels (e.g., LABEL, ECG)
        if np.all(mean_psd <= 0) or np.isnan(mean_psd).any() or np.isinf(mean_psd).any():
            return {"theta_peak_freq": self.fallback_freq}

        # ======================
        # FOOOF
        # ======================
        fm = FOOOF(verbose=False)
        try:
            fm.fit(freqs, mean_psd, freq_range=self.freq_range)
        except Exception as e:
            return {"theta_peak_freq": self.fallback_freq}

        # ======================
        # ESTRAZIONE THETA PEAK
        # ======================
        theta_peaks = []
        for peak in fm.peak_params_:
            cf, amp, bw = peak
            if self.theta_band[0] <= cf <= self.theta_band[1]:
                theta_peaks.append((cf, amp))

        if theta_peaks:
            # scegli il picco theta con ampiezza massima
            theta_peak_freq = max(theta_peaks, key=lambda x: x[1])[0]
        else:
            theta_peak_freq = self.fallback_freq

        return {
            "theta_peak_freq": theta_peak_freq
        }
