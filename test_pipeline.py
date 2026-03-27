import numpy as np
import pandas as pd
from feature_extraction.pipeline import run_pipeline

def test():
    # Create fake EEG data: 3 channels, 30 seconds at 250Hz = 7500 samples
    fs = 250.0
    n_channels = 3
    n_samples = int(fs * 30)
    
    # Generate random data with some theta oscillation
    t = np.arange(n_samples) / fs
    theta_wave = 1.5 * np.sin(2 * np.pi * 6 * t) # 6 Hz
    
    eeg_data = np.random.randn(n_channels, n_samples)
    for i in range(n_channels):
        eeg_data[i, :] += theta_wave
        
    channel_names = [f"EEG_CH_{i}" for i in range(n_channels)]
    metadata = {"subject": "Test_Subject", "condition": "Baseline"}
    
    print("Running pipeline on dummy data...")
    try:
        df = run_pipeline(eeg_data, fs, channel_names, metadata, apply_preprocess=True)
        print("Pipeline execution successful. Results:")
        print(df)
        print("Saving to test_output.csv")
        df.to_csv("test_output.csv", index=False)
        return True
    except Exception as e:
        print(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test()
