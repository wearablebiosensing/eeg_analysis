import os
import argparse
import numpy as np
import pandas as pd
import mne
import scipy.io

from config import *
from preprocessing import preprocess_eeg
from features import FOOOFThetaPeakExtractor

def load_data(file_path: str):
    """
    Loads EEG data from .edf or .mat file.
    Returns (eeg_cdata, fs, channel_names)
    """
    ext = os.path.splitext(file_path)[-1].lower()
    
    if ext == '.edf':
        data = mne.io.read_raw_edf(file_path, preload=True)
        eeg_data = data.get_data()
        fs = data.info['sfreq']
        channel_names = data.ch_names
    elif ext == '.mat':
        mat = scipy.io.loadmat(file_path)
        # We need to find the correct data key. Assume 'X' or 'raw_data'
        if 'X' in mat:
            eeg_data = mat['X']
        elif 'raw_data' in mat:
            eeg_data = mat['raw_data']
        else:
            # Fallback: find any array that looks like EEG data
            possible_keys = [k for k in mat.keys() if not k.startswith('__')]
            if possible_keys:
                eeg_data = mat[possible_keys[0]]
            else:
                raise ValueError("Could not find EEG data in MAT file.")
        
        # When loading MAT, we might not have fs and ch_names stored natively.
        # Fallback to defaults or parameters
        fs = 512.0 # Default fallback
        channel_names = [f"CH_{i}" for i in range(eeg_data.shape[0])]
    else:
        raise ValueError(f"Unsupported file format: {ext}")
        
    return eeg_data, fs, channel_names

def run_pipeline(eeg_data: np.ndarray, fs: float, channel_names: list, metadata: dict, apply_preprocess=True):
    """
    Runs the modular feature extraction pipeline on the provided EEG data.
    """
    # 1. Preprocessing
    if apply_preprocess:
        print("Applying preprocessing (Bandpass Filter & ASR)...")
        eeg_data = preprocess_eeg(
            eeg_data, 
            fs=fs, 
            lowcut=PREPROC_LOWCUT, 
            highcut=PREPROC_HIGHCUT, 
            order=PREPROC_ORDER, 
            asr_cutoff=PREPROC_ASR_CUTOFF
        )
        
    # 2. Setup Extractors
    extractors = {
        "fooof_theta": FOOOFThetaPeakExtractor(
            segment_length_sec=FOOOF_SEGMENT_LENGTH_SEC,
            theta_band=FOOOF_THETA_BAND,
            fallback_freq=FOOOF_FALLBACK_FREQ,
            freq_range=FOOOF_FREQ_RANGE
        )
    }
    
    # 3. Process Channels
    results = []
    for ch_idx, ch_name in enumerate(channel_names):
        print(f"Processing channel: {ch_name} ({ch_idx+1}/{len(channel_names)})")
        signal = eeg_data[ch_idx, :]
        
        row_data = {
            "channel": ch_name,
            **metadata
        }
        
        # Apply all extractors
        for ext_name, extractor in extractors.items():
            feats = extractor.extract(signal, fs)
            # Prefix column names with extractor name to avoid conflicts
            for f_name, f_val in feats.items():
                row_data[f"{ext_name}_{f_name}"] = f_val
                
        results.append(row_data)
        
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(description="Modular EEG Feature Extraction Pipeline")
    parser.add_argument("--input", type=str, required=True, help="Path to input .edf or .mat file")
    parser.add_argument("--output", type=str, default="features_extracted.csv", help="Path to output .csv file")
    parser.add_argument("--subject", type=str, default="Unknown", help="Subject ID")
    parser.add_argument("--condition", type=str, default="Unknown", help="Experimental condition")
    parser.add_argument("--no_preprocess", action="store_true", help="Skip preprocessing (Bandpass + ASR)")
    
    args = parser.parse_args()
    
    print(f"Loading data from {args.input}...")
    eeg_data, fs, channel_names = load_data(args.input)
    print(f"Loaded {eeg_data.shape[0]} channels, {eeg_data.shape[1]} samples at {fs} Hz.")
    
    metadata = {
        "subject": args.subject,
        "condition": args.condition,
        "fs": fs
    }
    
    df = run_pipeline(eeg_data, fs, channel_names, metadata, apply_preprocess=not args.no_preprocess)
    
    output_path = args.output
    if os.path.isdir(output_path):
        filename = f"{os.path.splitext(os.path.basename(args.input))[0]}_features.csv"
        output_path = os.path.join(output_path, filename)
        
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
    df.to_csv(output_path, index=False)
    print(f"Feature extraction complete. Results saved to {output_path}")

if __name__ == "__main__":
    main()
