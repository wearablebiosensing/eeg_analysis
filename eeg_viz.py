import mne
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import filedialog
# python3 pipeline.py \
#   --input "/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_EEG/EEG_DATASET/001_NBACK12026.04.14_17.52.36.hdf5" \
#   --output "/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_EEG/EEG_DATASET/results/001_nback__results.csv"
# -----------------------------
# 1. File Picker
# -----------------------------
root = tk.Tk()
root.withdraw()  # Hide main tkinter window

file_path = filedialog.askopenfilename(
    title="Select EEG file",
    filetypes=[("EEG files", "*.edf *.hdf5 *.h5"), ("All files", "*.*")]
)

if not file_path:
    raise ValueError("No file selected. Script terminated.")

print("Selected file:", file_path)
print("File path check:", os.path.exists(file_path))

# -----------------------------
# 2. Load Data
# -----------------------------
ext = os.path.splitext(file_path)[-1].lower()

if ext == '.edf':
    raw = mne.io.read_raw_edf(file_path, preload=True)
elif ext in ['.hdf5', '.h5']:
    import h5py
    with h5py.File(file_path, 'r') as f:
        # Check for g.tec HDF5 structure
        if 'RawData' in f and 'Samples' in f['RawData']:
            eeg_data = f['RawData']['Samples'][()].T 
            fs = f['RawData'].attrs.get('SamplingFrequency', 250.0)
        else:
            largest_ds = None
            max_size = 0
            def find_largest_dataset(name, obj):
                global largest_ds, max_size
                if isinstance(obj, h5py.Dataset) and obj.size > max_size:
                    largest_ds = obj[()]
                    max_size = obj.size
            f.visititems(find_largest_dataset)
            
            if largest_ds is None:
                raise ValueError("Could not find any datasets in HDF5 file.")
                
            eeg_data = largest_ds
            if eeg_data.shape[0] > eeg_data.shape[1]:
                eeg_data = eeg_data.T
            fs = 250.0
            
    # Create an MNE RawArray
    n_channels = eeg_data.shape[0]
    ch_names = [f"CH_{i}" for i in range(n_channels)]
    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types='eeg')
    raw = mne.io.RawArray(eeg_data, info)
else:
    raise ValueError(f"Unsupported file format: {ext}")

# -----------------------------
# 3. Clean Channel Names
# -----------------------------
raw.rename_channels(lambda x: x.replace('-Av', ''))

# -----------------------------
# 4. Filtering
# -----------------------------
raw.filter(l_freq=0.1, h_freq=40.0)

# Remove US powerline noise
raw.notch_filter(freqs=60)

# -----------------------------
# 5. Set EEG Montage
# -----------------------------
montage = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(montage, on_missing='ignore')

# -----------------------------
# 6. Summary Output
# -----------------------------
print("-" * 40)
print(f"Data Duration: {raw.times[-1]:.2f} seconds")
print(f"Sampling Rate: {raw.info['sfreq']} Hz")
print(f"Channels found: {raw.ch_names}")
print("-" * 40)

# -----------------------------
# 7. Visualization
# -----------------------------
raw.plot(
    n_channels=15,
    duration=10,
    title="EEG Browser - Press '?' for shortcuts",
    show_options=True,
    block=True
)

# -----------------------------
# 8. Optional PSD Plot
# -----------------------------
# raw.compute_psd(fmax=70).plot()
# plt.show(block=True)