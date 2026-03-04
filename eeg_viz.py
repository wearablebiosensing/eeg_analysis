import mne
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import filedialog

# -----------------------------
# 1. File Picker
# -----------------------------
root = tk.Tk()
root.withdraw()  # Hide main tkinter window

file_path = filedialog.askopenfilename(
    title="Select EEG EDF file",
    filetypes=[("EDF files", "*.edf"), ("All files", "*.*")]
)

if not file_path:
    raise ValueError("No file selected. Script terminated.")

print("Selected file:", file_path)
print("File path check:", os.path.exists(file_path))

# -----------------------------
# 2. Load EDF
# -----------------------------
raw = mne.io.read_raw_edf(file_path, preload=True)

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