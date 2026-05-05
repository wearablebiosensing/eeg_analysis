import h5py

file_path = '/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_EEG/EEG_DATASET/001_NBACK12026.04.14_17.52.36.hdf5'

def inspect_gtec_hdf5():
    with h5py.File(file_path, 'r') as f:
        print("--- Root Level ---")
        for key in f.keys():
            print(f"Key: {key}, Type: {type(f[key])}")
            
        print("\n--- RawData Level ---")
        if 'RawData' in f:
            raw_data = f['RawData']
            for key in raw_data.keys():
                print(f"Key: {key}, Type: {type(raw_data[key])}")
                if isinstance(raw_data[key], h5py.Dataset):
                    print(f"  -> Shape: {raw_data[key].shape}")
            
            print("\n--- Attributes in RawData ---")
            for k, v in raw_data.attrs.items():
                print(f"{k}: {v}")
                
        else:
            print("No 'RawData' found.")

if __name__ == '__main__':
    inspect_gtec_hdf5()
