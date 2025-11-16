import h5py
import numpy as np
import os

# Recursive loader to handle groups and datasets

print("\n")
def recursive_load(h5_obj):
    if isinstance(h5_obj, h5py.Dataset):
        return h5_obj[()]
    elif isinstance(h5_obj, h5py.Group):
        return {key: recursive_load(h5_obj[key]) for key in h5_obj.keys()}
    else:
        return None

# Load a single MAT v7.3+ file
def load_mat_file(filepath):
    mat_data = {}
    with h5py.File(filepath, 'r') as f:
        for key in f.keys():
            mat_data[key] = recursive_load(f[key])
    return mat_data

# Your folder path in Google Drive
folder = '/home/anower/All/Python/Thesis_VS_code/full_epoch/Data/raw_data'

# Load and print data for the first file
filepath = os.path.join(folder, 'music_listening_experiment_s01.mat')
data = load_mat_file(filepath)
print("Keys in MAT file:", data.keys())

# Loop over all 20 files
for i in range(1, 2):
    filename = f'music_listening_experiment_s{i:02d}.mat'
    filepath = os.path.join(folder, filename)

    print(f"\nLoading {filename}...")

    with h5py.File(filepath, 'r') as f:
        print("Keys in file:", list(f.keys()))

        # Load EEG_Songs
        eeg_songs = f['EEG_Songs'][:]
        print(f"EEG_Songs shape: {eeg_songs.shape}")

        # Load other variables
        fs = f['Fs'][()]  # Sampling rate
        ratings = f['song_ratings'][:]

        print(f"Sampling Rate (Fs): {fs}")
        print(f"Song Ratings shape: {ratings.shape}")

# Check data dimensions and structure for the first file


print("\nData structure for the first file:")
for key in data.keys():
    if not key.startswith('__'):  # Skip metadata
        print(f"{key}: {type(data[key])}, shape: {data[key].shape if hasattr(data[key], 'shape') else 'N/A'}")