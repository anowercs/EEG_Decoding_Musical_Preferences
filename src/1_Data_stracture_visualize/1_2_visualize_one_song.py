import h5py
import numpy as np
import os
import matplotlib
matplotlib.use('TkAgg')  # Set backend to TkAgg
import matplotlib.pyplot as plt

# File path
file_path = '/home/anower/All/Python/Thesis_VS_code/full_epoch/Data/raw_data/music_listening_experiment_s01.mat'

# Load data
with h5py.File(file_path, 'r') as f:
    eeg_songs = np.array(f['EEG_Songs'])          # shape: (10240, 14, 30)
    fs = int(np.array(f['Fs'])[0][0])             # scalar sampling rate

# Transpose and select first 10 seconds of song 0
eeg_songs = np.transpose(eeg_songs, (2, 1, 0))     # (30 songs, 14 channels, 10240 samples)
eeg_song = eeg_songs[0, :, :1280]                  # First 10 seconds of song 0

def visualize_eeg_song(eeg_data, sampling_rate=128):
    n_channels = eeg_data.shape[0]
    time = np.arange(eeg_data.shape[1]) / sampling_rate

    fig, axes = plt.subplots(n_channels, 1, figsize=(12, 2*n_channels), sharex=True)
    if n_channels == 1:
        axes = [axes]

    for i in range(n_channels):
        axes[i].plot(time, eeg_data[i])
        axes[i].set_ylabel(f'Ch {i+1}')
        axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    plt.suptitle("EEG Song 0 – First 10 Seconds")
    plt.tight_layout()

    # Save the plot to a file (optional, to avoid GUI issues)
    plt.savefig('/home/anower/All/Python/Thesis_VS_code/full_epoch/Figure/10_sec_eeg_song_visualize.png')
    print("Plot saved as '10_sec_eeg_song_visualize.png'")

    # Show the plot (comment out if using Agg backend)
    plt.show()

# Call the visualization function
visualize_eeg_song(eeg_song, sampling_rate=fs)