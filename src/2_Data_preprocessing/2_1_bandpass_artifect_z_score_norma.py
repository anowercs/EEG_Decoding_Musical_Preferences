import h5py
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Set backend to TkAgg
import matplotlib.pyplot as plt
from scipy import signal, stats
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

# Recursive loader to handle groups and datasets
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

# Bandpass filter function
def bandpass_filter(data, low_freq=1, high_freq=40, sampling_rate=128):
    """
    Apply bandpass filter to EEG data
    Handles both 2D (channels x time) and 3D (trials x channels x time) data
    """
    nyquist = sampling_rate / 2
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    if data.ndim == 2:
        time_axis = 1
    elif data.ndim == 3:
        time_axis = 2
    else:
        raise ValueError(f"Unsupported data dimensions: {data.shape}")
    filtered_data = signal.filtfilt(b, a, data, axis=time_axis)
    return filtered_data

# Visualize bandpass filter effects
def visualize_filter_effects(subject_id, original_data, filtered_data, sampling_rate=128,
                           channel_idx=0, trial_idx=0, save_path='/home/anower/All/Python/Thesis_VS_code/full_epoch/2_Data_preprocessing/Figure/',
                           show_plot=False):
    """
    Visualize the effects of bandpass filtering on EEG data
    """
    os.makedirs(save_path, exist_ok=True)

    original_signal = original_data[trial_idx, channel_idx, :]
    filtered_signal = filtered_data[trial_idx, channel_idx, :]
    difference_signal = original_signal - filtered_signal

    time_samples = original_signal.shape[0]
    time_vector = np.arange(time_samples) / sampling_rate
    freqs = np.fft.fftfreq(time_samples, 1/sampling_rate)[:time_samples//2]
    original_fft = np.abs(np.fft.fft(original_signal))[:time_samples//2]
    filtered_fft = np.abs(np.fft.fft(filtered_signal))[:time_samples//2]
    difference_fft = np.abs(np.fft.fft(difference_signal))[:time_samples//2]

    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(f'Bandpass Filter Effects - {subject_id}, Channel {channel_idx}, Trial {trial_idx}',
                 fontsize=16, fontweight='bold')

    # Time domain plots
    axes[0, 0].plot(time_vector, original_signal, 'b-', linewidth=0.8, alpha=0.8)
    axes[0, 0].set_title('Original EEG Signal', fontweight='bold')
    axes[0, 0].set_xlabel('Time (seconds)')
    axes[0, 0].set_ylabel('Amplitude (μV)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim([0, min(10, time_vector[-1])])

    axes[1, 0].plot(time_vector, filtered_signal, 'g-', linewidth=0.8, alpha=0.8)
    axes[1, 0].set_title('Filtered EEG Signal (1-40 Hz)', fontweight='bold')
    axes[1, 0].set_xlabel('Time (seconds)')
    axes[1, 0].set_ylabel('Amplitude (μV)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim([0, min(10, time_vector[-1])])

    axes[2, 0].plot(time_vector, difference_signal, 'r-', linewidth=0.8, alpha=0.8)
    axes[2, 0].set_title('Removed Components (Original - Filtered)', fontweight='bold')
    axes[2, 0].set_xlabel('Time (seconds)')
    axes[2, 0].set_ylabel('Amplitude (μV)')
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].set_xlim([0, min(10, time_vector[-1])])

    # Frequency domain plots
    axes[0, 1].semilogy(freqs, original_fft, 'b-', linewidth=1, alpha=0.8)
    axes[0, 1].axvline(x=1, color='red', linestyle='--', alpha=0.7, label='1 Hz cutoff')
    axes[0, 1].axvline(x=40, color='red', linestyle='--', alpha=0.7, label='40 Hz cutoff')
    axes[0, 1].set_title('Original Signal Spectrum', fontweight='bold')
    axes[0, 1].set_xlabel('Frequency (Hz)')
    axes[0, 1].set_ylabel('Magnitude')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim([0, 60])
    axes[0, 1].legend()

    axes[1, 1].semilogy(freqs, filtered_fft, 'g-', linewidth=1, alpha=0.8)
    axes[1, 1].axvline(x=1, color='red', linestyle='--', alpha=0.7, label='1 Hz cutoff')
    axes[1, 1].axvline(x=40, color='red', linestyle='--', alpha=0.7, label='40 Hz cutoff')
    axes[1, 1].set_title('Filtered Signal Spectrum', fontweight='bold')
    axes[1, 1].set_xlabel('Frequency (Hz)')
    axes[1, 1].set_ylabel('Magnitude')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim([0, 60])
    axes[1, 1].legend()

    axes[2, 1].semilogy(freqs, difference_fft, 'r-', linewidth=1, alpha=0.8)
    axes[2, 1].axvline(x=1, color='red', linestyle='--', alpha=0.7, label='1 Hz cutoff')
    axes[2, 1].axvline(x=40, color='red', linestyle='--', alpha=0.7, label='40 Hz cutoff')
    axes[2, 1].set_title('Removed Components Spectrum', fontweight='bold')
    axes[2, 1].set_xlabel('Frequency (Hz)')
    axes[2, 1].set_ylabel('Magnitude')
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].set_xlim([0, 60])
    axes[2, 1].legend()

    plt.tight_layout()
    output_file = os.path.join(save_path, f'2.1_eeg_bandpass_filter_effects_{subject_id}_ch{channel_idx}_trial{trial_idx}.png')
    plt.savefig(output_file)
    print(f"Filter effects plot saved as '{output_file}'")

    if show_plot:
        plt.show(block=False)  # Non-blocking display
        plt.pause(0.1)  # Brief pause to ensure rendering
    else:
        plt.close(fig)  # Close figure to prevent display and save memory

    print(f"\n=== Filter Effects Summary ===")
    print(f"Subject: {subject_id}, Channel: {channel_idx}, Trial: {trial_idx}")
    print(f"Sampling Rate: {sampling_rate} Hz")
    print(f"Signal Duration: {time_vector[-1]:.2f} seconds")
    print(f"Original Signal - RMS: {np.sqrt(np.mean(original_signal**2)):.3f} μV")
    print(f"Filtered Signal - RMS: {np.sqrt(np.mean(filtered_signal**2)):.3f} μV")
    print(f"Removed Components - RMS: {np.sqrt(np.mean(difference_signal**2)):.3f} μV")

    def calculate_band_power(fft_data, freqs, low_freq, high_freq):
        band_mask = (freqs >= low_freq) & (freqs <= high_freq)
        return np.sum(fft_data[band_mask]**2)

    total_power_orig = np.sum(original_fft**2)
    delta_power = calculate_band_power(original_fft, freqs, 0.5, 4) / total_power_orig * 100
    theta_power = calculate_band_power(original_fft, freqs, 4, 8) / total_power_orig * 100
    alpha_power = calculate_band_power(original_fft, freqs, 8, 13) / total_power_orig * 100
    beta_power = calculate_band_power(original_fft, freqs, 13, 30) / total_power_orig * 100
    gamma_power = calculate_band_power(original_fft, freqs, 30, 100) / total_power_orig * 100

    print(f"\n=== Power Distribution (Original Signal) ===")
    print(f"Delta (0.5-4 Hz): {delta_power:.1f}%")
    print(f"Theta (4-8 Hz): {theta_power:.1f}%")
    print(f"Alpha (8-13 Hz): {alpha_power:.1f}%")
    print(f"Beta (13-30 Hz): {beta_power:.1f}%")
    print(f"Gamma (30-100 Hz): {gamma_power:.1f}%")

# Artifact detection function
def detect_artifacts(data, amplitude_threshold=100, variance_threshold=5,
                    flat_threshold=0.1, sampling_rate=128):
    """
    Detect artifacts in EEG data using multiple criteria
    """
    n_trials, n_channels, n_timepoints = data.shape

    bad_trials = np.zeros(n_trials, dtype=bool)
    bad_channels = np.zeros(n_channels, dtype=bool)

    artifact_stats = {
        'amplitude_violations': [],
        'variance_violations': [],
        'flat_channels': [],
        'total_bad_trials': 0,
        'total_bad_channels': 0
    }

    print("=== Artifact Detection ===")

    print(f"1. Checking amplitude threshold (±{amplitude_threshold} μV)...")
    for trial in range(n_trials):
        max_amplitude = np.max(np.abs(data[trial, :, :]))
        if max_amplitude > amplitude_threshold:
            bad_trials[trial] = True
            artifact_stats['amplitude_violations'].append((trial, max_amplitude))

    print("2. Checking variance outliers...")
    for channel in range(n_channels):
        trial_variances = np.var(data[:, channel, :], axis=1)
        if np.std(trial_variances) > 0:
            variance_z_scores = np.abs(stats.zscore(trial_variances))
            high_variance_trials = variance_z_scores > variance_threshold
            bad_trials[high_variance_trials] = True
            if np.any(high_variance_trials):
                artifact_stats['variance_violations'].extend([
                    (trial, variance_z_scores[trial])
                    for trial in np.where(high_variance_trials)[0]
                ])

    print("3. Checking for flat channels...")
    for channel in range(n_channels):
        channel_variance = np.var(data[:, channel, :])
        if channel_variance < flat_threshold:
            bad_channels[channel] = True
            artifact_stats['flat_channels'].append((channel, channel_variance))

    print("4. Checking for muscle artifacts (high frequency content)...")
    for trial in range(n_trials):
        for channel in range(n_channels):
            signal_data = data[trial, channel, :]
            fft_signal = np.fft.fft(signal_data)
            freqs = np.fft.fftfreq(len(signal_data), 1/sampling_rate)
            muscle_band = (freqs >= 50) & (freqs <= 64)
            muscle_power = np.sum(np.abs(fft_signal[muscle_band])**2)
            total_power = np.sum(np.abs(fft_signal)**2)
            if muscle_power / total_power > 0.2:
                bad_trials[trial] = True

    artifact_stats['total_bad_trials'] = np.sum(bad_trials)
    artifact_stats['total_bad_channels'] = np.sum(bad_channels)

    print(f"\nArtifact Detection Results:")
    print(f"- Bad trials: {artifact_stats['total_bad_trials']}/{n_trials}")
    print(f"- Bad channels: {artifact_stats['total_bad_channels']}/{n_channels}")
    print(f"- Amplitude violations: {len(artifact_stats['amplitude_violations'])}")
    print(f"- Variance violations: {len(artifact_stats['variance_violations'])}")
    print(f"- Flat channels: {len(artifact_stats['flat_channels'])}")

    return bad_trials, bad_channels, artifact_stats

# Artifact removal function
def remove_artifacts(data, bad_trials, bad_channels, method='remove'):
    """
    Remove or interpolate artifacts from EEG data
    """
    print(f"\n=== Artifact Removal (method: {method}) ===")

    if method == 'remove':
        good_trials = ~bad_trials
        good_channels = ~bad_channels
        clean_data = data[good_trials, :, :][:, good_channels, :]
        print(f"Removed {np.sum(bad_trials)} bad trials")
        print(f"Removed {np.sum(bad_channels)} bad channels")
        channel_map = np.where(good_channels)[0]
    elif method == 'interpolate':
        clean_data = data.copy()
        good_trials = ~bad_trials
        clean_data = clean_data[good_trials, :, :]
        for bad_ch in np.where(bad_channels)[0]:
            good_channels_idx = np.where(~bad_channels)[0]
            if len(good_channels_idx) > 0:
                clean_data[:, bad_ch, :] = np.mean(clean_data[:, good_channels_idx, :], axis=1)
                print(f"Interpolated channel {bad_ch}")
        channel_map = np.arangeovanje(data.shape[1])
        print(f"Removed {np.sum(bad_trials)} bad trials")
        print(f"Interpolated {np.sum(bad_channels)} bad channels")

    print(f"Clean data shape: {clean_data.shape}")
    return clean_data, channel_map

# Normalization function
def normalize_signals(data, method='z_score', axis='channel'):
    """
    Normalize EEG signals
    """
    print(f"\n=== Signal Normalization (method: {method}, axis: {axis}) ===")

    normalized_data = data.copy()
    normalization_params = {}

    if method == 'z_score':
        if axis == 'global':
            mean_val = np.mean(data)
            std_val = np.std(data)
            normalized_data = (data - mean_val) / std_val
            normalization_params = {'mean': mean_val, 'std': std_val}
        elif axis == 'channel':
            means = np.mean(data, axis=(0, 2), keepdims=True)
            stds = np.std(data, axis=(0, 2), keepdims=True)
            normalized_data = (data - means) / stds
            normalization_params = {'means': means, 'stds': stds}
        elif axis == 'trial':
            means = np.mean(data, axis=(1, 2), keepdims=True)
            stds = np.std(data, axis=(1, 2), keepdims=True)
            normalized_data = (data - means) / stds
            normalization_params = {'means': means, 'stds': stds}
    elif method == 'minmax':
        if axis == 'global':
            min_val = np.min(data)
            max_val = np.max(data)
            normalized_data = (data - min_val) / (max_val - min_val)
            normalization_params = {'min': min_val, 'max': max_val}
        elif axis == 'channel':
            mins = np.min(data, axis=(0, 2), keepdims=True)
            maxs = np.max(data, axis=(0, 2), keepdims=True)
            normalized_data = (data - mins) / (maxs - mins)
            normalization_params = {'mins': mins, 'maxs': maxs}
    elif method == 'robust':
        if axis == 'global':
            median_val = np.median(data)
            q75, q25 = np.percentile(data, [75, 25])
            iqr = q75 - q25
            normalized_data = (data - median_val) / iqr
            normalization_params = {'median': median_val, 'iqr': iqr}
        elif axis == 'channel':
            medians = np.median(data, axis=(0, 2), keepdims=True)
            q75s = np.percentile(data, 75, axis=(0, 2), keepdims=True)
            q25s = np.percentile(data, 25, axis=(0, 2), keepdims=True)
            iqrs = q75s - q25s
            normalized_data = (data - medians) / iqrs
            normalization_params = {'medians': medians, 'iqrs': iqrs}

    print(f"Original data range: [{np.min(data):.3f}, {np.max(data):.3f}]")
    print(f"Normalized data range: [{np.min(normalized_data):.3f}, {np.max(normalized_data):.3f}]")
    print(f"Normalized data mean: {np.mean(normalized_data):.3f}")
    print(f"Normalized data std: {np.std(normalized_data):.3f}")

    return normalized_data, normalization_params

# Visualization function for preprocessing effects
def visualize_preprocessing_effects(original_data, clean_data, normalized_data,
                                  subject_id, channel_idx=0, trial_idx=0,
                                  save_path='/home/anower/All/Python/Thesis_VS_code/full_epoch/2_Data_preprocessing/Figure/',
                                  show_plot=False):
    """
    Visualize the effects of artifact removal and normalization
    """
    os.makedirs(save_path, exist_ok=True)

    if trial_idx >= clean_data.shape[0]:
        trial_idx = 0
    if trial_idx >= original_data.shape[0]:
        trial_idx = 0
    if channel_idx >= clean_data.shape[1]:
        channel_idx = 0
    if channel_idx >= original_data.shape[1]:
        channel_idx = 0

    original_signal = original_data[trial_idx, channel_idx, :]
    clean_signal = clean_data[trial_idx, channel_idx, :]
    normalized_signal = normalized_data[trial_idx, channel_idx, :]

    time_vector = np.arange(len(original_signal)) / 128.0

    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    fig.suptitle(f'Preprocessing Effects - {subject_id}, Channel {channel_idx}, Trial {trial_idx}',
                 fontsize=16, fontweight='bold')

    axes[0].plot(time_vector, original_signal, 'b-', linewidth=0.8, alpha=0.8)
    axes[0].set_title('1. Original Filtered Signal', fontweight='bold')
    axes[0].set_ylabel('Amplitude (μV)')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, min(10, time_vector[-1])])

    axes[1].plot(time_vector, clean_signal, 'g-', linewidth=0.8, alpha=0.8)
    axes[1].set_title('2. Clean Signal (After Artifact Removal)', fontweight='bold')
    axes[1].set_ylabel('Amplitude (μV)')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([0, min(10, time_vector[-1])])

    axes[2].plot(time_vector, normalized_signal, 'r-', linewidth=0.8, alpha=0.8)
    axes[2].set_title('3. Normalized Signal (Z-scored)', fontweight='bold')
    axes[2].set_xlabel('Time (seconds)')
    axes[2].set_ylabel('Z-score')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim([0, min(10, time_vector[-1])])

    plt.tight_layout()
    output_file = os.path.join(save_path, f'2.1_eeg_preprocessing_effects_{subject_id}_ch{channel_idx}_trial{trial_idx}.png')
    plt.savefig(output_file)
    print(f"Preprocessing effects plot saved as '{output_file}'")

    if show_plot:
        plt.show(block=False)  # Non-blocking display
        plt.pause(0.1)  # Brief pause to ensure rendering
    else:
        plt.close(fig)  # Close figure to prevent display and save memory

    print(f"\n=== Signal Statistics Comparison ===")
    print(f"Original Signal  - Mean: {np.mean(original_signal):.3f}, Std: {np.std(original_signal):.3f}")
    print(f"Clean Signal     - Mean: {np.mean(clean_signal):.3f}, Std: {np.std(clean_signal):.3f}")
    print(f"Normalized Signal- Mean: {np.mean(normalized_signal):.3f}, Std: {np.std(normalized_signal):.3f}")

# Main processing function
def process_eeg_data(base_path='/home/anower/All/Python/Thesis_VS_code/full_epoch/Data/raw_data',
                     save_path='/home/anower/All/Python/Thesis_VS_code/full_epoch/2_Data_preprocessing/Figure/',
                     data_save_path='/home/anower/All/Python/Thesis_VS_code/full_epoch/Data/Data_processed/',
                     amplitude_threshold=100, variance_threshold=5,
                     normalization_method='z_score', normalization_axis='channel',
                     artifact_method='remove'):
    """
    Process EEG data: load, filter, detect artifacts, remove artifacts, normalize, visualize, and save
    """
    os.makedirs(data_save_path, exist_ok=True)
    filtered_data_all_subjects = {}
    processed_data_all_subjects = {}
    processing_summary = {}

    # Step 1: Load and filter data
    print("=" * 60)
    print("STEP 1: LOADING AND BANDPASS FILTERING")
    print("=" * 60)

    for i in range(1, 21):
        subject_id = f's{i:02d}'
        filename = f'music_listening_experiment_{subject_id}.mat'
        filepath = os.path.join(base_path, filename)

        try:
            print(f"\nLoading {filename}...")
            data = load_mat_file(filepath)
            print("Keys in MAT file:", data.keys())

            # Get sampling frequency
            fs = int(np.array(data['Fs'])[0][0]) if 'Fs' in data.keys() else 128
            print(f"Sampling frequency: {fs} Hz")

            # Process both EEG_Songs and EEG_Rest
            eeg_data_types = ['EEG_Songs', 'EEG_Rest']
            subject_data = {}

            for eeg_type in eeg_data_types:
                if eeg_type in data.keys():
                    eeg_data = np.array(data[eeg_type])
                    if eeg_data.shape[0] > eeg_data.shape[1]:
                        eeg_data = eeg_data.T
                        print(f"Transposed {eeg_type} data to shape: {eeg_data.shape}")
                    print(f"Original {eeg_type} shape: {eeg_data.shape}")
                    filtered_eeg = bandpass_filter(eeg_data, sampling_rate=fs)
                    subject_data[eeg_type] = filtered_eeg
                    print(f"Successfully filtered {eeg_type}: shape {filtered_eeg.shape}")
                else:
                    subject_data[eeg_type] = None
                    print(f"{eeg_type} not found in {filename}")

            # Store song ratings
            subject_data['song_ratings'] = np.array(data['song_ratings']) if 'song_ratings' in data.keys() else None
            subject_data['Fs'] = fs

            filtered_data_all_subjects[subject_id] = subject_data

            # Visualize bandpass filter effects for EEG_Songs (show only for s01)
            if 'EEG_Songs' in subject_data and subject_data['EEG_Songs'] is not None:
                print(f"\n🎵 Visualizing bandpass filter effects for {subject_id} (EEG_Songs)...")
                visualize_filter_effects(
                    subject_id,
                    original_data=data['EEG_Songs'] if data['EEG_Songs'].shape[0] <= data['EEG_Songs'].shape[1] else data['EEG_Songs'].T,
                    filtered_data=subject_data['EEG_Songs'],
                    sampling_rate=fs,
                    channel_idx=0,
                    trial_idx=0,
                    save_path=save_path,
                    show_plot=(subject_id == 's01')
                )

        except FileNotFoundError:
            print(f"File not found: {filepath}")
        except Exception as e:
            print(f"Error processing {subject_id}: {str(e)}")

    # Step 2: Artifact removal and normalization for EEG_Songs
    print(f"\n{'='*60}")
    print("STEP 2: ARTIFACT REMOVAL AND NORMALIZATION (EEG_Songs)")
    print(f"{'='*60}")

    for subject_id, data_dict in filtered_data_all_subjects.items():
        print(f"\n{'='*20} Processing {subject_id} (EEG_Songs) {'='*20}")
        if 'EEG_Songs' not in data_dict or data_dict['EEG_Songs'] is None:
            print(f"No EEG_Songs data for {subject_id}. Skipping artifact removal and normalization.")
            processed_data_all_subjects[subject_id] = {
                'EEG_Songs': None,
                'EEG_Rest': data_dict['EEG_Rest'],
                'Fs': data_dict['Fs'],
                'song_ratings': data_dict['song_ratings']
            }
            continue

        eeg_data = data_dict['EEG_Songs']
        print(f"Processing EEG_Songs data: {eeg_data.shape}")

        # Artifact Detection
        bad_trials, bad_channels, artifact_stats = detect_artifacts(
            eeg_data,
            amplitude_threshold=amplitude_threshold,
            variance_threshold=variance_threshold,
            sampling_rate=data_dict['Fs']
        )

        # Artifact Removal
        clean_data, channel_map = remove_artifacts(
            eeg_data, bad_trials, bad_channels, method=artifact_method
        )

        # Normalization
        normalized_data, norm_params = normalize_signals(
            clean_data,
            method=normalization_method,
            axis=normalization_axis
        )

        # Store processed data
        processed_data_all_subjects[subject_id] = {
            'EEG_Songs_clean': clean_data,
            'EEG_Songs_normalized': normalized_data,
            'EEG_Rest': data_dict['EEG_Rest'],  # Store filtered EEG_Rest
            'bad_trials': bad_trials,
            'bad_channels': bad_channels,
            'channel_map': channel_map,
            'normalization_params': norm_params,
            'artifact_stats': artifact_stats,
            'Fs': data_dict['Fs'],
            'song_ratings': data_dict['song_ratings']
        }

        # Store summary statistics for EEG_Songs
        processing_summary[subject_id] = {
            'original_shape': eeg_data.shape,
            'clean_shape': clean_data.shape,
            'normalized_shape': normalized_data.shape,
            'trials_removed': np.sum(bad_trials),
            'channels_removed': np.sum(bad_channels),
            'data_retention': (clean_data.size / eeg_data.size) * 100
        }

        # Visualize preprocessing effects for EEG_Songs (show only for s01)
        print(f"\n🎵 Visualizing preprocessing effects for {subject_id} (EEG_Songs)...")
        visualize_preprocessing_effects(
            filtered_data_all_subjects[subject_id]['EEG_Songs'],
            processed_data_all_subjects[subject_id]['EEG_Songs_clean'],
            processed_data_all_subjects[subject_id]['EEG_Songs_normalized'],
            subject_id,
            channel_idx=0,
            trial_idx=0,
            save_path=save_path,
            show_plot=(subject_id == 's01')
        )

        print(f"✅ {subject_id} processing complete!")

    # Save processed data
    output_file = os.path.join(data_save_path, 'processed_eeg_data_full.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(processed_data_all_subjects, f)
    print(f"✅ Processed data saved to '{output_file}'")

    # Print overall summary
    print(f"\n{'='*20} PROCESSING SUMMARY {'='*20}")
    total_subjects = len(processing_summary)
    total_trials_removed = sum([s['trials_removed'] for s in processing_summary.values()])
    total_channels_removed = sum([s['channels_removed'] for s in processing_summary.values()])
    avg_retention = np.mean([s['data_retention'] for s in processing_summary.values()])

    print(f"Subjects processed: {total_subjects}")
    print(f"Total trials removed (EEG_Songs): {total_trials_removed}")
    print(f"Total channels removed (EEG_Songs): {total_channels_removed}")
    print(f"Average data retention (EEG_Songs): {avg_retention:.1f}%")

    print(f"\n{'='*50}")
    print("DETAILED PROCESSING SUMMARY (EEG_Songs)")
    print(f"{'='*50}")
    for subject_id, stats in processing_summary.items():
        print(f"\n{subject_id}:")
        print(f"  Original shape: {stats['original_shape']}")
        print(f"  Final shape: {stats['normalized_shape']}")
        print(f"  Trials removed: {stats['trials_removed']}")
        print(f"  Channels removed: {stats['channels_removed']}")
        print(f"  Data retention: {stats['data_retention']:.1f}%")

    print(f"\n{'='*50}")
    print("EEG DATA SUMMARY")
    print(f"{'='*50}")
    for subject_id, data_dict in processed_data_all_subjects.items():
        print(f"\n{subject_id}:")
        if data_dict['EEG_Songs_clean'] is not None:
            print(f"  EEG_Songs (clean): {data_dict['EEG_Songs_clean'].shape}")
        if data_dict['EEG_Songs_normalized'] is not None:
            print(f"  EEG_Songs (normalized): {data_dict['EEG_Songs_normalized'].shape}")
        if data_dict['EEG_Rest'] is not None:
            print(f"  EEG_Rest (filtered): {data_dict['EEG_Rest'].shape}")
        print(f"  Sampling rate: {data_dict['Fs']} Hz")
        if data_dict['song_ratings'] is not None:
            print(f"  Song ratings: {data_dict['song_ratings'].shape}")

    print(f"\n{'='*50}")
    print("HOW TO ACCESS YOUR PROCESSED DATA")
    print(f"{'='*50}")
    print("Your processed data is now available in 'processed_data_all_subjects' dictionary:")
    print("- processed_data_all_subjects[subject_id]['EEG_Songs_clean']      # After artifact removal")
    print("- processed_data_all_subjects[subject_id]['EEG_Songs_normalized'] # Final normalized data")
    print("- processed_data_all_subjects[subject_id]['EEG_Rest']             # Filtered rest data")
    print("- processed_data_all_subjects[subject_id]['bad_trials']           # Which trials were removed (EEG_Songs)")
    print("- processed_data_all_subjects[subject_id]['bad_channels']         # Which channels were removed (EEG_Songs)")
    print("- processed_data_all_subjects[subject_id]['channel_map']          # Channel mapping after removal (EEG_Songs)")
    print("- processed_data_all_subjects[subject_id]['Fs']                   # Sampling rate")
    print("- processed_data_all_subjects[subject_id]['song_ratings']         # Song ratings")

    if 's01' in processed_data_all_subjects and processed_data_all_subjects['s01']['EEG_Songs_normalized'] is not None:
        final_data = processed_data_all_subjects['s01']['EEG_Songs_normalized']
        print(f"\nExample - Final processed EEG_Songs data for s01:")
        print(f"Shape: {final_data.shape}")
        print(f"Mean: {np.mean(final_data):.6f} (should be ~0 for z-score)")
        print(f"Std: {np.std(final_data):.6f} (should be ~1 for z-score)")
        print(f"Range: [{np.min(final_data):.3f}, {np.max(final_data):.3f}]")

    return processed_data_all_subjects, processing_summary

# Run the full pipeline
if __name__ == "__main__":
    processed_data, summary = process_eeg_data(
        base_path='/home/anower/All/Python/Thesis_VS_code/full_epoch/Data/raw_data',
        save_path='/home/anower/All/Python/Thesis_VS_code/full_epoch/2_Data_preprocessing/Figure/',
        data_save_path='/home/anower/All/Python/Thesis_VS_code/full_epoch/Data/Data_processed/',
        amplitude_threshold=100,
        variance_threshold=5,
        normalization_method='z_score',
        normalization_axis='channel',
        artifact_method='remove'
    )