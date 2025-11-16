########################2222222222222222###############

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Set backend to TkAgg
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from matplotlib.gridspec import GridSpec
import h5py
import pickle
import os

def load_processed_data(data_path='/home/anower/All/Python/Thesis_VS_code/full_epoch/Data/Data_processed/processed_eeg_data_full.pkl'):
    """Load processed EEG data from pickle file"""
    try:
        with open(data_path, 'rb') as f:
            processed_data = pickle.load(f)
        print(f"Successfully loaded processed data from '{data_path}'")
        return processed_data
    except FileNotFoundError:
        print(f"Error: Processed data file not found at '{data_path}'")
        raise
    except Exception as e:
        print(f"Error loading processed data: {str(e)}")
        raise

def segment_full_epochs(data, fs, is_rest=False, song_ratings=None):
    """
    Segment EEG data into full-duration epochs (complete songs/rest)
    Parameters:
    - data: EEG data (n_original_epochs, n_channels, n_samples) for songs, (n_samples, n_channels) for rest
    - fs: Sampling frequency (Hz)
    - is_rest: Boolean indicating if data is EEG_Rest
    - song_ratings: Array of song ratings (n_songs, 1) for labeling song epochs
    Returns:
    - epochs: Full-duration epochs (n_epochs, n_channels, n_samples_per_epoch)
    - labels: Associated labels (song_ratings for songs, 0 for rest)
    """
    if is_rest:
        # Rest data comes as (n_samples, n_channels), reshape to (1, n_channels, n_samples)
        data = data[np.newaxis, :, :]
        labels = np.array([0])  # Neutral label for rest
    else:
        # Song data is already (n_songs, n_channels, n_samples)
        labels = song_ratings.flatten() if song_ratings is not None else np.arange(data.shape[0])
    
    return data, labels

def create_full_epoch_documentation(processed_data, original_fs=128, subject_example='s01',
                                   save_path='/home/anower/All/Python/Thesis_VS_code/full_epoch/2_Data_preprocessing/Figure/'):
    """
    Create comprehensive full-epoch documentation for thesis/paper
    """
    os.makedirs(save_path, exist_ok=True)

    print("="*70)
    print("STEP 2.3: FULL-DURATION EPOCH DEFINITION AND SEGMENTATION")
    print("="*70)

    # Get example subject data
    if subject_example not in processed_data:
        raise ValueError(f"Subject {subject_example} not found in processed data")
    
    subject_data = processed_data[subject_example]
    eeg_songs = subject_data['EEG_Songs_normalized']
    eeg_rest = subject_data.get('EEG_Rest_normalized')
    song_ratings = subject_data.get('song_ratings')
    
    if eeg_songs is None:
        raise ValueError(f"No EEG_Songs_normalized data for subject {subject_example}")

    # Process full epochs
    songs_epoched, song_labels = segment_full_epochs(eeg_songs, original_fs, song_ratings=song_ratings)
    n_epochs_songs, n_channels, n_samples_per_epoch = songs_epoched.shape
    epoch_duration = n_samples_per_epoch / original_fs

    # Process rest epoch (if available)
    rest_epoched, rest_labels = None, None
    if eeg_rest is not None:
        rest_epoched, rest_labels = segment_full_epochs(eeg_rest.T, original_fs, is_rest=True)

    print(f"\n📊 FULL EPOCH PARAMETERS (EEG_Songs):")
    print(f"├─ Epoch Definition: Complete song duration")
    print(f"├─ Average Epoch Duration: {epoch_duration:.1f} seconds")
    print(f"├─ Sampling Frequency: {original_fs} Hz")
    print(f"├─ Samples per Epoch: {n_samples_per_epoch:,}")
    print(f"├─ Number of Channels: {n_channels}")
    print(f"├─ Epochs per Song: 1 (full song)")
    print(f"└─ Total Song Epochs (after artifact removal): {n_epochs_songs}")
    
    print(f"\n📊 FULL EPOCH PARAMETERS (EEG_Rest):")
    if eeg_rest is not None:
        rest_duration = rest_epoched.shape[2] / original_fs
        print(f"├─ Epoch Definition: Complete rest period")
        print(f"├─ Rest Duration: {rest_duration:.1f} seconds")
        print(f"└─ Total Rest Epochs: {rest_epoched.shape[0]}")
    else:
        print(f"└─ No rest data available")

    # Create visualization
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(4, 2, height_ratios=[1, 1.5, 1.5, 1], hspace=0.3)

    # 1. Epoch Structure Diagram
    ax1 = fig.add_subplot(gs[0, :])
    create_full_epoch_structure_diagram(ax1, epoch_duration, n_epochs_songs)

    # 2. Single Epoch Visualization (first 10 seconds)
    ax2 = fig.add_subplot(gs[1, :])
    visualize_full_epoch_sample(ax2, songs_epoched[0], original_fs)

    # 3. Multi-Channel Epoch Display
    ax3 = fig.add_subplot(gs[2, 0])
    display_full_epoch_heatmap(ax3, songs_epoched[0], original_fs)

    # 4. Epoch Power Spectrum
    ax4 = fig.add_subplot(gs[2, 1])
    plot_full_epoch_spectrum(ax4, songs_epoched[0], original_fs, rest_epoched[0] if rest_epoched is not None else None)

    # 5. Statistics Summary
    ax5 = fig.add_subplot(gs[3, :])
    create_full_epoch_statistics_table(ax5, processed_data)

    plt.suptitle('EEG Full-Duration Epoch Structure and Characteristics', fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save the figure
    output_file = os.path.join('/home/anower/All/Python/Thesis_VS_code/full_epoch/2_Data_preprocessing/Figure/', f'2.1_eeg_full_epoch_structure_{subject_example}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Full epoch structure figure saved as '{output_file}'")

    plt.show(block=False)
    plt.pause(0.1)

    return fig, {'EEG_Songs_epoched': songs_epoched, 'song_labels': song_labels, 
                 'EEG_Rest_epoched': rest_epoched, 'rest_labels': rest_labels}

def create_full_epoch_structure_diagram(ax, epoch_duration, total_epochs):
    """Create visual diagram showing full epoch structure"""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)

    ax.text(5, 2.5, 'Full-Duration Epoch Structure', ha='center', fontsize=14, fontweight='bold')

    # Show sample songs
    n_display = min(5, total_epochs)
    epoch_width = 9 / n_display
    
    for i in range(n_display):
        x_start = 0.5 + i * epoch_width
        # Different colors for different songs
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
        epoch_rect = patches.Rectangle((x_start, 1.5), epoch_width-0.1, 0.6,
                                     facecolor=colors[i % len(colors)], edgecolor='black')
        ax.add_patch(epoch_rect)
        ax.text(x_start + epoch_width/2, 1.8, f'Song {i+1}\n({epoch_duration:.0f}s)',
                ha='center', va='center', fontsize=9, fontweight='bold')

    # Add labels
    ax.text(0.5, 1.0, 'Individual Complete Songs', fontsize=10, fontweight='bold')
    ax.text(9.5, 1.0, f'n = {total_epochs} songs\n(after cleaning)',
            fontsize=10, ha='right', style='italic')
    
    # Add rest period
    rest_rect = patches.Rectangle((0.5, 0.3), 2, 0.4, facecolor='gray', edgecolor='black', alpha=0.7)
    ax.add_patch(rest_rect)
    ax.text(1.5, 0.5, 'Rest Period\n(~70s)', ha='center', va='center', fontsize=9, fontweight='bold')

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

def visualize_full_epoch_sample(ax, epoch_data, fs):
    """Visualize first 10 seconds of a full epoch"""
    # Show only first 10 seconds for visualization
    time_limit = 10
    samples_limit = int(time_limit * fs)
    time = np.arange(samples_limit) / fs

    # Create offset for each channel
    offsets = np.arange(epoch_data.shape[0]) * 4

    for ch in range(epoch_data.shape[0]):
        signal = epoch_data[ch, :samples_limit] + offsets[ch]
        ax.plot(time, signal, linewidth=0.8, alpha=0.8)

    ax.set_xlabel('Time (seconds)', fontsize=11)
    ax.set_ylabel('Channels', fontsize=11)
    ax.set_title('Full Song Epoch Example (First 10 seconds shown)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, time_limit)

    # Add channel labels
    channel_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
                     'O1', 'O2', 'F7', 'F8', 'T3', 'T4']
    ax.set_yticks(offsets)
    ax.set_yticklabels(channel_names[:len(offsets)], fontsize=9)

def display_full_epoch_heatmap(ax, epoch_data, fs):
    """Display first 5 seconds of epoch as heatmap"""
    time_limit = 5
    samples_limit = int(time_limit * fs)
    data_subset = epoch_data[:, :samples_limit]

    im = ax.imshow(data_subset, aspect='auto', cmap='RdBu_r',
                   extent=[0, time_limit, data_subset.shape[0], 0], vmin=-3, vmax=3)

    ax.set_xlabel('Time (seconds)', fontsize=11)
    ax.set_ylabel('Channel Index', fontsize=11)
    ax.set_title('Full Song Epoch Heatmap (First 5 seconds)', fontsize=12, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Amplitude (z-score)', fontsize=10)

def plot_full_epoch_spectrum(ax, song_epoch_data, fs, rest_epoch_data=None):
    """Plot power spectrum of full epoch"""
    from scipy import signal

    # Compute PSD for song epoch (use entire duration)
    all_psd_songs = []
    for ch in range(song_epoch_data.shape[0]):
        f, psd = signal.welch(song_epoch_data[ch], fs, nperseg=1024)  # Larger window for full epochs
        all_psd_songs.append(psd)
    mean_psd_songs = np.mean(all_psd_songs, axis=0)

    # Compute PSD for rest epoch (if available)
    mean_psd_rest = None
    if rest_epoch_data is not None:
        all_psd_rest = []
        for ch in range(rest_epoch_data.shape[0]):
            f, psd = signal.welch(rest_epoch_data[ch], fs, nperseg=1024)
            all_psd_rest.append(psd)
        mean_psd_rest = np.mean(all_psd_rest, axis=0)

    # Plot
    freq_mask = f <= 40
    ax.semilogy(f[freq_mask], mean_psd_songs[freq_mask], 'b-', label='Full Song', linewidth=2)
    if mean_psd_rest is not None:
        ax.semilogy(f[freq_mask], mean_psd_rest[freq_mask], 'r--', label='Rest Period', linewidth=2)

    # Highlight frequency bands
    bands = {
        'δ': (1, 4, 'red'),
        'θ': (4, 8, 'orange'), 
        'α': (8, 13, 'green'),
        'β': (13, 30, 'blue'),
        'γ': (30, 40, 'purple')
    }
    for band, (low, high, color) in bands.items():
        ax.axvspan(low, high, alpha=0.2, color=color, label=f'{band} ({low}-{high} Hz)')

    ax.set_xlabel('Frequency (Hz)', fontsize=11)
    ax.set_ylabel('Power Spectral Density', fontsize=11)
    ax.set_title('Average Power Spectrum of Full-Duration Epochs', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(0, 40)

def create_full_epoch_statistics_table(ax, processed_data):
    """Create summary statistics table for full epochs"""
    ax.axis('off')

    # Collect statistics
    stats = []
    for subject_id in sorted(processed_data.keys())[:5]:  # Show first 5 subjects
        data = processed_data[subject_id]['EEG_Songs_normalized']
        rest_data = processed_data[subject_id].get('EEG_Rest_normalized')
        
        if data is None:
            stats.append([subject_id, '0', '0', '0%', 'N/A', 'N/A'])
            continue
            
        n_songs, n_channels, n_samples = data.shape
        song_duration = n_samples / 128  # Assuming 128 Hz
        retention = (n_songs / 30) * 100  # Assuming 30 original songs
        
        # Rest data statistics
        rest_duration = rest_data.shape[0] / 128 if rest_data is not None else 0
        
        stats.append([
            subject_id,
            str(n_songs),
            f"{rest_duration:.1f}s" if rest_data is not None else "N/A",
            f"{retention:.1f}%",
            f"{np.mean(data):.3f} ± {np.std(data):.3f}",
            f"[{np.min(data):.2f}, {np.max(data):.2f}]"
        ])

    # Create table
    headers = ['Subject', 'Song Epochs', 'Rest Duration', 'Song Retention', 'Mean ± Std', 'Range']
    table = ax.table(cellText=stats, colLabels=headers,
                     cellLoc='center', loc='center',
                     colWidths=[0.15, 0.15, 0.15, 0.15, 0.25, 0.15])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # Style the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    ax.set_title('Full-Duration Epoch Summary Statistics (First 5 Subjects)', 
                fontsize=12, fontweight='bold', pad=20)

def save_full_epoched_data(processed_data, original_fs=128,
                          data_save_path='/home/anower/All/Python/Thesis_VS_code/full_epoch/Data/Data_processed/'):
    """Save full-duration epoched data"""
    os.makedirs(data_save_path, exist_ok=True)
    epoched_data = {}

    for subject_id in processed_data.keys():
        epoched_data[subject_id] = {}
        songs_data = processed_data[subject_id]['EEG_Songs_normalized']
        rest_data = processed_data[subject_id].get('EEG_Rest_normalized')
        song_ratings = processed_data[subject_id].get('song_ratings')

        # Process full song epochs
        if songs_data is not None:
            songs_epoched, song_labels = segment_full_epochs(songs_data, original_fs, song_ratings=song_ratings)
            epoched_data[subject_id]['EEG_Songs_epoched'] = songs_epoched
            epoched_data[subject_id]['song_labels'] = song_labels
        else:
            epoched_data[subject_id]['EEG_Songs_epoched'] = None
            epoched_data[subject_id]['song_labels'] = None

        # Process full rest epoch
        if rest_data is not None:
            rest_epoched, rest_labels = segment_full_epochs(rest_data.T, original_fs, is_rest=True)
            epoched_data[subject_id]['EEG_Rest_epoched'] = rest_epoched
            epoched_data[subject_id]['rest_labels'] = rest_labels
        else:
            epoched_data[subject_id]['EEG_Rest_epoched'] = None
            epoched_data[subject_id]['rest_labels'] = None

    # Save to pickle file
    output_file = os.path.join(data_save_path, 'epoched_eeg_data_full.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(epoched_data, f)
    print(f"Full epoched data saved to '{output_file}'")

    return epoched_data

# Usage example
if __name__ == "__main__":
    # Load processed data
    processed_data = load_processed_data()
    
    # Create full-epoch documentation
    epoch_fig, epoched_data_subject = create_full_epoch_documentation(
        processed_data,
        save_path='/home/anower/All/Python/Thesis_VS_code/full_epoch/Figure/'
    )
    
    # Save full epoched data
    epoched_data = save_full_epoched_data(processed_data)
    
    # Print formal epoch definition for thesis
    print("\n" + "="*70)
    print("FORMAL FULL-DURATION EPOCH DEFINITION FOR THESIS")
    print("="*70)
    print("""
In this study, EEG data was segmented into full-duration epochs corresponding to complete 
musical stimuli and the entire resting-state period to preserve the holistic neural response 
patterns for music preference analysis.

• Epoch Definition: Complete song duration (~80 seconds) and full rest period (~70 seconds)
• Epoch Onset: Beginning of stimulus presentation
• Epoch Offset: End of stimulus presentation  
• Songs per Subject: 30 (before artifact rejection)
• Rest Epochs per Subject: 1 complete period
• Data Preservation: Maintains temporal dynamics and sustained neural responses
• Labels: Each song epoch inherits its corresponding preference rating; rest epoch = neutral (0)

This approach captures the complete neural response to musical stimuli, enabling analysis of 
sustained brain activity patterns that may be critical for understanding music preference 
mechanisms and supporting machine learning classification models.
    """)