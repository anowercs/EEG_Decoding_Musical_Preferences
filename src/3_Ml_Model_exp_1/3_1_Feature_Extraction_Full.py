#2   best code

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import skew, kurtosis, entropy
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import pickle
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import os

print("="*70)
print("STEP 3.1: FEATURE EXTRACTION (FULL-SONG EPOCHS)")
print("="*70)

class EEGFeatureExtractor:
    def __init__(self, fs=128, epoch_duration_s=80):
        self.fs = fs
        self.epoch_duration_s = epoch_duration_s
        self.bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 40)
        }
        self.channel_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
                             'O1', 'O2', 'F7', 'F8', 'T3', 'T4']

    def extract_all_features(self, eeg_epoch, rest_epoch=None):
        features = {}
        try:
            psd_features = self.extract_psd_features(eeg_epoch, rest_epoch)
            features.update(psd_features)
            stat_features = self.extract_statistical_features(eeg_epoch)
            features.update(stat_features)
            conn_features = self.extract_connectivity_features(eeg_epoch)
            features.update(conn_features)
            nonlinear_features = self.extract_nonlinear_features(eeg_epoch)
            features.update(nonlinear_features)
            temporal_features = self.extract_temporal_features(eeg_epoch)
            features.update(temporal_features)
            # Check for NaN or infinite values
            for key, value in features.items():
                if not np.isfinite(value):
                    features[key] = 0
        except Exception as e:
            print(f"Error extracting features: {str(e)}")
            return None
        return features

    def extract_psd_features(self, eeg_epoch, rest_epoch=None):
        features = {}
        nperseg = min(512, eeg_epoch.shape[1])  # Reduced for speed
        psd_songs = []
        for ch_idx, ch_name in enumerate(self.channel_names):
            channel_data = eeg_epoch[ch_idx, :]
            freqs, psd = signal.welch(channel_data, self.fs, nperseg=nperseg)
            psd_songs.append(psd)
            total_power = np.sum(psd[(freqs >= 1) & (freqs <= 40)])
            for band_name, (low, high) in self.bands.items():
                band_mask = (freqs >= low) & (freqs <= high)
                band_power = np.sum(psd[band_mask])
                features[f'{ch_name}_{band_name}_power'] = band_power
                relative_power = band_power / total_power if total_power > 0 else 0
                features[f'{ch_name}_{band_name}_rel_power'] = relative_power
                if np.any(band_mask):
                    peak_freq = freqs[band_mask][np.argmax(psd[band_mask])]
                    features[f'{ch_name}_{band_name}_peak_freq'] = peak_freq
        if rest_epoch is not None:
            psd_rest = []
            for ch_idx in range(rest_epoch.shape[0]):
                freqs, psd = signal.welch(rest_epoch[ch_idx, :], self.fs, nperseg=nperseg)
                psd_rest.append(psd)
            mean_psd_rest = np.mean(psd_rest, axis=0)
            mean_psd_songs = np.mean(psd_songs, axis=0)
            for band_name, (low, high) in self.bands.items():
                band_mask = (freqs >= low) & (freqs <= high)
                song_power = np.sum(mean_psd_songs[band_mask])
                rest_power = np.sum(mean_psd_rest[band_mask])
                features[f'normalized_{band_name}_power'] = song_power / rest_power if rest_power > 0 else 0
        for band_name in self.bands:
            f3_power = features[f'F3_{band_name}_power']
            f4_power = features[f'F4_{band_name}_power']
            features[f'frontal_{band_name}_asymmetry'] = (f4_power - f3_power) / (f4_power + f3_power + 1e-10)
            p3_power = features[f'P3_{band_name}_power']
            p4_power = features[f'P4_{band_name}_power']
            features[f'parietal_{band_name}_asymmetry'] = (p4_power - p3_power) / (p4_power + p3_power + 1e-10)
        return features

    def extract_statistical_features(self, eeg_epoch):
        features = {}
        for ch_idx, ch_name in enumerate(self.channel_names):
            channel_data = eeg_epoch[ch_idx, :]
            features[f'{ch_name}_mean'] = np.mean(channel_data)
            features[f'{ch_name}_std'] = np.std(channel_data)
            features[f'{ch_name}_var'] = np.var(channel_data)
            features[f'{ch_name}_skew'] = skew(channel_data)
            features[f'{ch_name}_kurtosis'] = kurtosis(channel_data)
            features[f'{ch_name}_p25'] = np.percentile(channel_data, 25)
            features[f'{ch_name}_p75'] = np.percentile(channel_data, 75)
            features[f'{ch_name}_iqr'] = features[f'{ch_name}_p75'] - features[f'{ch_name}_p25']
            features[f'{ch_name}_ptp'] = np.ptp(channel_data)
            zero_crossings = np.sum(np.diff(np.sign(channel_data)) != 0)
            features[f'{ch_name}_zcr'] = zero_crossings / len(channel_data)
            features[f'{ch_name}_rms'] = np.sqrt(np.mean(channel_data**2))
        return features

    def extract_connectivity_features(self, eeg_epoch):
        features = {}
        n_channels = eeg_epoch.shape[0]
        corr_matrix = np.corrcoef(eeg_epoch)
        triu_indices = np.triu_indices(n_channels, k=1)
        correlations = corr_matrix[triu_indices]
        features['mean_connectivity'] = np.nanmean(correlations)
        features['std_connectivity'] = np.nanstd(correlations)
        features['max_connectivity'] = np.nanmax(correlations)
        features['min_connectivity'] = np.nanmin(correlations)
        strong_connections = np.sum(np.abs(correlations) > 0.5)
        total_connections = len(correlations)
        features['network_density'] = strong_connections / total_connections if total_connections > 0 else 0
        connectivity_pairs = [
            ('Fp1', 'Fp2'), ('F3', 'F4'), ('C3', 'C4'), ('P3', 'P4'), ('O1', 'O2'),
            ('Fp1', 'O1'), ('Fp2', 'O2'), ('F3', 'C3'), ('F4', 'C4')
        ]
        for ch1, ch2 in connectivity_pairs:
            idx1 = self.channel_names.index(ch1)
            idx2 = self.channel_names.index(ch2)
            features[f'corr_{ch1}_{ch2}'] = corr_matrix[idx1, idx2]
        for band_name, (low, high) in self.bands.items():
            b, a = signal.butter(4, [low/self.fs*2, high/self.fs*2], btype='band')
            filtered_signals = signal.filtfilt(b, a, eeg_epoch, axis=1)
            analytic_signals = signal.hilbert(filtered_signals, axis=1)
            phases = np.angle(analytic_signals)
            plv_f3_f4 = np.abs(np.mean(np.exp(1j * (phases[2] - phases[3]))))
            features[f'plv_F3_F4_{band_name}'] = plv_f3_f4
        return features

    def extract_nonlinear_features(self, eeg_epoch):
        features = {}
        for ch_idx, ch_name in enumerate(self.channel_names[:4]):
            channel_data = eeg_epoch[ch_idx, :]
            mobility, complexity = self.hjorth_parameters(channel_data)
            features[f'{ch_name}_hjorth_mobility'] = mobility
            features[f'{ch_name}_hjorth_complexity'] = complexity
            features[f'{ch_name}_sample_entropy'] = self.sample_entropy(channel_data, m=2, r=0.2*np.std(channel_data))
            _, psd = signal.welch(channel_data, self.fs, nperseg=512)
            psd_norm = psd / np.sum(psd)
            features[f'{ch_name}_spectral_entropy'] = entropy(psd_norm)
        return features

    def extract_temporal_features(self, eeg_epoch):
        features = {}
        segment_length = int(20 * self.fs)
        n_segments = eeg_epoch.shape[1] // segment_length
        if n_segments < 1:
            n_segments = 1
            segment_length = eeg_epoch.shape[1]
        for ch_idx, ch_name in enumerate(self.channel_names[:4]):
            channel_data = eeg_epoch[ch_idx, :]
            segment_powers = []
            for seg in range(n_segments):
                start = seg * segment_length
                end = min((seg + 1) * segment_length, channel_data.shape[0])
                segment_power = np.mean(channel_data[start:end]**2)
                segment_powers.append(segment_power)
            features[f'{ch_name}_temporal_std'] = np.std(segment_powers) if len(segment_powers) > 1 else 0
            features[f'{ch_name}_temporal_trend'] = np.polyfit(range(n_segments), segment_powers, 1)[0] if len(segment_powers) > 1 else 0
        return features

    def hjorth_parameters(self, signal_data):
        activity = np.var(signal_data)
        diff1 = np.diff(signal_data)
        mobility = np.sqrt(np.var(diff1) / activity) if activity > 0 else 0
        diff2 = np.diff(diff1)
        complexity = np.sqrt(np.var(diff2) / np.var(diff1)) / mobility if mobility > 0 else 0
        return mobility, complexity

    def sample_entropy(self, signal_data, m=2, r=0.2, N=1000):
        if len(signal_data) > N:
            signal_data = signal_data[:N]
        def _maxdist(xi, xj, m):
            return max([abs(float(xi[k]) - float(xj[k])) for k in range(m)])
        def _phi(m):
            patterns = np.array([signal_data[i:i+m] for i in range(len(signal_data)-m+1)])
            C = 0
            for i in range(len(patterns)):
                template = patterns[i]
                for j in range(len(patterns)):
                    if i != j and _maxdist(template, patterns[j], m) <= r:
                        C += 1
            return C / (len(patterns) * (len(patterns) - 1)) if len(patterns) > 1 else 0
        try:
            return -np.log(_phi(m+1) / _phi(m)) if _phi(m) > 0 else 0
        except:
            return 0

def process_all_subjects(epoched_data_path, processed_data_path, base_path, use_rest=True):
    print("Loading epoched data...")
    with open(epoched_data_path, 'rb') as f:
        epoched_data = pickle.load(f)
    print("Loading processed data for bad_trials...")
    with open(processed_data_path, 'rb') as f:
        processed_data = pickle.load(f)

    extractor = EEGFeatureExtractor(fs=128, epoch_duration_s=80)
    all_features = []
    all_labels = []
    all_subjects = []
    all_songs = []

    print(f"\nExtracting features from {len(epoched_data)} subjects...")
    for subject_id in tqdm(sorted(epoched_data.keys())):
        eeg_songs = epoched_data[subject_id]['EEG_Songs_epoched']
        eeg_rest = epoched_data[subject_id]['EEG_Rest_epoched'] if use_rest else None
        song_labels = epoched_data[subject_id]['song_labels']
        n_epochs = eeg_songs.shape[0]
        bad_trials = processed_data[subject_id].get('bad_trials', [])

        # Correctly filter song_labels based on bad_trials
        if isinstance(bad_trials, np.ndarray) and bad_trials.dtype == bool:
            kept_indices = np.where(~bad_trials)[0].tolist()
        else:
            kept_indices = [i for i in range(30) if i not in bad_trials]
        if len(song_labels) > n_epochs:
            print(f"Warning: Truncating song_labels for {subject_id} from {len(song_labels)} to {n_epochs}")
            song_labels = song_labels[:n_epochs]
            kept_indices = kept_indices[:n_epochs]
        elif len(song_labels) < n_epochs:
            print(f"Error: Too few labels for {subject_id} - {n_epochs} epochs but {len(song_labels)} labels")
            n_epochs = len(song_labels)
            eeg_songs = eeg_songs[:n_epochs]

        print(f"{subject_id}: {n_epochs} epochs, {len(song_labels)} labels, bad_trials={bad_trials}, kept_indices={kept_indices}")

        for epoch_idx in tqdm(range(n_epochs), desc=f"Processing {subject_id} epochs", leave=False):
            try:
                epoch_features = extractor.extract_all_features(
                    eeg_songs[epoch_idx],
                    eeg_rest[0] if eeg_rest is not None else None
                )
                if epoch_features is None:
                    print(f"Skipping {subject_id}, epoch {epoch_idx} due to feature extraction error")
                    continue
                feature_values = list(epoch_features.values())
                all_features.append(feature_values)
                all_labels.append(song_labels[epoch_idx])
                all_subjects.append(subject_id)
                all_songs.append(kept_indices[epoch_idx])
            except Exception as e:
                print(f"Error processing {subject_id}, epoch {epoch_idx}: {str(e)}")
                continue

    if not all_features:
        raise ValueError("No features extracted. Check data or feature extraction process.")
    X = np.array(all_features)
    y = np.array(all_labels)
    feature_names = list(epoch_features.keys())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"\n✅ Feature extraction complete!")
    print(f"Feature matrix shape: {X_scaled.shape}")
    print(f"Number of features: {len(feature_names)}")
    print(f"Number of samples: {len(y)}")
    print(f"Rating statistics: mean={y.mean():.2f}, std={y.std():.2f}, range=[{y.min()}, {y.max()}]")

    return X_scaled, y, feature_names, all_subjects, all_songs, scaler

def visualize_feature_statistics(X, y, feature_names, is_categorical=False, save_path='/home/anower/All/Python/Thesis_VS_code/full_epoch/3_Ml_Model/Figure/'):
    os.makedirs(save_path, exist_ok=True)
    plt.figure(figsize=(12, 10))
    feature_subset = X[:, :30]
    corr_matrix = np.corrcoef(feature_subset.T)
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0,
                xticklabels=feature_names[:30], yticklabels=feature_names[:30])
    plt.title('Feature Correlation Matrix (First 30 Features)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{save_path}/3.1_feature_correlation_full.png', dpi=300, bbox_inches='tight')
    plt.show(block=False)
    plt.pause(0.1)

    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.hist(y, bins=20 if not is_categorical else len(np.unique(y)), edgecolor='black', alpha=0.7)
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.title('Distribution of Music Ratings')
    plt.subplot(1, 2, 2)
    plt.boxplot(y)
    plt.ylabel('Rating')
    plt.title('Rating Distribution')
    plt.tight_layout()
    plt.savefig(f'{save_path}/3.1_rating_distribution_full.png', dpi=300, bbox_inches='tight')
    plt.show(block=False)
    plt.pause(0.1)

    if is_categorical:
        from scipy.stats import f_oneway
        correlations = []
        for i in range(X.shape[1]):
            groups = [X[y == val, i] for val in np.unique(y) if len(X[y == val, i]) > 0]
            if len(groups) > 1:
                f_stat, _ = f_oneway(*groups)
                correlations.append(f_stat)
            else:
                correlations.append(0)
    else:
        correlations = []
        for i in range(X.shape[1]):
            corr = np.corrcoef(X[:, i], y)[0, 1]
            correlations.append(np.abs(corr) if np.isfinite(corr) else 0)
    top_indices = np.argsort(correlations)[::-1][:20]
    plt.figure(figsize=(10, 6))
    plt.bar(range(20), [correlations[i] for i in top_indices])
    plt.xticks(range(20), [feature_names[i] for i in top_indices], rotation=45, ha='right')
    plt.xlabel('Features')
    plt.ylabel('ANOVA F-statistic' if is_categorical else 'Absolute Correlation with Rating')
    plt.title('Top 20 Features by Importance (Full Epochs)')
    plt.tight_layout()
    plt.savefig(f'{save_path}/3.1_feature_importance_full.png', dpi=300, bbox_inches='tight')
    plt.show(block=False)
    plt.pause(0.1)

if __name__ == "__main__":
    epoched_data_path = '/home/anower/All/Python/Thesis_VS_code/full_epoch/Data/Data_processed/epoched_eeg_data_full.pkl'
    processed_data_path = '/home/anower/All/Python/Thesis_VS_code/full_epoch/Data/Data_processed/processed_eeg_data_full.pkl'
    base_path = '/home/anower/All/Python/Thesis_VS_code/full_epoch/Data/'
    save_path = '/home/anower/All/Python/Thesis_VS_code/full_epoch/Figure/'

    X, y, feature_names, subjects, songs, scaler = process_all_subjects(
        epoched_data_path, processed_data_path, base_path, use_rest=True
    )

    visualize_feature_statistics(X, y, feature_names, is_categorical=False, save_path=save_path)

    feature_data = {
        'X': X,
        'y': y,
        'feature_names': feature_names,
        'subjects': subjects,
        'songs': songs,
        'scaler': scaler
    }
    with open(f'{save_path}/eeg_features_full.pkl', 'wb') as f:
        pickle.dump(feature_data, f)
    print(f"\n✅ Features saved to '{save_path}eeg_features_full.pkl'")

    print("\n" + "="*70)
    print("FEATURE EXTRACTION SUMMARY (FULL-SONG EPOCHS)")
    print("="*70)
    print(f"\n📊 Feature Categories:")
    print(f"1. Power Spectral Density (PSD):")
    print(f"   - Absolute & relative power for 5 bands × 14 channels = 140 features")
    print(f"   - Peak frequencies for 5 bands × 14 channels = 70 features")
    print(f"   - Normalized power for 5 bands = 5 features")
    print(f"   - Hemispheric asymmetry for 5 bands × 2 regions = 10 features")
    print(f"\n2. Statistical Features:")
    print(f"   - 11 statistical measures × 14 channels = 154 features")
    print(f"\n3. Connectivity Features:")
    print(f"   - Global connectivity metrics = 5 features")
    print(f"   - Specific channel pairs = 9 features")
    print(f"   - Phase locking values = 5 features")
    print(f"\n4. Nonlinear Features:")
    print(f"   - Hjorth parameters, entropy measures for 4 channels = 16 features")
    print(f"\n5. Temporal Features:")
    print(f"   - Temporal variability for 4 channels = 8 features")
    print(f"\n📈 Total: ~{len(feature_names)} features")