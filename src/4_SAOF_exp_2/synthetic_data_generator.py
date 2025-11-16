##Best

import numpy as np
import pandas as pd
import pickle
import os
import time
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("SYNTHETIC DATA GENERATION FOR EEG FEATURES")
print("="*70)

class SyntheticDataGenerator:
    """
    Generate synthetic EEG feature data using multiple techniques
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
    
    def map_to_valid_labels(self, y):
        """Map synthetic labels to nearest valid label [1.0, 3.0, 5.0]"""
        valid_labels = np.array([1.0, 3.0, 5.0])
        y_mapped = np.zeros_like(y)
        for i, val in enumerate(y):
            # Find nearest valid label
            distances = np.abs(valid_labels - val)
            y_mapped[i] = valid_labels[np.argmin(distances)]
        return y_mapped
    
    def load_original_data(self, features_path):
        """Load original features and ratings from pickle file"""
        try:
            with open(features_path, 'rb') as f:
                features_data = pickle.load(f)
            
            X = features_data['X']
            y = features_data['y']
            subjects = features_data['subjects']
            songs = features_data['songs']
            feature_names = features_data['feature_names']
            scaler = features_data.get('scaler')
            
            print(f"✅ Original data loaded successfully!")
            print(f"   Feature matrix shape: {X.shape}")
            print(f"   Ratings shape: {y.shape}")
            print(f"   Rating range: [{y.min()}, {y.max()}]")
            print(f"   Rating mean: {y.mean():.2f} ± {y.std():.2f}")
            print(f"   Number of subjects: {len(np.unique(subjects))}")
            print(f"   Number of songs: {len(np.unique(songs))}")
            print(f"   Number of features: {X.shape[1]}")
            
            return X, y, subjects, songs, feature_names, scaler
            
        except FileNotFoundError:
            print(f"❌ Error: Feature file not found at '{features_path}'")
            raise
        except Exception as e:
            print(f"❌ Error loading features: {str(e)}")
            raise
    
    def generate_gaussian_noise(self, X, y, subjects, songs, n_synthetic=None, noise_level=0.05): #0.1
        """Generate synthetic data by adding Gaussian noise to original samples"""
        print(f"\n🔧 Generating Gaussian noise synthetic data...")
        
        if n_synthetic is None:
            n_synthetic = len(X)
        
        sample_indices = np.random.choice(len(X), size=n_synthetic, replace=True)
        
        X_synthetic = X[sample_indices].copy()
        y_synthetic = y[sample_indices].copy()
        subjects_synthetic = [subjects[i] for i in sample_indices]
        songs_synthetic = [songs[i] for i in sample_indices]
        
        noise = np.random.normal(0, noise_level, X_synthetic.shape)
        feature_stds = np.std(X, axis=0)
        noise = noise * feature_stds
        X_synthetic = X_synthetic + noise
        
        y_synthetic = self.map_to_valid_labels(y_synthetic)
        
        print(f"   Generated {len(X_synthetic)} synthetic samples using Gaussian noise")
        return X_synthetic, y_synthetic, subjects_synthetic, songs_synthetic
    
    def generate_smote_like(self, X, y, subjects, songs, n_synthetic=None, k_neighbors=3): #5
        """Generate synthetic data using SMOTE-like interpolation"""
        print(f"\n🔧 Generating SMOTE-like synthetic data...")
        
        if n_synthetic is None:
            n_synthetic = len(X)
        
        nbrs = NearestNeighbors(n_neighbors=k_neighbors+1, algorithm='ball_tree').fit(X)
        distances, indices = nbrs.kneighbors(X)
        
        X_synthetic = []
        y_synthetic = []
        subjects_synthetic = []
        songs_synthetic = []
        
        for i in range(n_synthetic):
            base_idx = np.random.randint(0, len(X))
            neighbor_idx = np.random.choice(indices[base_idx][1:])
            
            alpha = np.random.random()
            x_new = X[base_idx] + alpha * (X[neighbor_idx] - X[base_idx])
            y_new = y[base_idx] + alpha * (y[neighbor_idx] - y[base_idx])
            
            X_synthetic.append(x_new)
            y_synthetic.append(y_new)
            subjects_synthetic.append(subjects[base_idx])
            songs_synthetic.append(songs[base_idx])
        
        X_synthetic = np.array(X_synthetic)
        y_synthetic = np.array(y_synthetic)
        y_synthetic = self.map_to_valid_labels(y_synthetic)
        
        print(f"   Generated {len(X_synthetic)} synthetic samples using SMOTE-like interpolation")
        return X_synthetic, y_synthetic, subjects_synthetic, songs_synthetic
    
    def generate_gaussian_mixture(self, X, y, subjects, songs, n_synthetic=None, n_components=3): #5
        """Generate synthetic data using Gaussian Mixture Models"""
        print(f"\n🔧 Generating GMM synthetic data...")
        
        if n_synthetic is None:
            n_synthetic = len(X)
        
        gmm = GaussianMixture(n_components=n_components, random_state=self.random_state)
        X_scaled = StandardScaler().fit_transform(X)
        gmm.fit(X_scaled)
        
        X_synthetic_scaled, _ = gmm.sample(n_synthetic)
        scaler = StandardScaler()
        scaler.fit(X)
        X_synthetic = scaler.inverse_transform(X_synthetic_scaled)
        
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(X)
        distances, indices = nbrs.kneighbors(X_synthetic)
        
        y_synthetic = y[indices.flatten()]
        subjects_synthetic = [subjects[i] for i in indices.flatten()]
        songs_synthetic = [songs[i] for i in indices.flatten()]
        
        y_synthetic = self.map_to_valid_labels(y_synthetic)
        
        print(f"   Generated {len(X_synthetic)} synthetic samples using GMM")
        return X_synthetic, y_synthetic, subjects_synthetic, songs_synthetic
    
    def generate_bootstrap_samples(self, X, y, subjects, songs, n_synthetic=None):
        """Generate synthetic data using bootstrap sampling with small perturbations"""
        print(f"\n🔧 Generating bootstrap synthetic data...")
        
        if n_synthetic is None:
            n_synthetic = len(X)
        
        bootstrap_indices = np.random.choice(len(X), size=n_synthetic, replace=True)
        
        X_synthetic = X[bootstrap_indices].copy()
        y_synthetic = y[bootstrap_indices].copy()
        subjects_synthetic = [subjects[i] for i in bootstrap_indices]
        songs_synthetic = [songs[i] for i in bootstrap_indices]
        
        feature_stds = np.std(X, axis=0)
        perturbations = np.random.normal(0, 0.05, X_synthetic.shape)
        perturbations = perturbations * feature_stds
        X_synthetic = X_synthetic + perturbations
        
        y_synthetic = self.map_to_valid_labels(y_synthetic)
        
        print(f"   Generated {len(X_synthetic)} synthetic samples using bootstrap")
        return X_synthetic, y_synthetic, subjects_synthetic, songs_synthetic
    
    def generate_rating_stratified(self, X, y, subjects, songs, n_synthetic_per_rating=None):
        """Generate synthetic data stratified by rating to balance classes"""
        print(f"\n🔧 Generating rating-stratified synthetic data...")
        
        unique_ratings = np.unique(y)
        print(f"   Original rating distribution:")
        rating_labels = {1: "Dislike", 3: "Medium", 5: "Like"}
        for rating in [1, 3, 5]:
            count = np.sum(y == rating)
            label = rating_labels[rating]
            print(f"   {label} ({rating}): {count} samples ({count/len(y)*100:.1f}%)")
        
        if n_synthetic_per_rating is None:
            max_count = max([np.sum(y == rating) for rating in unique_ratings])
            n_synthetic_per_rating = max_count
        
        X_synthetic_all = []
        y_synthetic_all = []
        subjects_synthetic_all = []
        songs_synthetic_all = []
        
        for rating in unique_ratings:
            rating_mask = y == rating
            X_rating = X[rating_mask]
            y_rating = y[rating_mask]
            subjects_rating = [subjects[i] for i in np.where(rating_mask)[0]]
            songs_rating = [songs[i] for i in np.where(rating_mask)[0]]
            
            current_count = len(X_rating)
            needed_count = max(0, n_synthetic_per_rating - current_count)
            
            if needed_count > 0:
                sample_indices = np.random.choice(len(X_rating), size=needed_count, replace=True)
                X_syn_rating = X_rating[sample_indices].copy()
                y_syn_rating = y_rating[sample_indices].copy()
                subjects_syn_rating = [subjects_rating[i] for i in sample_indices]
                songs_syn_rating = [songs_rating[i] for i in sample_indices]
                
                noise = np.random.normal(0, 0.1, X_syn_rating.shape)
                feature_stds = np.std(X_rating, axis=0)
                noise = noise * feature_stds
                X_syn_rating = X_syn_rating + noise
                
                y_syn_rating = self.map_to_valid_labels(y_syn_rating)
                
                X_synthetic_all.append(X_syn_rating)
                y_synthetic_all.append(y_syn_rating)
                subjects_synthetic_all.extend(subjects_syn_rating)
                songs_synthetic_all.extend(songs_syn_rating)
                
                print(f"   Generated {needed_count} samples for rating {rating}")
        
        if X_synthetic_all:
            X_synthetic = np.vstack(X_synthetic_all)
            y_synthetic = np.hstack(y_synthetic_all)
            print(f"   Total synthetic samples generated: {len(X_synthetic)}")
            return X_synthetic, y_synthetic, subjects_synthetic_all, songs_synthetic_all
        else:
            print(f"   No synthetic samples needed (data already balanced)")
            return np.array([]).reshape(0, X.shape[1]), np.array([]), [], []
    
    def combine_synthetic_data(self, original_data, synthetic_datasets):
        """Combine original data with multiple synthetic datasets"""
        print(f"\n🔗 Combining original and synthetic data...")
        
        X_orig, y_orig, subjects_orig, songs_orig, feature_names = original_data
        
        X_combined = [X_orig]
        y_combined = [y_orig]
        subjects_combined = [subjects_orig]
        songs_combined = [songs_orig]
        
        total_synthetic = 0
        for i, (X_syn, y_syn, subjects_syn, songs_syn) in enumerate(synthetic_datasets):
            if len(X_syn) > 0:
                X_combined.append(X_syn)
                y_combined.append(y_syn)
                subjects_combined.append(subjects_syn)
                songs_combined.append(songs_syn)
                total_synthetic += len(X_syn)
                print(f"   Added synthetic dataset {i+1}: {len(X_syn)} samples")
        
        X_final = np.vstack(X_combined)
        y_final = np.hstack(y_combined)
        subjects_final = sum(subjects_combined, [])
        songs_final = sum(songs_combined, [])
        
        print(f"\n📊 Final combined dataset:")
        print(f"   Original samples: {len(X_orig)}")
        print(f"   Synthetic samples: {total_synthetic}")
        print(f"   Total samples: {len(X_final)}")
        print(f"   Augmentation ratio: {total_synthetic/len(X_orig):.1f}x")
        
        print(f"\n📊 Final rating distribution:")
        unique_ratings, counts = np.unique(y_final, return_counts=True)
        rating_labels = {1: "Dislike", 3: "Medium", 5: "Like"}
        for rating, count in zip(unique_ratings, counts):
            label = rating_labels.get(int(rating), f"Rating {int(rating)}")
            print(f"   {label} ({int(rating)}): {count} samples ({count/len(y_final)*100:.1f}%)")
        
        return X_final, y_final, subjects_final, songs_final, feature_names
    
    def visualize_pca_comparison(self, X_orig, y_orig, X_synth, y_synth, save_dir=None):
        """Create PCA visualization comparing original and synthetic data"""
        print(f"\n📊 Creating PCA visualization...")
        
        try:
            # Set matplotlib backend to avoid GUI issues
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Combine data for PCA fitting
            X_combined = np.vstack([X_orig, X_synth])
            
            # Standardize features
            scaler = StandardScaler()
            X_combined_scaled = scaler.fit_transform(X_combined)
            
            # Apply PCA
            pca = PCA(n_components=2, random_state=42)
            X_pca = pca.fit_transform(X_combined_scaled)
            
            # Split back to original and synthetic
            X_orig_pca = X_pca[:len(X_orig)]
            X_synth_pca = X_pca[len(X_orig):]
            
            # Create visualization
            plt.figure(figsize=(15, 5))
            
            # Color mapping
            rating_colors = {1: 'red', 3: 'orange', 5: 'green'}
            rating_labels = {1: 'Dislike (1)', 3: 'Medium (3)', 5: 'Like (5)'}
            
            # Plot 1: Original data only
            plt.subplot(1, 3, 1)
            for rating in [1, 3, 5]:
                mask = y_orig == rating
                if np.any(mask):
                    plt.scatter(X_orig_pca[mask, 0], X_orig_pca[mask, 1], 
                               c=rating_colors[rating], label=rating_labels[rating], 
                               alpha=0.6, s=20)
            plt.title('Original Data\nPCA Visualization')
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot 2: Synthetic data only
            plt.subplot(1, 3, 2)
            for rating in [1, 3, 5]:
                mask = y_synth == rating
                if np.any(mask):
                    plt.scatter(X_synth_pca[mask, 0], X_synth_pca[mask, 1], 
                               c=rating_colors[rating], label=rating_labels[rating], 
                               alpha=0.6, s=20)
            plt.title('Synthetic Data\nPCA Visualization')
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot 3: Combined overlay
            plt.subplot(1, 3, 3)
            # Plot synthetic data first (background)
            for rating in [1, 3, 5]:
                mask = y_synth == rating
                if np.any(mask):
                    plt.scatter(X_synth_pca[mask, 0], X_synth_pca[mask, 1], 
                               c=rating_colors[rating], alpha=0.3, s=15, 
                               marker='s', label=f'Synthetic {rating_labels[rating]}')
            
            # Plot original data on top
            for rating in [1, 3, 5]:
                mask = y_orig == rating
                if np.any(mask):
                    plt.scatter(X_orig_pca[mask, 0], X_orig_pca[mask, 1], 
                               c=rating_colors[rating], alpha=0.8, s=25, 
                               marker='o', label=f'Original {rating_labels[rating]}', 
                               edgecolor='black', linewidth=0.5)
            
            plt.title('Original vs Synthetic\nPCA Comparison')
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, 'pca_comparison.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"   PCA visualization saved to: {save_path}")
            
            # Close the figure to free memory
            plt.close()
            
            # Print PCA information
            print(f"   PCA explained variance ratio:")
            print(f"     PC1: {pca.explained_variance_ratio_[0]:.1%}")
            print(f"     PC2: {pca.explained_variance_ratio_[1]:.1%}")
            print(f"     Total: {sum(pca.explained_variance_ratio_):.1%}")
            
            return pca, X_orig_pca, X_synth_pca
            
        except Exception as e:
            print(f"   Warning: PCA visualization failed: {e}")
            print(f"   Continuing without visualization...")
            return None, None, None
    
    def generate_all_synthetic_data(self, X, y, subjects, songs, feature_names, 
                                  synthetic_ratio=2.0, balance_ratings=True):
        """Generate synthetic data using multiple methods"""
        print(f"\n🎯 Generating synthetic data (target ratio: {synthetic_ratio}x)...")
        
        synthetic_datasets = []
        
        n_original = len(X)
        n_total_synthetic = int(n_original * synthetic_ratio)
        
        if balance_ratings:
            X_syn1, y_syn1, subj_syn1, songs_syn1 = self.generate_rating_stratified(X, y, subjects, songs)
            if len(X_syn1) > 0:
                synthetic_datasets.append((X_syn1, y_syn1, subj_syn1, songs_syn1))
        
        remaining_samples = max(0, n_total_synthetic - (len(X_syn1) if balance_ratings and len(X_syn1) > 0 else 0))
        samples_per_method = remaining_samples // 4
        
        if samples_per_method > 0:
            X_syn2, y_syn2, subj_syn2, songs_syn2 = self.generate_gaussian_noise(
                X, y, subjects, songs, n_synthetic=samples_per_method, noise_level=0.1
            )
            synthetic_datasets.append((X_syn2, y_syn2, subj_syn2, songs_syn2))
            
            X_syn3, y_syn3, subj_syn3, songs_syn3 = self.generate_smote_like(
                X, y, subjects, songs, n_synthetic=samples_per_method
            )
            synthetic_datasets.append((X_syn3, y_syn3, subj_syn3, songs_syn3))
            
            try:
                X_syn4, y_syn4, subj_syn4, songs_syn4 = self.generate_gaussian_mixture(
                    X, y, subjects, songs, n_synthetic=samples_per_method
                )
                synthetic_datasets.append((X_syn4, y_syn4, subj_syn4, songs_syn4))
            except Exception as e:
                print(f"   Warning: GMM generation failed: {e}")
            
            X_syn5, y_syn5, subj_syn5, songs_syn5 = self.generate_bootstrap_samples(
                X, y, subjects, songs, n_synthetic=samples_per_method
            )
            synthetic_datasets.append((X_syn5, y_syn5, subj_syn5, songs_syn5))
        
        X_combined, y_combined, subjects_combined, songs_combined, feature_names = self.combine_synthetic_data(
            (X, y, subjects, songs, feature_names), synthetic_datasets
        )
        
        return X_combined, y_combined, subjects_combined, songs_combined, feature_names
    
    def save_synthetic_data(self, X, y, subjects, songs, feature_names, scaler, save_path):
        """Save the combined synthetic dataset to pickle file"""
        print(f"\n💾 Saving synthetic dataset...")
        
        synthetic_data = {
            'X': X,
            'y': y,
            'subjects': subjects,
            'songs': songs,
            'feature_names': feature_names,
            'scaler': scaler
        }
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'wb') as f:
            pickle.dump(synthetic_data, f)
        
        print(f"✅ Synthetic dataset saved to: {save_path}")
        print(f"   Shape: {X.shape}")
        print(f"   Features: {len(feature_names)}")
        print(f"   Subjects: {len(np.unique(subjects))}")
        print(f"   Songs: {len(np.unique(songs))}")
        
        return save_path

def validate_synthetic_data(original_path, synthetic_path):
    """Validate that synthetic data has similar properties to original"""
    print(f"\n🔍 Validating synthetic data...")
    
    with open(original_path, 'rb') as f:
        orig_data = pickle.load(f)
    
    with open(synthetic_path, 'rb') as f:
        synth_data = pickle.load(f)
    
    X_orig, y_orig, subjects_orig, songs_orig = orig_data['X'], orig_data['y'], orig_data['subjects'], orig_data['songs']
    X_synth, y_synth, subjects_synth, songs_synth = synth_data['X'], synth_data['y'], synth_data['subjects'], synth_data['songs']
    
    print(f"📊 Data validation:")
    print(f"   Original: {X_orig.shape[0]} samples, {X_orig.shape[1]} features")
    print(f"   Synthetic: {X_synth.shape[0]} samples, {X_synth.shape[1]} features")
    print(f"   Augmentation ratio: {X_synth.shape[0]/X_orig.shape[0]:.1f}x")
    
    print(f"\n📊 Rating distribution comparison:")
    valid_labels = [1.0, 3.0, 5.0]
    unique_labels = np.unique(y_synth)
    if not np.all(np.isin(unique_labels, valid_labels)):
        print(f"⚠️ Invalid labels found in synthetic data: {unique_labels}")
        print(f"   Expected labels: {valid_labels}")
    rating_labels = {1: "Dislike", 3: "Medium", 5: "Like"}
    for rating in valid_labels:
        orig_count = np.sum(y_orig == rating)
        synth_count = np.sum(y_synth == rating)
        orig_pct = orig_count / len(y_orig) * 100 if len(y_orig) > 0 else 0
        synth_pct = synth_count / len(y_synth) * 100 if len(y_synth) > 0 else 0
        label = rating_labels[rating]
        print(f"   {label} ({rating}): Original {orig_pct:.1f}% vs Synthetic {synth_pct:.1f}%")
    
    print(f"\n📊 Subject distribution comparison:")
    orig_subj_counts = pd.Series(subjects_orig).value_counts()
    synth_subj_counts = pd.Series(subjects_synth).value_counts()
    print(f"   Original subjects: {len(orig_subj_counts)}")
    print(f"   Synthetic subjects: {len(synth_subj_counts)}")
    
    print(f"\n📊 Song distribution comparison:")
    orig_song_counts = pd.Series(songs_orig).value_counts()
    synth_song_counts = pd.Series(songs_synth).value_counts()
    print(f"   Original songs: {len(orig_song_counts)}")
    print(f"   Synthetic songs: {len(synth_song_counts)}")
    
    print(f"\n📊 Feature statistics comparison:")
    orig_mean = np.mean(X_orig, axis=0)
    synth_mean = np.mean(X_synth, axis=0)
    orig_std = np.std(X_orig, axis=0)
    synth_std = np.std(X_synth, axis=0)
    
    mean_correlation = np.corrcoef(orig_mean, synth_mean)[0, 1]
    std_correlation = np.corrcoef(orig_std, synth_std)[0, 1]
    
    print(f"   Feature means correlation: {mean_correlation:.3f}")
    print(f"   Feature stds correlation: {std_correlation:.3f}")
    
    if mean_correlation > 0.9 and std_correlation > 0.8 and np.all(np.isin(unique_labels, valid_labels)):
        print(f"✅ Synthetic data validation passed!")
    else:
        print(f"⚠️ Synthetic data may have different properties than original")
    
    return True

if __name__ == "__main__":
    original_features_path = '/home/anower/All/Python/Thesis_VS_code/full_epoch/Figure/eeg_features_full.pkl'
    synthetic_features_path = '/home/anower/All/Python/Thesis_VS_code/full_epoch/emon/eeg_features_full_synthetic.pkl'
    visualization_dir = '/home/anower/All/Python/Thesis_VS_code/full_epoch/emon/'
    
    SYNTHETIC_RATIO = 0.5
    BALANCE_RATINGS = True
    
    try:
        start_time = time.time()
        
        generator = SyntheticDataGenerator(random_state=42)
        
        X, y, subjects, songs, feature_names, scaler = generator.load_original_data(original_features_path)
        
        print(f"\n⏱️ Starting synthetic data generation at {time.strftime('%H:%M:%S')}")
        
        X_combined, y_combined, subjects_combined, songs_combined, feature_names = generator.generate_all_synthetic_data(
            X, y, subjects, songs, feature_names,
            synthetic_ratio=SYNTHETIC_RATIO,
            balance_ratings=BALANCE_RATINGS
        )
        
        # Extract synthetic data (excluding original data)
        n_original = len(X)
        X_synthetic_only = X_combined[n_original:]
        y_synthetic_only = y_combined[n_original:]
        
        # Create PCA visualization (with error handling)
        try:
            generator.visualize_pca_comparison(
                X, y, X_synthetic_only, y_synthetic_only, 
                save_dir=visualization_dir
            )
        except Exception as e:
            print(f"⚠️ PCA visualization failed: {e}")
            print("   Continuing without visualization...")
        
        generator.save_synthetic_data(
            X_combined, y_combined, subjects_combined, songs_combined, feature_names, scaler,
            synthetic_features_path
        )
        
        validate_synthetic_data(original_features_path, synthetic_features_path)
        
        generation_time = time.time() - start_time
        print(f"\n✅ Synthetic data generation complete! (Total time: {generation_time:.1f}s)")
        
        print(f"\n" + "="*70)
        print("SYNTHETIC DATA GENERATION SUMMARY")
        print("="*70)
        print(f"📊 Original dataset: {X.shape[0]} samples")
        print(f"📊 Synthetic dataset: {X_combined.shape[0]} samples")
        print(f"📊 Augmentation ratio: {X_combined.shape[0]/X.shape[0]:.1f}x")
        print(f"📊 Additional samples: {X_combined.shape[0] - X.shape[0]}")
        
        print(f"\n🎯 Benefits of synthetic data:")
        print(f"   ✅ Increased training data for better ML performance")
        print(f"   ✅ Better class balance for classification")
        print(f"   ✅ Reduced overfitting risk")
        print(f"   ✅ More robust model training")
        print(f"   ✅ PCA visualization shows data distribution quality")
        
        print(f"\n🔄 Next steps:")
        print(f"   1. Use '{synthetic_features_path}' in your ML model")
        print(f"   2. Compare performance with original data")
        print(f"   3. Analyze which synthetic methods work best")
        print(f"   4. Review PCA visualization in '{visualization_dir}'")
        
        print(f"\n💡 Tips for ML model:")
        print(f"   • Update features_path to use synthetic data")
        print(f"   • Monitor for overfitting with larger dataset")
        print(f"   • Consider separate validation on original data only")
        print(f"   • Compare results with and without synthetic data")
        print(f"   • Use PCA plots to understand data quality and overlap")
        
    except Exception as e:
        print(f"❌ Error during synthetic data generation: {str(e)}")
        import traceback
        traceback.print_exc()
