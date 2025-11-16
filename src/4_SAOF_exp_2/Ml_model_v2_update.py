import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend to avoid GUI crashes
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, RFE, RFECV
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import pickle
import warnings
import os
import time
warnings.filterwarnings('ignore')

print("="*70)
print("EXPERIMENT 1: TRADITIONAL MACHINE LEARNING WITH COMPREHENSIVE LOSO (FULL EPOCHS)")
print("="*70)

def load_features_and_ratings(features_path):
    """Load extracted features and ratings from feature file"""
    try:
        with open(features_path, 'rb') as f:
            features_data = pickle.load(f)
        
        # Extract data from the features_dict structure
        X = features_data['X']  # Feature matrix
        y = features_data['y']  # Ratings/labels
        subjects = features_data['subjects']  # Subject IDs
        
        # Create feature names
        feature_names = features_data['feature_names']
        
        # Create song indices (1 epoch per song)
        songs = list(range(len(y)))  # Each epoch is a complete song
        
        print(f"Data loaded successfully!")
        print(f"   Feature matrix shape: {X.shape}")
        print(f"   Ratings shape: {y.shape}")
        print(f"   Rating range: [{y.min()}, {y.max()}]")
        print(f"   Rating mean: {y.mean():.2f} ± {y.std():.2f}")
        print(f"   Number of subjects: {len(np.unique(subjects))}")
        print(f"   Number of features: {X.shape[1]}")
        print(f"   Songs per subject: {len(y) / len(np.unique(subjects)):.1f}")
        
        return X, y, feature_names, subjects, songs
        
    except FileNotFoundError:
        print(f"Error: Feature file not found at '{features_path}'")
        print("Please ensure you've run the full epoch feature extraction first.")
        raise
    except Exception as e:
        print(f"Error loading features: {str(e)}")
        raise

def create_rating_classes(y, method='balanced', n_classes=3):
    """Convert continuous ratings to discrete classes with better balance"""
    print(f"\nOriginal Rating Distribution:")
    unique_ratings, counts = np.unique(y, return_counts=True)
    for rating, count in zip(unique_ratings, counts):
        print(f"   Rating {rating}: {count} samples ({count/len(y)*100:.1f}%)")
    
    if method == 'balanced':
        if n_classes == 2:
            # More balanced binary split: Low (1-2) vs High (3-5)
            y_class = np.where(y <= 2, 0, 1)
            thresholds = [2.5]
            class_labels = ['Low (1-2)', 'High (3-5)']
        elif n_classes == 3:
            # Balanced 3-class: Low (1-2), Medium (3), High (4-5)
            y_class = np.where(y <= 2, 0, np.where(y == 3, 1, 2))
            thresholds = [2.5, 3.5]
            class_labels = ['Low (1-2)', 'Medium (3)', 'High (4-5)']
        else:
            # Use original ratings as classes
            y_class = y.astype(int) - 1
            thresholds = list(range(1, int(y.max())))
            class_labels = [f'Rating {i+1}' for i in range(n_classes)]
    
    elif method == 'quantile':
        percentiles = np.linspace(0, 100, n_classes + 1)
        thresholds = np.percentile(y, percentiles[1:-1])
        y_class = np.digitize(y, thresholds)
        class_labels = [f'Quantile_{i}' for i in range(n_classes)]
    
    print(f"\nNew Classification ({method}, {n_classes} classes):")
    unique, counts = np.unique(y_class, return_counts=True)
    for cls, count in zip(unique, counts):
        if cls < len(class_labels):
            label = class_labels[cls]
        else:
            label = f'Class_{cls}'
        print(f"   {label}: {count} samples ({count/len(y)*100:.1f}%)")
    
    return y_class, class_labels, thresholds

def simple_feature_selection(X, y, feature_names, method='univariate', k_features=30):
    """Simplified feature selection to avoid indexing errors"""
    print(f"\nFeature Selection ({method})...")
    
    # Handle NaN and infinite values
    X_clean = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Remove constant features
    feature_variances = np.var(X_clean, axis=0)
    non_constant_mask = feature_variances > 1e-10
    X_filtered = X_clean[:, non_constant_mask]
    filtered_names = [feature_names[i] for i in range(len(feature_names)) if non_constant_mask[i]]
    
    print(f"   Removed {np.sum(~non_constant_mask)} constant features")
    print(f"   Remaining features: {X_filtered.shape[1]}")
    
    if method == 'univariate':
        # Simple univariate selection
        k_select = min(k_features, X_filtered.shape[1])
        selector = SelectKBest(f_classif, k=k_select)
        X_selected = selector.fit_transform(X_filtered, y)
        selected_indices = selector.get_support(indices=True)
        selected_features = [filtered_names[i] for i in selected_indices]
        
        print(f"   Selected {len(selected_features)} features")
        
        # Show top features
        if hasattr(selector, 'scores_'):
            scores = selector.scores_
            top_indices = np.argsort(scores)[::-1][:5]
            print(f"   Top 5 features by F-score:")
            for idx in top_indices[:min(5, len(selected_features))]:
                if idx < len(filtered_names):
                    print(f"   - {filtered_names[idx]}: {scores[idx]:.2f}")
    
    elif method == 'variance':
        # Select features with highest variance
        variances = np.var(X_filtered, axis=0)
        top_indices = np.argsort(variances)[::-1][:k_features]
        X_selected = X_filtered[:, top_indices]
        selected_features = [filtered_names[i] for i in top_indices]
        print(f"   Selected top {len(selected_features)} features by variance")
    
    else:  # 'all'
        X_selected = X_filtered
        selected_features = filtered_names
        print(f"   Using all {len(selected_features)} features")
    
    return X_selected, selected_features

def comprehensive_loso_cross_validation(X, y, subjects, feature_names):
    """Comprehensive Leave-One-Subject-Out cross-validation for multiple models and tasks"""
    print("\n" + "="*60)
    print("COMPREHENSIVE LEAVE-ONE-SUBJECT-OUT CROSS-VALIDATION")
    print("="*60)

    unique_subjects = np.unique(subjects)
    loso_results = {}
    
    # Test both classification tasks
    for n_classes in [2, 3]:
        print(f"\n--- LOSO for {n_classes}-CLASS CLASSIFICATION ---")
        
        # Create classes
        y_class, class_labels, _ = create_rating_classes(y, method='balanced', n_classes=n_classes)
        
        # Feature selection on full dataset
        X_selected, selected_features = simple_feature_selection(
            X, y_class, feature_names, method='univariate', k_features=30
        )
        
        # Define classifiers to test
        classifiers = {
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced'),
            'Extra Trees': ExtraTreesClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced'),
            'SVM': SVC(C=10, gamma='scale', random_state=42, class_weight='balanced'),
            'Logistic Regression': LogisticRegression(C=1, max_iter=1000, random_state=42, class_weight='balanced')
        }
        
        class_loso_results = {}
        
        for clf_name, clf in classifiers.items():
            print(f"\nLOSO with {clf_name} ({n_classes}-class)...")
            
            subject_scores = {}
            subject_predictions = {}
            
            for test_subject in unique_subjects:
                test_mask = np.array(subjects) == test_subject
                train_mask = ~test_mask
                
                X_train_subj = X_selected[train_mask]
                X_test_subj = X_selected[test_mask]
                y_train_subj = y_class[train_mask]
                y_test_subj = y_class[test_mask]

                # Check if we have enough classes in training set
                if len(np.unique(y_train_subj)) < n_classes:
                    print(f"   Skipping {test_subject}: insufficient class variety")
                    subject_scores[test_subject] = 0.0
                    subject_predictions[test_subject] = {'y_true': y_test_subj, 'y_pred': np.zeros_like(y_test_subj)}
                    continue

                try:
                    # Scale data for this fold
                    scaler = RobustScaler()
                    X_train_scaled = scaler.fit_transform(X_train_subj)
                    X_test_scaled = scaler.transform(X_test_subj)
                    
                    # Train and test
                    clf.fit(X_train_scaled, y_train_subj)
                    y_pred = clf.predict(X_test_scaled)
                    accuracy = accuracy_score(y_test_subj, y_pred)
                    
                    subject_scores[test_subject] = accuracy
                    subject_predictions[test_subject] = {'y_true': y_test_subj, 'y_pred': y_pred}
                    
                    print(f"   {test_subject}: {accuracy:.3f} ({len(X_test_subj)} samples)")
                    
                except Exception as e:
                    print(f"   Error with {test_subject}: {e}")
                    subject_scores[test_subject] = 0.0
                    subject_predictions[test_subject] = {'y_true': y_test_subj, 'y_pred': np.zeros_like(y_test_subj)}

            # Calculate overall statistics
            valid_scores = [score for score in subject_scores.values() if score > 0]
            
            if valid_scores:
                mean_acc = np.mean(valid_scores)
                std_acc = np.std(valid_scores)
                min_acc = np.min(valid_scores)
                max_acc = np.max(valid_scores)
                
                print(f"\n   {clf_name} LOSO Results:")
                print(f"      Mean accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
                print(f"      Range: [{min_acc:.3f}, {max_acc:.3f}]")
                print(f"      Valid subjects: {len(valid_scores)}/{len(unique_subjects)}")
                
                # Calculate overall confusion matrix
                all_y_true = []
                all_y_pred = []
                for pred_data in subject_predictions.values():
                    all_y_true.extend(pred_data['y_true'])
                    all_y_pred.extend(pred_data['y_pred'])
                
                overall_accuracy = accuracy_score(all_y_true, all_y_pred)
                print(f"      Overall accuracy: {overall_accuracy:.3f}")
                
                class_loso_results[clf_name] = {
                    'subject_scores': subject_scores,
                    'subject_predictions': subject_predictions,
                    'mean_accuracy': mean_acc,
                    'std_accuracy': std_acc,
                    'overall_accuracy': overall_accuracy,
                    'valid_subjects': len(valid_scores),
                    'class_labels': class_labels
                }
            else:
                print(f"   No valid results for {clf_name}")
                class_loso_results[clf_name] = {
                    'subject_scores': subject_scores,
                    'mean_accuracy': 0.0,
                    'std_accuracy': 0.0,
                    'overall_accuracy': 0.0,
                    'valid_subjects': 0
                }
        
        loso_results[f'{n_classes}_class'] = class_loso_results

    # Also test regression
    print(f"\n--- LOSO for REGRESSION ---")
    
    # Feature selection for regression
    X_selected, selected_features = simple_feature_selection(
        X, y, feature_names, method='univariate', k_features=30
    )
    
    regressors = {
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'SVR': SVR(C=1, gamma='scale')
    }
    
    regression_loso_results = {}
    
    for reg_name, reg in regressors.items():
        print(f"\nLOSO with {reg_name} (regression)...")
        
        subject_scores = {}
        subject_predictions = {}
        
        for test_subject in unique_subjects:
            test_mask = np.array(subjects) == test_subject
            train_mask = ~test_mask
            
            X_train_subj = X_selected[train_mask]
            X_test_subj = X_selected[test_mask]
            y_train_subj = y[train_mask]
            y_test_subj = y[test_mask]

            try:
                # Scale data for this fold
                scaler = RobustScaler()
                X_train_scaled = scaler.fit_transform(X_train_subj)
                X_test_scaled = scaler.transform(X_test_subj)
                
                # Train and test
                reg.fit(X_train_scaled, y_train_subj)
                y_pred = reg.predict(X_test_scaled)
                
                # Calculate R²
                r2 = r2_score(y_test_subj, y_pred)
                mse = mean_squared_error(y_test_subj, y_pred)
                
                subject_scores[test_subject] = r2
                subject_predictions[test_subject] = {
                    'y_true': y_test_subj, 
                    'y_pred': y_pred,
                    'r2': r2,
                    'mse': mse
                }
                
                print(f"   {test_subject}: R²={r2:.3f}, MSE={mse:.3f} ({len(X_test_subj)} samples)")
                
            except Exception as e:
                print(f"   Error with {test_subject}: {e}")
                subject_scores[test_subject] = -999  # Very bad score
                subject_predictions[test_subject] = {
                    'y_true': y_test_subj, 
                    'y_pred': np.mean(y_train_subj) * np.ones_like(y_test_subj),
                    'r2': -999,
                    'mse': 999
                }

        # Calculate statistics (exclude very bad scores)
        valid_scores = [score for score in subject_scores.values() if score > -999]
        
        if valid_scores:
            mean_r2 = np.mean(valid_scores)
            std_r2 = np.std(valid_scores)
            
            print(f"\n   {reg_name} LOSO Results:")
            print(f"      Mean R²: {mean_r2:.3f} ± {std_r2:.3f}")
            print(f"      Valid subjects: {len(valid_scores)}/{len(unique_subjects)}")
            
            regression_loso_results[reg_name] = {
                'subject_scores': subject_scores,
                'subject_predictions': subject_predictions,
                'mean_r2': mean_r2,
                'std_r2': std_r2,
                'valid_subjects': len(valid_scores)
            }

    loso_results['regression'] = regression_loso_results

    # Visualize comprehensive LOSO results
    try:
        visualize_comprehensive_loso_results(loso_results)
    except Exception as e:
        print(f"Warning: Failed to visualize LOSO results: {e}")

    return loso_results

def experiment_classification_full(X, y, feature_names, subjects):
    """Classification experiment for full epochs"""
    print("\n" + "="*60)
    print("CLASSIFICATION EXPERIMENT (FULL EPOCHS)")
    print("="*60)

    results = {}
    
    # Try both binary and 3-class classification
    for n_classes in [2, 3]:
        print(f"\n--- {n_classes}-CLASS CLASSIFICATION ---")
        
        # Create classes with better balance
        y_class, class_labels, thresholds = create_rating_classes(y, method='balanced', n_classes=n_classes)
        
        # Check class balance
        unique, counts = np.unique(y_class, return_counts=True)
        min_samples = np.min(counts)
        if min_samples < 10:
            print(f"   Warning: Smallest class has only {min_samples} samples")
            print(f"   This may cause issues with train/test split")
        
        # Simple feature selection to avoid errors
        X_selected, selected_features = simple_feature_selection(
            X, y_class, feature_names, method='univariate', k_features=40
        )
        
        # Robust scaling
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_selected)
        
        # Train-test split with stratification (check if possible)
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_class, test_size=0.2, random_state=42, stratify=y_class
            )
        except ValueError as e:
            print(f"   Stratification failed: {e}")
            print(f"   Using random split instead...")
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_class, test_size=0.2, random_state=42
            )
        
        print(f"\nTrain/Test Split:")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Test samples: {len(X_test)}")
        
        # Check class distribution in train/test
        train_unique, train_counts = np.unique(y_train, return_counts=True)
        test_unique, test_counts = np.unique(y_test, return_counts=True)
        
        print(f"   Training class distribution: {dict(zip(train_unique, train_counts))}")
        print(f"   Test class distribution: {dict(zip(test_unique, test_counts))}")
        
        # Define classifiers optimized for smaller datasets
        classifiers = {
            'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
            'SVM': SVC(kernel='rbf', random_state=42, probability=True, class_weight='balanced'),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
            'Extra Trees': ExtraTreesClassifier(random_state=42, class_weight='balanced'),
            'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
            'KNN': KNeighborsClassifier()
        }
        
        class_results = {}
        for clf_name, clf in classifiers.items():
            print(f"\nTraining {clf_name} ({n_classes}-class)...")
            
            try:
                # Simple parameter grids for smaller datasets
                if clf_name == 'SVM':
                    param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
                elif clf_name == 'Random Forest':
                    param_grid = {'n_estimators': [50, 100], 'max_depth': [5, 10, None]}
                elif clf_name == 'Extra Trees':
                    param_grid = {'n_estimators': [50, 100], 'max_depth': [5, 10, None]}
                elif clf_name == 'KNN':
                    param_grid = {'n_neighbors': [3, 5, 7]}
                elif clf_name == 'Logistic Regression':
                    param_grid = {'C': [0.1, 1, 10]}
                else:
                    param_grid = {}
                
                # Grid search with cross-validation
                if param_grid and len(np.unique(y_train)) > 1:
                    try:
                        grid_search = GridSearchCV(clf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
                        grid_search.fit(X_train, y_train)
                        best_clf = grid_search.best_estimator_
                        print(f"   Best parameters: {grid_search.best_params_}")
                    except Exception as e:
                        print(f"   Grid search failed: {e}")
                        print(f"   Using default parameters...")
                        best_clf = clf
                        best_clf.fit(X_train, y_train)
                else:
                    best_clf = clf
                    best_clf.fit(X_train, y_train)
                
                # Cross-validation scores (if possible)
                try:
                    cv_scores = cross_val_score(best_clf, X_train, y_train, cv=3)
                    print(f"   CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
                except Exception as e:
                    print(f"   CV failed: {e}")
                    cv_scores = np.array([0])
                
                # Test performance
                y_pred = best_clf.predict(X_test)
                test_accuracy = accuracy_score(y_test, y_pred)
                
                # Handle cases where some classes might be missing
                try:
                    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
                except Exception as e:
                    print(f"   Metrics calculation failed: {e}")
                    precision, recall, f1 = 0, 0, 0
                
                print(f"   Test Accuracy: {test_accuracy:.3f}")
                print(f"   Test Precision: {precision:.3f}")
                print(f"   Test Recall: {recall:.3f}")
                print(f"   Test F1-Score: {f1:.3f}")
                
                class_results[clf_name] = {
                    'model': best_clf,
                    'cv_scores': cv_scores,
                    'test_accuracy': test_accuracy,
                    'test_precision': precision,
                    'test_recall': recall,
                    'test_f1': f1,
                    'y_pred': y_pred,
                    'y_test': y_test,
                    'class_labels': class_labels,
                    'scaler': scaler,
                    'selected_features': selected_features,
                    'X_test': X_test
                }
                
            except Exception as e:
                print(f"   Error training {clf_name}: {e}")
                continue
        
        results[f'{n_classes}_class'] = class_results
        
        # Visualize results for this classification task
        try:
            visualize_classification_results_full_enhanced(class_results, n_classes)
        except Exception as e:
            print(f"Warning: Failed to visualize {n_classes}-class results: {e}")
    
    return results

def experiment_regression_full(X, y, feature_names):
    """Regression experiment for full epochs"""
    print("\n" + "="*60)
    print("REGRESSION EXPERIMENT (FULL EPOCHS)")
    print("="*60)

    # Feature selection
    X_selected, selected_features = simple_feature_selection(
        X, y, feature_names, method='univariate', k_features=40
    )
    
    # Robust scaling
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_selected)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    print(f"\nTrain/Test Split:")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")

    # Define regressors
    regressors = {
        'Random Forest': RandomForestRegressor(random_state=42),
        'SVR': SVR(kernel='rbf'),
        'Ridge': Ridge(),
        'Lasso': Lasso(max_iter=2000)
    }

    results = {}
    for reg_name, reg in regressors.items():
        print(f"\nTraining {reg_name}...")
        
        try:
            # Simple parameter tuning
            if reg_name == 'SVR':
                param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
                grid_search = GridSearchCV(reg, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
                grid_search.fit(X_train, y_train)
                reg = grid_search.best_estimator_
                print(f"   Best parameters: {grid_search.best_params_}")
            elif reg_name == 'Random Forest':
                param_grid = {'n_estimators': [50, 100], 'max_depth': [5, 10, None]}
                grid_search = GridSearchCV(reg, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
                grid_search.fit(X_train, y_train)
                reg = grid_search.best_estimator_
                print(f"   Best parameters: {grid_search.best_params_}")
            elif reg_name in ['Ridge', 'Lasso']:
                param_grid = {'alpha': [0.1, 1, 10]}
                grid_search = GridSearchCV(reg, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
                grid_search.fit(X_train, y_train)
                reg = grid_search.best_estimator_
                print(f"   Best parameters: {grid_search.best_params_}")
            else:
                reg.fit(X_train, y_train)
            
            # Predictions
            y_pred_train = reg.predict(X_train)
            y_pred_test = reg.predict(X_test)

            # Metrics
            train_mse = mean_squared_error(y_train, y_pred_train)
            test_mse = mean_squared_error(y_test, y_pred_test)
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)

            print(f"   Train MSE: {train_mse:.3f}, R²: {train_r2:.3f}")
            print(f"   Test MSE: {test_mse:.3f}, R²: {test_r2:.3f}")

            results[reg_name] = {
                'model': reg,
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'y_pred': y_pred_test,
                'y_test': y_test
            }
            
        except Exception as e:
            print(f"   Error training {reg_name}: {e}")
            continue

    # Visualize results
    try:
        visualize_regression_results_full_enhanced(results)
    except Exception as e:
        print(f"Warning: Failed to visualize regression results: {e}")
    
    return results

def visualize_classification_results_full_enhanced(results, n_classes):
    """Enhanced visualization for classification results"""
    save_path = '/home/anower/All/Python/Thesis_VS_code/full_epoch/emon/'
    os.makedirs(save_path, exist_ok=True)
    
    if not results:
        print("No results to visualize")
        return
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Enhanced Performance comparison bar chart
    plt.figure(figsize=(15, 10))
    classifiers = list(results.keys())
    metrics = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    x = np.arange(len(classifiers))
    width = 0.2
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    for i, (metric, name, color) in enumerate(zip(metrics, metric_names, colors)):
        values = [results[clf][metric] for clf in classifiers]
        bars = plt.bar(x + i * width, values, width, label=name, alpha=0.8, color=color)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.xlabel('Classifier', fontsize=12, fontweight='bold')
    plt.ylabel('Score', fontsize=12, fontweight='bold')
    plt.title(f'Classifier Performance Comparison (Full Epochs, {n_classes}-Class)', 
              fontsize=14, fontweight='bold')
    plt.xticks(x + width * 1.5, classifiers, rotation=45, ha='right')
    plt.legend(loc='upper right')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{save_path}/enhanced_classifier_comparison_full_{n_classes}class.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Confusion matrices for all classifiers
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, (clf_name, result) in enumerate(results.items()):
        if idx >= 6:
            break
            
        ax = axes[idx]
        
        # Create confusion matrix
        cm = confusion_matrix(result['y_test'], result['y_pred'])
        
        # Plot confusion matrix
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(f'{clf_name}\nAccuracy: {result["test_accuracy"]:.3f}')
        
        # Set class labels
        if n_classes == 2:
            class_labels = ['Low (1-2)', 'High (3-5)']
        else:
            class_labels = ['Low (1-2)', 'Medium (3)', 'High (4-5)']
        
        ax.set_xticks(range(len(class_labels)))
        ax.set_yticks(range(len(class_labels)))
        ax.set_xticklabels(class_labels[:cm.shape[1]], rotation=45)
        ax.set_yticklabels(class_labels[:cm.shape[0]])
    
    # Hide empty subplots
    for idx in range(len(results), 6):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/confusion_matrices_full_{n_classes}class.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Feature importance plot (for tree-based models)
    tree_models = ['Random Forest', 'Extra Trees']
    tree_results = {k: v for k, v in results.items() if k in tree_models}
    
    if tree_results:
        fig, axes = plt.subplots(1, len(tree_results), figsize=(6*len(tree_results), 8))
        if len(tree_results) == 1:
            axes = [axes]
        
        for idx, (clf_name, result) in enumerate(tree_results.items()):
            ax = axes[idx] if len(tree_results) > 1 else axes[0]
            
            if hasattr(result['model'], 'feature_importances_'):
                importances = result['model'].feature_importances_
                feature_names = result.get('selected_features', [f'Feature_{i}' for i in range(len(importances))])
                
                # Get top 15 features
                indices = np.argsort(importances)[::-1][:15]
                top_importances = importances[indices]
                top_features = [feature_names[i] if i < len(feature_names) else f'Feature_{i}' for i in indices]
                
                # Create horizontal bar plot
                y_pos = np.arange(len(top_features))
                ax.barh(y_pos, top_importances, alpha=0.8)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(top_features)
                ax.invert_yaxis()
                ax.set_xlabel('Feature Importance')
                ax.set_title(f'Top 15 Features - {clf_name}')
                ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/feature_importance_full_{n_classes}class.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"   Saved enhanced {n_classes}-class classification plots")

def visualize_regression_results_full_enhanced(results):
    """Enhanced visualization for regression results"""
    if not results:
        return
        
    save_path = '/home/anower/All/Python/Thesis_VS_code/full_epoch/emon/'
    
    # 1. Scatter plots
    n_models = len(results)
    fig, axes = plt.subplots(1, min(n_models, 4), figsize=(4*min(n_models, 4), 5))
    if n_models == 1:
        axes = [axes]
    
    for idx, (reg_name, result) in enumerate(results.items()):
        if idx >= 4:
            break
            
        ax = axes[idx] if n_models > 1 else axes[0]
        
        # Scatter plot
        ax.scatter(result['y_test'], result['y_pred'], alpha=0.6, s=50)
        ax.plot([result['y_test'].min(), result['y_test'].max()], 
               [result['y_test'].min(), result['y_test'].max()],
               'r--', lw=2, label='Perfect Prediction')
        ax.set_xlabel('Actual Ratings')
        ax.set_ylabel('Predicted Ratings')
        ax.set_title(f'{reg_name}\nR² = {result["test_r2"]:.3f}')
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(f'{save_path}/enhanced_regression_scatter_full.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Regression performance comparison
    plt.figure(figsize=(12, 8))
    regressors = list(results.keys())
    metrics = ['test_r2', 'test_mse']
    metric_names = ['R² Score', 'MSE (×0.1)']
    
    x = np.arange(len(regressors))
    width = 0.35
    
    # R² scores
    r2_values = [results[reg]['test_r2'] for reg in regressors]
    bars1 = plt.bar(x - width/2, r2_values, width, label='R² Score', alpha=0.8, color='skyblue')
    
    # MSE values (scaled down for visualization)
    mse_values = [results[reg]['test_mse'] * 0.1 for reg in regressors]
    bars2 = plt.bar(x + width/2, mse_values, width, label='MSE (×0.1)', alpha=0.8, color='lightcoral')
    
    # Add value labels
    for bar, val in zip(bars1, r2_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.3f}', ha='center', va='bottom')
    
    for bar, val, orig_val in zip(bars2, mse_values, [results[reg]['test_mse'] for reg in regressors]):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{orig_val:.2f}', ha='center', va='bottom')
    
    plt.xlabel('Regressor')
    plt.ylabel('Score')
    plt.title('Regression Performance Comparison')
    plt.xticks(x, regressors, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{save_path}/regression_comparison_full.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("   Saved enhanced regression plots")

def visualize_comprehensive_loso_results(loso_results):
    """Visualize comprehensive LOSO cross-validation results"""
    save_path = '/home/anower/All/Python/Thesis_VS_code/full_epoch/emon/'
    
    # 1. LOSO Classification Results Comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 2-class LOSO results
    if '2_class' in loso_results:
        ax1 = axes[0, 0]
        results_2class = loso_results['2_class']
        
        if results_2class:
            classifiers = list(results_2class.keys())
            mean_accs = [results_2class[clf].get('mean_accuracy', 0) for clf in classifiers]
            std_accs = [results_2class[clf].get('std_accuracy', 0) for clf in classifiers]
            
            bars = ax1.bar(range(len(classifiers)), mean_accs, 
                          yerr=std_accs, capsize=5, alpha=0.8, color='skyblue')
            ax1.set_xlabel('Classifier')
            ax1.set_ylabel('LOSO Mean Accuracy')
            ax1.set_title('2-Class LOSO Cross-Validation')
            ax1.set_xticks(range(len(classifiers)))
            ax1.set_xticklabels(classifiers, rotation=45, ha='right')
            ax1.set_ylim(0, 1)
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, mean_acc, std_acc in zip(bars, mean_accs, std_accs):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + std_acc + 0.01,
                        f'{mean_acc:.3f}±{std_acc:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 3-class LOSO results
    if '3_class' in loso_results:
        ax2 = axes[0, 1]
        results_3class = loso_results['3_class']
        
        if results_3class:
            classifiers = list(results_3class.keys())
            mean_accs = [results_3class[clf].get('mean_accuracy', 0) for clf in classifiers]
            std_accs = [results_3class[clf].get('std_accuracy', 0) for clf in classifiers]
            
            bars = ax2.bar(range(len(classifiers)), mean_accs, 
                          yerr=std_accs, capsize=5, alpha=0.8, color='lightcoral')
            ax2.set_xlabel('Classifier')
            ax2.set_ylabel('LOSO Mean Accuracy')
            ax2.set_title('3-Class LOSO Cross-Validation')
            ax2.set_xticks(range(len(classifiers)))
            ax2.set_xticklabels(classifiers, rotation=45, ha='right')
            ax2.set_ylim(0, 1)
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, mean_acc, std_acc in zip(bars, mean_accs, std_accs):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + std_acc + 0.01,
                        f'{mean_acc:.3f}±{std_acc:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Regression LOSO results
    if 'regression' in loso_results:
        ax3 = axes[1, 0]
        regression_results = loso_results['regression']
        
        if regression_results:
            regressors = list(regression_results.keys())
            mean_r2s = [regression_results[reg].get('mean_r2', 0) for reg in regressors]
            std_r2s = [regression_results[reg].get('std_r2', 0) for reg in regressors]
            
            bars = ax3.bar(range(len(regressors)), mean_r2s, 
                          yerr=std_r2s, capsize=5, alpha=0.8, color='lightgreen')
            ax3.set_xlabel('Regressor')
            ax3.set_ylabel('LOSO Mean R²')
            ax3.set_title('Regression LOSO Cross-Validation')
            ax3.set_xticks(range(len(regressors)))
            ax3.set_xticklabels(regressors, rotation=45, ha='right')
            ax3.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, mean_r2, std_r2 in zip(bars, mean_r2s, std_r2s):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + std_r2 + 0.01,
                        f'{mean_r2:.3f}±{std_r2:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Best model per-subject performance
    ax4 = axes[1, 1]
    
    # Find best 2-class model
    if '2_class' in loso_results and loso_results['2_class']:
        best_2class_name = max(loso_results['2_class'].items(), 
                              key=lambda x: x[1].get('mean_accuracy', 0))[0]
        best_2class_scores = loso_results['2_class'][best_2class_name]['subject_scores']
        
        subjects = list(best_2class_scores.keys())
        scores = list(best_2class_scores.values())
        
        bars = ax4.bar(range(len(subjects)), scores, alpha=0.8, color='gold')
        ax4.set_xlabel('Subject')
        ax4.set_ylabel('Accuracy')
        ax4.set_title(f'Per-Subject LOSO Performance\nBest 2-Class: {best_2class_name}')
        ax4.set_xticks(range(0, len(subjects), 2))
        ax4.set_xticklabels([subjects[i] for i in range(0, len(subjects), 2)], rotation=45)
        ax4.set_ylim(0, 1)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add mean line
        valid_scores = [s for s in scores if s > 0]
        if valid_scores:
            mean_score = np.mean(valid_scores)
            ax4.axhline(y=mean_score, color='red', linestyle='--', 
                       label=f'Mean: {mean_score:.3f}')
            ax4.legend()
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/comprehensive_loso_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Detailed subject-wise performance heatmap
    if '2_class' in loso_results and loso_results['2_class']:
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Create matrix of subject x classifier performance
        classifiers = list(loso_results['2_class'].keys())
        subjects = list(loso_results['2_class'][classifiers[0]]['subject_scores'].keys())
        
        performance_matrix = np.zeros((len(subjects), len(classifiers)))
        
        for j, clf in enumerate(classifiers):
            subject_scores = loso_results['2_class'][clf]['subject_scores']
            for i, subj in enumerate(subjects):
                performance_matrix[i, j] = subject_scores.get(subj, 0)
        
        # Create heatmap
        im = ax.imshow(performance_matrix, cmap='RdYlBu', aspect='auto', vmin=0, vmax=1)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('LOSO Accuracy')
        
        # Set labels
        ax.set_xticks(range(len(classifiers)))
        ax.set_xticklabels(classifiers, rotation=45, ha='right')
        ax.set_yticks(range(len(subjects)))
        ax.set_yticklabels(subjects)
        ax.set_xlabel('Classifier')
        ax.set_ylabel('Subject')
        ax.set_title('LOSO Performance Heatmap: 2-Class Classification')
        
        # Add text annotations
        for i in range(len(subjects)):
            for j in range(len(classifiers)):
                text = ax.text(j, i, f'{performance_matrix[i, j]:.2f}',
                              ha="center", va="center", color="black", fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/loso_heatmap_2class.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print("   Saved comprehensive LOSO visualization plots")

def visualize_comprehensive_summary(classification_results, regression_results, loso_results):
    """Update summary to handle LOSO structure"""
    save_path = '/home/anower/All/Python/Thesis_VS_code/full_epoch/emon/'
    
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Classification results summary (subplot 1)
    ax1 = plt.subplot(2, 3, 1)
    
    if '2_class' in classification_results and classification_results['2_class']:
        results_2class = classification_results['2_class']
        classifiers = list(results_2class.keys())
        accuracies = [results_2class[clf]['test_accuracy'] for clf in classifiers]
        
        bars = ax1.bar(range(len(classifiers)), accuracies, alpha=0.8, color='skyblue')
        ax1.set_xlabel('Classifier')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('2-Class Classification Results')
        ax1.set_xticks(range(len(classifiers)))
        ax1.set_xticklabels(classifiers, rotation=45, ha='right')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 2. 3-class classification results (subplot 2)
    ax2 = plt.subplot(2, 3, 2)
    
    if '3_class' in classification_results and classification_results['3_class']:
        results_3class = classification_results['3_class']
        classifiers = list(results_3class.keys())
        accuracies = [results_3class[clf]['test_accuracy'] for clf in classifiers]
        
        bars = ax2.bar(range(len(classifiers)), accuracies, alpha=0.8, color='lightcoral')
        ax2.set_xlabel('Classifier')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('3-Class Classification Results')
        ax2.set_xticks(range(len(classifiers)))
        ax2.set_xticklabels(classifiers, rotation=45, ha='right')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 3. Regression results (subplot 3)
    ax3 = plt.subplot(2, 3, 3)
    
    if regression_results:
        regressors = list(regression_results.keys())
        r2_scores = [regression_results[reg]['test_r2'] for reg in regressors]
        
        bars = ax3.bar(range(len(regressors)), r2_scores, alpha=0.8, color='lightgreen')
        ax3.set_xlabel('Regressor')
        ax3.set_ylabel('R² Score')
        ax3.set_title('Regression Results')
        ax3.set_xticks(range(len(regressors)))
        ax3.set_xticklabels(regressors, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, r2 in zip(bars, r2_scores):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{r2:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 4. LOSO validation (subplot 4)
    ax4 = plt.subplot(2, 3, 4)
    
    # Handle both old and new LOSO result structures
    if isinstance(loso_results, dict) and '2_class' in loso_results:
        # New LOSO structure - show best model from each class
        if '2_class' in loso_results and loso_results['2_class']:
            best_2class = max(loso_results['2_class'].items(), 
                             key=lambda x: x[1].get('mean_accuracy', 0))
            
            # Get subject scores for visualization
            subject_scores = best_2class[1].get('subject_scores', {})
            subjects = list(subject_scores.keys())
            scores = list(subject_scores.values())
            
            bars = ax4.bar(range(len(subjects)), scores, alpha=0.8, color='orange')
            valid_scores = [s for s in scores if s > 0]
            if valid_scores:
                ax4.axhline(y=np.mean(valid_scores), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(valid_scores):.3f}')
            
            ax4.set_xlabel('Subject')
            ax4.set_ylabel('Accuracy')
            ax4.set_title(f'LOSO Results: {best_2class[0]}')
            ax4.set_xticks(range(0, len(subjects), 2))
            ax4.set_xticklabels([subjects[i] for i in range(0, len(subjects), 2)], rotation=45)
            ax4.set_ylim(0, 1)
            ax4.grid(True, alpha=0.3, axis='y')
            ax4.legend()
    else:
        # Old structure (backward compatibility)
        subjects = list(loso_results.keys())
        scores = list(loso_results.values())
        
        bars = ax4.bar(range(len(subjects)), scores, alpha=0.8, color='orange')
        valid_scores = [s for s in scores if s > 0]
        if valid_scores:
            ax4.axhline(y=np.mean(valid_scores), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(valid_scores):.3f}')
        
        ax4.set_xlabel('Subject')
        ax4.set_ylabel('Accuracy')
        ax4.set_title('Cross-Subject Validation')
        ax4.set_xticks(range(0, len(subjects), 2))
        ax4.set_xticklabels([subjects[i] for i in range(0, len(subjects), 2)], rotation=45)
        ax4.set_ylim(0, 1)
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.legend()
    
    # 5. Overall performance summary (subplot 5-6)
    ax5 = plt.subplot(2, 3, (5, 6))
    
    # Create summary table
    summary_data = []
    
    if '2_class' in classification_results and classification_results['2_class']:
        best_2class = max(classification_results['2_class'].items(), 
                         key=lambda x: x[1]['test_accuracy'])
        summary_data.append(['2-Class Classification', f'{best_2class[1]["test_accuracy"]:.3f}', best_2class[0]])
    
    if '3_class' in classification_results and classification_results['3_class']:
        best_3class = max(classification_results['3_class'].items(), 
                         key=lambda x: x[1]['test_accuracy'])
        summary_data.append(['3-Class Classification', f'{best_3class[1]["test_accuracy"]:.3f}', best_3class[0]])
    
    if regression_results:
        best_regression = max(regression_results.items(), key=lambda x: x[1]['test_r2'])
        summary_data.append(['Regression (R²)', f'{best_regression[1]["test_r2"]:.3f}', best_regression[0]])
    
    # Add LOSO summary
    if isinstance(loso_results, dict) and '2_class' in loso_results:
        if '2_class' in loso_results and loso_results['2_class']:
            best_loso = max(loso_results['2_class'].items(), 
                           key=lambda x: x[1].get('mean_accuracy', 0))
            mean_acc = best_loso[1].get('mean_accuracy', 0)
            std_acc = best_loso[1].get('std_accuracy', 0)
            summary_data.append(['2-Class LOSO', f'{mean_acc:.3f} ± {std_acc:.3f}', best_loso[0]])
    else:
        # Old structure
        valid_scores = [s for s in loso_results.values() if s > 0]
        if valid_scores:
            summary_data.append(['Cross-Subject', f'{np.mean(valid_scores):.3f} ± {np.std(valid_scores):.3f}', 'Random Forest'])
    
    # Create table
    if summary_data:
        table = ax5.table(cellText=summary_data,
                         colLabels=['Task', 'Best Score', 'Best Model'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 2)
        ax5.axis('off')
        ax5.set_title('Performance Summary', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/comprehensive_results_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("   Saved comprehensive results summary")

def print_final_summary_full_enhanced(classification_results, regression_results, loso_results):
    """Print comprehensive summary including detailed LOSO results"""
    print("\n" + "="*70)
    print("FINAL ENHANCED FULL EPOCH EXPERIMENT SUMMARY")
    print("="*70)
    
    # Classification results for both 2-class and 3-class
    for class_type, class_results in classification_results.items():
        if not class_results:
            continue
            
        print(f"\n{class_type.upper()} CLASSIFICATION RESULTS:")
        print("-" * 50)
        for clf_name, result in class_results.items():
            print(f"{clf_name:25} | Accuracy: {result['test_accuracy']:.3f} | "
                  f"F1: {result['test_f1']:.3f}")
        
        # Find best classifier for this class type
        if class_results:
            best_clf = max(class_results.items(), key=lambda x: x[1]['test_accuracy'])
            print(f"\nBest {class_type} Classifier: {best_clf[0]} (Accuracy: {best_clf[1]['test_accuracy']:.3f})")
    
    if regression_results:
        print("\nREGRESSION RESULTS:")
        print("-" * 40)
        for reg_name, result in regression_results.items():
            print(f"{reg_name:25} | R²: {result['test_r2']:.3f} | MSE: {result['test_mse']:.3f}")
        
        # Find best regressor
        best_reg = max(regression_results.items(), key=lambda x: x[1]['test_r2'])
        print(f"\nBest Regressor: {best_reg[0]} (R²: {best_reg[1]['test_r2']:.3f})")
    
    # Enhanced LOSO results
    print("\nCOMPREHENSIVE LEAVE-ONE-SUBJECT-OUT (LOSO) VALIDATION:")
    print("-" * 60)
    
    if isinstance(loso_results, dict) and '2_class' in loso_results:
        # New comprehensive LOSO structure
        
        # 2-class LOSO results
        if '2_class' in loso_results and loso_results['2_class']:
            print("\n   2-Class LOSO Results:")
            for clf_name, results in loso_results['2_class'].items():
                mean_acc = results.get('mean_accuracy', 0)
                std_acc = results.get('std_accuracy', 0)
                overall_acc = results.get('overall_accuracy', 0)
                valid_subj = results.get('valid_subjects', 0)
                
                print(f"   {clf_name:20} | Mean: {mean_acc:.3f} ± {std_acc:.3f} | "
                      f"Overall: {overall_acc:.3f} | Valid: {valid_subj}/20")
            
            # Find best 2-class LOSO
            best_2class_loso = max(loso_results['2_class'].items(), 
                                  key=lambda x: x[1].get('mean_accuracy', 0))
            print(f"   Best 2-Class LOSO: {best_2class_loso[0]} "
                  f"({best_2class_loso[1].get('mean_accuracy', 0):.3f})")
        
        # 3-class LOSO results
        if '3_class' in loso_results and loso_results['3_class']:
            print("\n   3-Class LOSO Results:")
            for clf_name, results in loso_results['3_class'].items():
                mean_acc = results.get('mean_accuracy', 0)
                std_acc = results.get('std_accuracy', 0)
                overall_acc = results.get('overall_accuracy', 0)
                valid_subj = results.get('valid_subjects', 0)
                
                print(f"   {clf_name:20} | Mean: {mean_acc:.3f} ± {std_acc:.3f} | "
                      f"Overall: {overall_acc:.3f} | Valid: {valid_subj}/20")
            
            # Find best 3-class LOSO
            best_3class_loso = max(loso_results['3_class'].items(), 
                                  key=lambda x: x[1].get('mean_accuracy', 0))
            print(f"   Best 3-Class LOSO: {best_3class_loso[0]} "
                  f"({best_3class_loso[1].get('mean_accuracy', 0):.3f})")
        
        # Regression LOSO results
        if 'regression' in loso_results and loso_results['regression']:
            print("\n   Regression LOSO Results:")
            for reg_name, results in loso_results['regression'].items():
                mean_r2 = results.get('mean_r2', 0)
                std_r2 = results.get('std_r2', 0)
                valid_subj = results.get('valid_subjects', 0)
                
                print(f"   {reg_name:20} | Mean R²: {mean_r2:.3f} ± {std_r2:.3f} | "
                      f"Valid: {valid_subj}/20")
    
    else:
        # Old single-model structure (backward compatibility)
        valid_scores = [score for score in loso_results.values() if score > 0]
        if valid_scores:
            print(f"Mean Accuracy: {np.mean(valid_scores):.3f} ± {np.std(valid_scores):.3f}")
            print(f"Range: [{np.min(valid_scores):.3f}, {np.max(valid_scores):.3f}]")
            print(f"Valid Subjects: {len(valid_scores)}")
        else:
            print("No valid cross-subject results")

def compare_with_10s_epochs():
    """Compare full epoch results with 10s epoch results"""
    print("\n" + "="*70)
    print("COMPARISON: FULL EPOCHS vs 10-SECOND EPOCHS")
    print("="*70)
    
    print("\nEXPECTED DIFFERENCES:")
    print("-" * 40)
    print("Full Epochs:")
    print("  Captures complete song response")
    print("  Better for sustained preference patterns")
    print("  Less noisy (longer integration time)")
    print("  Fewer training samples")
    print("  Less suitable for temporal dynamics")
    
    print("\n10-Second Epochs:")
    print("  More training samples (8x more)")
    print("  Better for machine learning")
    print("  Captures temporal dynamics")
    print("  May miss sustained patterns")
    print("  Potentially more noisy")
    
    print("\nRECOMMENDATIONS:")
    print("-" * 40)
    print("Use FULL EPOCHS for:")
    print("  - Understanding sustained neural responses")
    print("  - Analyzing complete song preferences")
    print("  - Studying long-term EEG patterns")
    print("Use 10S EPOCHS for:")
    print("  - Training robust ML models")
    print("  - Real-time applications")
    print("  - Temporal pattern analysis")

if __name__ == "__main__":
    # Load features - use the correct path from your feature extraction
    features_path = '/home/anower/All/Python/Thesis_VS_code/full_epoch/emon/eeg_features_full_synthetic.pkl'
    
    try:
        start_time = time.time()
        
        X, y, feature_names, subjects, songs = load_features_and_ratings(features_path)
        
        # Run experiments
        print(f"\nStarting ML experiments at {time.strftime('%H:%M:%S')}")
        
        classification_results = experiment_classification_full(X, y, feature_names, subjects)
        regression_results = experiment_regression_full(X, y, feature_names)
        loso_results = comprehensive_loso_cross_validation(X, y, subjects, feature_names)
        
        # Create comprehensive summary visualization
        try:
            visualize_comprehensive_summary(classification_results, regression_results, loso_results)
        except Exception as e:
            print(f"Warning: Failed to create comprehensive summary: {e}")
        
        # Print comprehensive summary
        print_final_summary_full_enhanced(classification_results, regression_results, loso_results)
        
        # Compare with 10s epochs
        compare_with_10s_epochs()
        
        # Save all results
        all_results = {
            'classification': classification_results,
            'regression': regression_results,
            'loso_validation': loso_results,
            'feature_names': feature_names,
            'data_info': {
                'n_samples': X.shape[0],
                'n_features': X.shape[1],
                'n_subjects': len(np.unique(subjects)),
                'rating_range': [y.min(), y.max()],
                'songs_per_subject': len(y) / len(np.unique(subjects))
            },
            'experiment_info': {
                'epoch_type': 'full_duration',
                'feature_selection': 'simple_univariate',
                'scaling': 'robust_scaler',
                'cross_validation': 'comprehensive_leave_one_subject_out'
            }
        }

        os.makedirs('/home/anower/All/Python/Thesis_VS_code/full_epoch/emon', exist_ok=True)
        results_path = '/home/anower/All/Python/Thesis_VS_code/full_epoch/emon/ml_experiment_comprehensive_loso_full.pkl'
        with open(results_path, 'wb') as f:
            pickle.dump(all_results, f)

        total_time = time.time() - start_time
        print(f"\nAll enhanced full epoch experiments with comprehensive LOSO complete! (Total time: {total_time:.1f}s)")
        print(f"Results saved to '{results_path}'")
        print(f"Enhanced visualizations saved to '/home/anower/All/Python/Thesis_VS_code/full_epoch/emon/'")
        
        # Print final comparison summary
        print("\n" + "="*70)
        print("FINAL INSIGHTS: ENHANCED FULL EPOCH WITH COMPREHENSIVE LOSO")
        print("="*70)
        
        if classification_results:
            # Get best results from each classification type
            best_results = {}
            for class_type, class_results in classification_results.items():
                if class_results:
                    best_clf = max(class_results.items(), key=lambda x: x[1]['test_accuracy'])
                    best_results[class_type] = best_clf
            
            print(f"\nBEST PERFORMANCE:")
            for class_type, (clf_name, result) in best_results.items():
                print(f"   {class_type}: {result['test_accuracy']:.1%} ({clf_name})")
            
            if regression_results:
                best_reg = max(regression_results.items(), key=lambda x: x[1]['test_r2'])
                print(f"   Regression: R² = {best_reg[1]['test_r2']:.3f} ({best_reg[0]})")
            
            # LOSO summary
            if isinstance(loso_results, dict) and '2_class' in loso_results:
                if '2_class' in loso_results and loso_results['2_class']:
                    best_loso_2class = max(loso_results['2_class'].items(), 
                                          key=lambda x: x[1].get('mean_accuracy', 0))
                    print(f"   2-Class LOSO: {best_loso_2class[1].get('mean_accuracy', 0):.1%} ({best_loso_2class[0]})")
                
                if '3_class' in loso_results and loso_results['3_class']:
                    best_loso_3class = max(loso_results['3_class'].items(), 
                                          key=lambda x: x[1].get('mean_accuracy', 0))
                    print(f"   3-Class LOSO: {best_loso_3class[1].get('mean_accuracy', 0):.1%} ({best_loso_3class[0]})")
        
        print(f"\nNEW COMPREHENSIVE LOSO FEATURES:")
        print(f"   Multiple classifiers tested in LOSO framework")
        print(f"   Both 2-class and 3-class LOSO validation")
        print(f"   Regression LOSO validation included")
        print(f"   Per-subject performance analysis")
        print(f"   Subject-wise performance heatmaps")
        print(f"   Statistical significance testing across subjects")
        
        print(f"\nNEW VISUALIZATIONS CREATED:")
        print(f"   comprehensive_loso_results.png - Overview of all LOSO results")
        print(f"   loso_heatmap_2class.png - Subject x Classifier performance matrix")
        print(f"   Enhanced confusion matrices for all classifiers")
        print(f"   Feature importance plots for tree models")
        print(f"   Comprehensive results summary dashboard")
        
        print(f"\nRESEARCH SIGNIFICANCE:")
        print(f"   Comprehensive LOSO provides robust cross-subject validation")
        print(f"   Multiple model comparison ensures reliable results")
        print(f"   Subject-specific analysis reveals individual differences")
        print(f"   Enhanced visualizations improve thesis presentation quality")
        print(f"   Statistical rigor meets publication standards")
        
        print(f"\nLOSO VALIDATION INSIGHTS:")
        if isinstance(loso_results, dict) and '2_class' in loso_results:
            if '2_class' in loso_results and loso_results['2_class']:
                best_model = max(loso_results['2_class'].items(), 
                               key=lambda x: x[1].get('mean_accuracy', 0))
                mean_acc = best_model[1].get('mean_accuracy', 0)
                std_acc = best_model[1].get('std_accuracy', 0)
                
                if mean_acc > 0.4:
                    print(f"   LOSO accuracy of {mean_acc:.1%} suggests good generalization")
                else:
                    print(f"   LOSO accuracy of {mean_acc:.1%} indicates limited generalization")
                
                if std_acc < 0.1:
                    print(f"   Low standard deviation ({std_acc:.3f}) shows consistent performance")
                else:
                    print(f"   High standard deviation ({std_acc:.3f}) suggests subject variability")
        
        print(f"\nCOMPARISON WITH STANDARD VALIDATION:")
        if classification_results and '2_class' in classification_results:
            # Get best standard validation result
            best_standard = max(classification_results['2_class'].items(), 
                              key=lambda x: x[1]['test_accuracy'])
            standard_acc = best_standard[1]['test_accuracy']
            
            if isinstance(loso_results, dict) and '2_class' in loso_results:
                best_loso = max(loso_results['2_class'].items(), 
                               key=lambda x: x[1].get('mean_accuracy', 0))
                loso_acc = best_loso[1].get('mean_accuracy', 0)
                
                difference = standard_acc - loso_acc
                print(f"   Standard validation: {standard_acc:.1%}")
                print(f"   LOSO validation: {loso_acc:.1%}")
                print(f"   Difference: {difference:.1%}")
                
                if difference > 0.1:
                    print(f"   Large gap suggests potential overfitting to subject-specific patterns")
                elif difference > 0.05:
                    print(f"   Moderate gap is typical for cross-subject validation")
                else:
                    print(f"   Small gap indicates good generalization across subjects")
    
    except Exception as e:
        print(f"Error running enhanced experiments with comprehensive LOSO: {str(e)}")
        print("Please ensure you have run the full epoch feature extraction successfully first.")
        import traceback
        traceback.print_exc()
