"""
Data Pipeline Module
Handles all data loading, feature engineering, and pipeline analysis
"""

import sys
import os
import pandas as pd
import numpy as np
import json
from collections import Counter
from sklearn.preprocessing import StandardScaler

# Add paths for imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE_DIR, 'data'))
sys.path.append(os.path.join(BASE_DIR, 'src'))

# Import required modules
from data.loader import load_and_engineer_features
from data.raw import download_fin_data, tickers
from src.models.dataset import TimeSeriesWindowDataset
from src.models.splitter import RollingWindowSplitter


def run_data_pipeline(config):
    """
    Complete data pipeline execution
    
    Args:
        config: Dictionary with pipeline configuration
        
    Returns:
        tuple: (data, feature_cols, target_col, pipeline_output) or (None, None, None, None) if failed
    """
    print("=" * 60)
    print("DATA PIPELINE")
    print("=" * 60)
    
    # Step 1: Load and prepare data
    data, feature_cols, target_col = load_and_prepare_data(config)
    
    if data is None:
        return None, None, None, None
    
    # Step 2: Create rolling windows analysis
    pipeline_info, total_train_sequences, total_test_sequences = create_rolling_windows_analysis(
        data, config, feature_cols, target_col
    )
    
    # Step 3: Generate pipeline summary
    pipeline_output = generate_pipeline_summary(
        pipeline_info, config, feature_cols, target_col, total_train_sequences, total_test_sequences
    )
    
    return data, feature_cols, target_col, pipeline_output


def load_and_prepare_data(config):
    """
    Load and engineer features
    
    Args:
        config: Dictionary with configuration including 'days' key
        
    Returns:
        tuple: (data, feature_cols, target_col) or (None, None, None) if failed
    """
    print("\nStep 1: Loading and engineering features...")
    
    try:
        raw_data = download_fin_data(tickers=tickers, lookback_days=config['days'])
        data, feature_cols, target_col = load_and_engineer_features(raw_data)
        
        print(f"Data loaded: {data.shape}")
        print(f"Features: {feature_cols}")
        print(f"Target: {target_col}")
        print(f"Data period: {data.index[0]} to {data.index[-1]}")
        
        return data, feature_cols, target_col
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None


def create_rolling_windows_analysis(data, config, feature_cols, target_col):
    """
    Create and analyze rolling windows
    
    Args:
        data: DataFrame with features and target
        config: Dictionary with configuration
        feature_cols: List of feature column names
        target_col: Target column name
        
    Returns:
        tuple: (pipeline_info, total_train_sequences, total_test_sequences)
    """
    print("\nStep 2: Creating rolling windows...")
    
    splitter = RollingWindowSplitter(
        data, 
        n_splits=config['n_splits'], 
        overlap_ratio=config['overlap_ratio'], 
        train_ratio=config['train_ratio']
    )
    
    pipeline_info = []
    total_train_sequences = 0
    total_test_sequences = 0
    
    for fold, (train_df, test_df, window_info) in enumerate(splitter.split()):
        print(f"\nWindow {fold}:")
        print(f"   Period: {window_info['start_date'].strftime('%Y-%m-%d')} to {window_info['end_date'].strftime('%Y-%m-%d')}")
        print(f"   Days: {window_info['total_days']} (train: {window_info['train_days']}, test: {window_info['test_days']})")
        
        # Separate features and target
        train_features = train_df[feature_cols]
        train_target = train_df[target_col]
        test_features = test_df[feature_cols]
        test_target = test_df[target_col]
        
        # Scale features
        scaler = StandardScaler()
        train_features_scaled = scaler.fit_transform(train_features)
        test_features_scaled = scaler.transform(test_features)
        
        # Create datasets
        train_dataset = TimeSeriesWindowDataset(
            pd.DataFrame(train_features_scaled, columns=feature_cols, index=train_features.index),
            train_target,
            seq_len=config['seq_len'],
            horizon=config['horizon']
        )
        
        test_dataset = TimeSeriesWindowDataset(
            pd.DataFrame(test_features_scaled, columns=feature_cols, index=test_features.index),
            test_target,
            seq_len=config['seq_len'],
            horizon=config['horizon']
        )
        
        class_weights = train_dataset.get_class_weights()
        
        print(f"    Train sequences: {len(train_dataset)}")
        print(f"    Test sequences: {len(test_dataset)}")
        
        total_train_sequences += len(train_dataset)
        total_test_sequences += len(test_dataset)
        
        pipeline_info.append({
            'fold': fold,
            'window_info': {
                'start_date': window_info['start_date'].isoformat(),
                'end_date': window_info['end_date'].isoformat(),
                'total_days': window_info['total_days'],
                'train_days': window_info['train_days'],
                'test_days': window_info['test_days']
            },
            'sequences': {
                'train': len(train_dataset),
                'test': len(test_dataset)
            },
            'class_weights': class_weights,
            'target_distribution': {
                'train': dict(train_target.value_counts().sort_index()),
                'test': dict(test_target.value_counts().sort_index())
            }
        })
    
    return pipeline_info, total_train_sequences, total_test_sequences


def generate_pipeline_summary(pipeline_info, config, feature_cols, target_col, 
                             total_train_sequences, total_test_sequences):
    """
    Generate and save pipeline summary
    
    Args:
        pipeline_info: List of pipeline information for each fold
        config: Dictionary with configuration
        feature_cols: List of feature column names
        target_col: Target column name
        total_train_sequences: Total number of training sequences
        total_test_sequences: Total number of test sequences
        
    Returns:
        dict: Pipeline output dictionary
    """
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    
    print(f"Dataset Statistics:")
    print(f"   Total training sequences: {total_train_sequences}")
    print(f"   Total test sequences: {total_test_sequences}")
    print(f"   Features: {len(feature_cols)}")
    print(f"   Sequence length: {config['seq_len']} days")
    print(f"   Prediction horizon: {config['horizon']} day")
    
    print(f"\nCross-Validation Setup:")
    print(f"   Number of windows: {config['n_splits']}")
    print(f"   Window overlap: {config['overlap_ratio']*100:.0f}%")
    print(f"   Train/test ratio: {config['train_ratio']*100:.0f}%/{(1-config['train_ratio'])*100:.0f}%")
    
    all_train_targets = []
    all_test_targets = []
    for info in pipeline_info:
        for class_id, count in info['target_distribution']['train'].items():
            all_train_targets.extend([class_id] * count)
        for class_id, count in info['target_distribution']['test'].items():
            all_test_targets.extend([class_id] * count)
    
    train_dist = Counter(all_train_targets)
    test_dist = Counter(all_test_targets)
    
    print(f"\nTarget Distribution (All Windows):")
    print(f"   Training: Down={train_dist.get(0,0)}, Neutral={train_dist.get(1,0)}, Up={train_dist.get(2,0)}")
    print(f"   Testing:  Down={test_dist.get(0,0)}, Neutral={test_dist.get(1,0)}, Up={test_dist.get(2,0)}")
    
    avg_weights = {}
    for class_id in [0, 1, 2]:
        weights = [info['class_weights'].get(float(class_id), 0) for info in pipeline_info]
        if weights:
            avg_weights[class_id] = np.mean([w for w in weights if w > 0])
    
    print(f"\nAverage Class Weights:")
    print(f"   Down (0): {avg_weights.get(0, 0):.2f}")
    print(f"   Neutral (1): {avg_weights.get(1, 0):.2f} <- Rare signals get higher weight!")
    print(f"   Up (2): {avg_weights.get(2, 0):.2f}")
    
    pipeline_output = save_pipeline_config(pipeline_info, config, feature_cols, target_col, 
                                         total_train_sequences, total_test_sequences, avg_weights)
    
    return pipeline_output


def save_pipeline_config(pipeline_info, config, feature_cols, target_col, 
                        total_train_sequences, total_test_sequences, avg_weights):
    """
    Save pipeline configuration to JSON file
    
    Args:
        pipeline_info: List of pipeline information for each fold
        config: Dictionary with configuration
        feature_cols: List of feature column names
        target_col: Target column name
        total_train_sequences: Total number of training sequences
        total_test_sequences: Total number of test sequences
        avg_weights: Dictionary of average class weights
        
    Returns:
        dict: Pipeline output dictionary
    """
    print("\nStep 4: Saving pipeline configuration...")
    
    os.makedirs('results', exist_ok=True)

    cleaned_pipeline_info = []
    for info in pipeline_info:
        cleaned_info = {
            'fold': info['fold'],
            'window_info': info['window_info'],
            'sequences': info['sequences'],
            'class_weights': {str(k): float(v) for k, v in info['class_weights'].items()},
            'target_distribution': {
                'train': {str(k): int(v) for k, v in info['target_distribution']['train'].items()},
                'test': {str(k): int(v) for k, v in info['target_distribution']['test'].items()}
            }
        }
        cleaned_pipeline_info.append(cleaned_info)

    # Fix avg_weights
    cleaned_avg_weights = {str(k): float(v) for k, v in avg_weights.items()}

    pipeline_output = {
        'config': config,
        'pipeline_summary': {
            'total_train_sequences': int(total_train_sequences),
            'total_test_sequences': int(total_test_sequences),
            'features': feature_cols,
            'target': target_col,
            'avg_class_weights': cleaned_avg_weights
        },
        'windows': cleaned_pipeline_info
    }

    with open('results/pipeline_config.json', 'w') as f:
        json.dump(pipeline_output, f, indent=2)

    print(f"Pipeline configuration saved to: results/pipeline_config.json")
    
    return pipeline_output