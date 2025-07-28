"""
Training Pipeline Module
Handles LSTM model training and evaluation
"""

import sys
import os
import glob

# Add paths for imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE_DIR, 'src'))

# Import LSTM training function
from src.models.LSTM import run_lstm_training, DEFAULT_CONFIG


def get_next_version_number(config_name, results_dir='results'):
    """
    Get the next version number for a given config
    
    Args:
        config_name: Configuration name
        results_dir: Directory to check for existing files
        
    Returns:
        int: Next version number
    """
    if not os.path.exists(results_dir):
        return 1
    
    # Look for existing training result files with this config name
    pattern = os.path.join(results_dir, f'lstm_training_{config_name}_v*.json')
    existing_files = glob.glob(pattern)
    
    if not existing_files:
        return 1
    
    # Extract version numbers and find the highest
    version_numbers = []
    for filepath in existing_files:
        filename = os.path.basename(filepath)
        try:
            # Extract version number from filename like "lstm_training_heavy_model_v3.json"
            version_part = filename.split('_v')[1].split('.')[0]
            version_numbers.append(int(version_part))
        except (IndexError, ValueError):
            continue
    
    return max(version_numbers) + 1 if version_numbers else 1


def run_training_pipeline(data, feature_cols, target_col, training_config, config_name):
    """
    Complete LSTM training pipeline
    
    Args:
        data: DataFrame with features and target
        feature_cols: List of feature column names
        target_col: Target column name
        training_config: Dictionary with training configuration
        config_name: Configuration name for file naming
        
    Returns:
        dict: Training results or None if failed
    """
    print("=" * 60)
    print("TRAINING PIPELINE")
    print("=" * 60)
    
    print(f"Configuration: {config_name}")
    print(f"Training parameters: {training_config}")
    
    # Get version number for this config
    version = get_next_version_number(config_name)
    print(f"Saving results as version {version} for config '{config_name}'")
    
    # Add config info to training config for the LSTM module to use
    enhanced_training_config = training_config.copy()
    enhanced_training_config['config_name'] = config_name
    enhanced_training_config['version'] = version
    
    # Run LSTM training using the imported function
    lstm_results = run_lstm_training(data, feature_cols, target_col, enhanced_training_config)
    
    # Add config metadata to results
    if lstm_results:
        lstm_results['config_name'] = config_name
        lstm_results['version'] = version
        
        print(f"Training completed for {config_name} v{version}")
        
        # Print quick summary
        summary = lstm_results.get('summary', {})
        if summary:
            accuracy = summary.get('avg_accuracy', 0)
            improvement = summary.get('improvement_over_random', 0)
            print(f"Results: {accuracy:.1%} accuracy ({improvement:+.1f}% vs random)")
    
    return lstm_results


def prepare_training_config(pipeline_config, custom_training_config=None):
    """
    Prepare training configuration by merging pipeline config with training config
    
    Args:
        pipeline_config: Dictionary with pipeline configuration
        custom_training_config: Optional custom training configuration
        
    Returns:
        dict: Complete training configuration
    """
    # Start with default LSTM config
    training_config = DEFAULT_CONFIG.copy()
    
    # Update with pipeline-specific settings
    training_config.update({
        'seq_len': pipeline_config['seq_len'],
        'horizon': pipeline_config['horizon'],
        'n_splits': pipeline_config['n_splits'],
        'overlap_ratio': pipeline_config['overlap_ratio'],
        'train_ratio': pipeline_config['train_ratio'],
        'tickers': pipeline_config['tickers'],
        'days': pipeline_config['days']
    })
    
    # Apply any custom overrides
    if custom_training_config:
        training_config.update(custom_training_config)
    
    return training_config


def get_training_recommendations(lstm_results):
    """
    Analyze training results and provide recommendations
    
    Args:
        lstm_results: Dictionary with training results
        
    Returns:
        dict: Recommendations for improving the model
    """
    if not lstm_results:
        return {
            'status': 'failed',
            'recommendations': ['Check data pipeline and model configuration']
        }
    
    summary = lstm_results.get('summary', {})
    avg_accuracy = summary.get('avg_accuracy', 0)
    improvement = summary.get('improvement_over_random', 0)
    config_name = lstm_results.get('config_name', 'unknown')
    version = lstm_results.get('version', 1)
    
    recommendations = []
    
    # Performance-based recommendations
    if avg_accuracy < 0.35:  # Less than 35% accuracy (barely above random)
        recommendations.extend([
            f'Low accuracy for {config_name} v{version}. Try:',
            '- Switch to light_model or heavy_model config',
            '- Increase data size (more days)',
            '- Check feature engineering quality',
            '- Try binary classification instead of 3-class'
        ])
    elif avg_accuracy < 0.4:  # 35-40% accuracy
        recommendations.extend([
            f'Moderate accuracy for {config_name} v{version}. Consider:',
            '- Fine-tuning hyperparameters',
            '- Adding more features',
            '- Increasing model complexity',
            '- More training epochs'
        ])
    elif avg_accuracy < 0.5:  # 40-50% accuracy
        recommendations.extend([
            f'Good progress with {config_name} v{version}. Try:',
            '- Ensemble methods',
            '- Feature importance analysis',
            '- Different sequence lengths',
            '- Advanced architectures (Transformer, CNN-LSTM)'
        ])
    else:  # 50%+ accuracy
        recommendations.extend([
            f'Excellent results with {config_name} v{version}!',
            '- Analyze feature importance',
            '- Test on out-of-sample data',
            '- Develop trading strategies',
            '- Consider production deployment'
        ])
    
    # Config-specific recommendations
    if config_name == 'quick_test':
        recommendations.append('Try development or production config for better results')
    elif config_name == 'light_model' and avg_accuracy > 0.4:
        recommendations.append('Good results with light model - try heavy_model for potential improvement')
    elif config_name == 'heavy_model' and avg_accuracy < 0.4:
        recommendations.append('Heavy model underperforming - try light_model to reduce overfitting')
    
    # Check for overfitting signs
    fold_results = lstm_results.get('fold_results', [])
    if fold_results and len(fold_results) > 1:
        accuracies = [r['accuracy'] for r in fold_results]
        accuracy_range = max(accuracies) - min(accuracies)
        
        if accuracy_range > 0.3:  # High variance between folds
            recommendations.append(f'High variance ({accuracy_range:.1%}) between folds - reduce model complexity')
        elif accuracy_range < 0.1:  # Very consistent
            recommendations.append(f'Very consistent results ({accuracy_range:.1%}) - model is stable')
    
    # Improvement analysis
    if improvement < 5:
        recommendations.append('Model barely beats random - fundamental changes needed')
    elif improvement > 20:
        recommendations.append('Strong improvement over random - model found real patterns')
    
    # Determine overall status
    if avg_accuracy > 0.45 and improvement > 15:
        status = 'success'
    elif avg_accuracy > 0.37 or improvement > 8:
        status = 'needs_improvement'
    else:
        status = 'failed'
    
    return {
        'status': status,
        'avg_accuracy': avg_accuracy,
        'improvement_over_random': improvement,
        'config_name': config_name,
        'version': version,
        'recommendations': recommendations
    }


def compare_with_previous_versions(config_name, current_accuracy):
    """
    Compare current results with previous versions of the same config
    
    Args:
        config_name: Configuration name
        current_accuracy: Current model accuracy
        
    Returns:
        dict: Comparison results
    """
    results_dir = 'results'
    if not os.path.exists(results_dir):
        return {'previous_versions': 0, 'best_previous': None, 'improvement': None}
    
    # Look for previous versions
    pattern = os.path.join(results_dir, f'lstm_training_{config_name}_v*.json')
    existing_files = glob.glob(pattern)
    
    if len(existing_files) <= 1:  # Only current version exists
        return {'previous_versions': 0, 'best_previous': None, 'improvement': None}
    
    # Analyze previous versions
    previous_results = []
    for filepath in existing_files:
        try:
            import json
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            summary = data.get('summary', {})
            accuracy = summary.get('avg_accuracy', 0)
            version = data.get('version', 1)
            
            previous_results.append({
                'version': version,
                'accuracy': accuracy
            })
        except Exception:
            continue
    
    if not previous_results:
        return {'previous_versions': 0, 'best_previous': None, 'improvement': None}
    
    # Find best previous version (excluding current)
    previous_results.sort(key=lambda x: x['version'])
    best_previous = max(previous_results[:-1], key=lambda x: x['accuracy']) if len(previous_results) > 1 else None
    
    improvement = None
    if best_previous:
        improvement = current_accuracy - best_previous['accuracy']
    
    return {
        'previous_versions': len(previous_results) - 1,
        'best_previous': best_previous,
        'improvement': improvement
    }