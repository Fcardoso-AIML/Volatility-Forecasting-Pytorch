"""
VIX Direction Prediction - Main
"""
import sys
import os
import json
import glob
from datetime import datetime

# Add paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'src'))
sys.path.append(os.path.join(BASE_DIR, 'config'))

# Import pipeline modules
from src.pipeline.data_pipeline import run_data_pipeline
from src.pipeline.training_pipeline import run_training_pipeline, prepare_training_config, get_training_recommendations
from config.settings import get_config, print_config, get_recommended_config


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
    
    # Look for existing files with this config name
    pattern = os.path.join(results_dir, f'complete_pipeline_{config_name}_v*.json')
    existing_files = glob.glob(pattern)
    
    if not existing_files:
        return 1
    
    # Extract version numbers and find the highest
    version_numbers = []
    for filepath in existing_files:
        filename = os.path.basename(filepath)
        try:
            # Extract version number from filename like "complete_pipeline_heavy_model_v3.json"
            version_part = filename.split('_v')[1].split('.')[0]
            version_numbers.append(int(version_part))
        except (IndexError, ValueError):
            continue
    
    return max(version_numbers) + 1 if version_numbers else 1


def save_combined_results(pipeline_output, lstm_results, config_name):
    """
    Save combined pipeline and training results with versioned config names
    
    Args:
        pipeline_output: Pipeline analysis results
        lstm_results: Training results
        config_name: Configuration name used for this run
        
    Returns:
        dict: Dictionary with saved file paths
    """
    try:
        os.makedirs('results', exist_ok=True)
        
        # Get next version number for this config
        version = get_next_version_number(config_name)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        saved_files = {}
        
        if lstm_results:
            # Create versioned filenames
            base_filename = f"{config_name}_v{version}"
            
            # Combined results
            combined_file = f'results/complete_pipeline_{base_filename}.json'
            combined_output = pipeline_output.copy()
            combined_output['lstm_training'] = lstm_results
            combined_output['config_used'] = config_name
            combined_output['version'] = version
            combined_output['timestamp'] = timestamp
            
            with open(combined_file, 'w') as f:
                json.dump(combined_output, f, indent=2, default=str)
            
            saved_files['combined'] = combined_file
            
            # Pipeline config only
            pipeline_file = f'results/pipeline_config_{base_filename}.json'
            with open(pipeline_file, 'w') as f:
                json.dump(pipeline_output, f, indent=2, default=str)
            
            saved_files['pipeline'] = pipeline_file
            
            # Training results only
            training_file = f'results/lstm_training_{base_filename}.json'
            with open(training_file, 'w') as f:
                json.dump(lstm_results, f, indent=2, default=str)
            
            saved_files['training'] = training_file
            
            # Plot file
            plot_file = f'results/lstm_training_{base_filename}.png'
            saved_files['plot'] = plot_file
            
            print(f" Results saved as '{config_name}' version {version}:")
            for file_type, filepath in saved_files.items():
                print(f"   {file_type}: {filepath}")
            
            return saved_files
        else:
            print(" No training results to save.")
            return {}
            
    except Exception as e:
        print(f" Error saving results: {e}")
        return {}


def create_results_summary():
    """Create a summary of all saved results"""
    results_dir = 'results'
    if not os.path.exists(results_dir):
        return
    
    # Find all complete pipeline files
    pattern = os.path.join(results_dir, 'complete_pipeline_*.json')
    result_files = glob.glob(pattern)
    
    if not result_files:
        print(" No previous results found.")
        return
    
    print(f"\n Found {len(result_files)} previous runs:")
    print("-" * 80)
    print(f"{'Config':<15} | {'Version':<8} | {'Accuracy':<10} | {'Improvement':<12} | {'Date'}")
    print("-" * 80)
    
    # Sort files by config name and version
    file_data = []
    for file_path in result_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            config_name = data.get('config_used', 'unknown')
            version = data.get('version', 0)
            timestamp = data.get('timestamp', 'unknown')
            
            # Extract performance metrics
            lstm_results = data.get('lstm_training', {})
            summary = lstm_results.get('summary', {})
            accuracy = summary.get('avg_accuracy', 0)
            improvement = summary.get('improvement_over_random', 0)
            
            # Extract just date from timestamp
            try:
                date_part = timestamp.split(' ')[0] if ' ' in timestamp else timestamp[:10]
            except:
                date_part = 'unknown'
            
            file_data.append({
                'config': config_name,
                'version': version,
                'accuracy': accuracy,
                'improvement': improvement,
                'date': date_part
            })
            
        except Exception as e:
            print(f" Error reading {file_path}: {e}")
    
    # Sort by config name, then by version
    file_data.sort(key=lambda x: (x['config'], x['version']))
    
    for data in file_data:
        print(f"{data['config']:<15} | v{data['version']:<7} | {data['accuracy']:>8.1%} | {data['improvement']:>+10.1f}% | {data['date']}")
    
    print("-" * 80)


def get_best_performing_config():
    """Find the best performing configuration from previous runs"""
    results_dir = 'results'
    if not os.path.exists(results_dir):
        return None
    
    pattern = os.path.join(results_dir, 'complete_pipeline_*.json')
    result_files = glob.glob(pattern)
    
    if not result_files:
        return None
    
    best_config = None
    best_accuracy = 0
    
    for file_path in result_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            config_name = data.get('config_used', 'unknown')
            version = data.get('version', 0)
            
            lstm_results = data.get('lstm_training', {})
            summary = lstm_results.get('summary', {})
            accuracy = summary.get('avg_accuracy', 0)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_config = {
                    'config': config_name,
                    'version': version,
                    'accuracy': accuracy
                }
                
        except Exception:
            continue
    
    return best_config


def print_final_summary(lstm_results, recommendations, saved_files, config_name):
    """
    Print final summary and next steps
    
    Args:
        lstm_results: Training results
        recommendations: Analysis recommendations
        saved_files: Dictionary of saved file paths
        config_name: Configuration name used
    """
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    if not lstm_results:
        print(" Training failed - no results to analyze")
        return
    
    summary = lstm_results.get('summary', {})
    status = recommendations.get('status', 'unknown')
    
    # Performance summary
    avg_accuracy = summary.get('avg_accuracy', 0)
    improvement = summary.get('improvement_over_random', 0)
    
    print(f" Model Performance ({config_name}):")
    print(f"   Average Accuracy: {avg_accuracy:.1%}")
    print(f"   vs Random Baseline: {improvement:+.1f}%")
    print(f"   Status: {status.upper()}")
    
    # Compare with best previous result
    best_config = get_best_performing_config()
    if best_config and best_config['config'] != config_name:
        print(f"   Best Previous: {best_config['config']} v{best_config['version']} ({best_config['accuracy']:.1%})")
        if avg_accuracy > best_config['accuracy']:
            print("   NEW BEST RESULT!")
    
    # Recommendations
    print(f"\n Recommendations:")
    for rec in recommendations.get('recommendations', []):
        print(f"   - {rec}")
    
    # Files generated
    print(f"\n Generated Files:")
    for file_type, filepath in saved_files.items():
        print(f"   {file_type}: {filepath}")
    
    # Next steps based on performance
    print(f"\n Next Steps:")
    if status == 'success':
        print("   Great results! Consider:")
        print("   - Feature importance analysis")
        print("   - Out-of-sample validation")
        print("   - Trading strategy development")
    elif status == 'needs_improvement':
        print("   Model needs improvement. Try:")
        print("   - Different configurations (heavy_model, light_model)")
        print("   - Feature engineering")
        print("   - More data or different time periods")
    else:
        print("   Investigate issues:")
        print("   - Check data quality")
        print("   - Verify model configuration")
        print("   - Try 'quick_test' config for debugging")


def run_quick_test():
    """Run a quick test with minimal data for debugging"""
    print("Running quick test configuration...")
    return main('quick_test')


def run_production():
    """Run production configuration with full data"""
    print("Running production configuration...")
    return main('production')


def main(config_name='development'):
    """
    Main pipeline orchestrator - clean and simple
    
    Args:
        config_name: Configuration to use ('quick_test', 'development', 'production', etc.)
    """
    print("=" * 80)
    print("VIX DIRECTION PREDICTION PIPELINE")
    print("=" * 80)
    
    # Show previous results summary
    create_results_summary()
    
    # Get configuration
    pipeline_config, training_config = get_config(config_name)
    print_config(config_name)
    
    # Step 1: Data Pipeline
    print(f"\n Running data pipeline with '{config_name}' config...")
    data, feature_cols, target_col, pipeline_output = run_data_pipeline(pipeline_config)
    
    if data is None:
        print(" Data pipeline failed. Exiting.")
        return False
    
    # Check data size and recommend config if needed
    total_sequences = pipeline_output['pipeline_summary']['total_train_sequences']
    recommended_config = get_recommended_config(total_sequences)
    
    if recommended_config != config_name:
        print(f"\n Recommendation: With {total_sequences} sequences, consider using '{recommended_config}' config")
    
    # Step 2: Prepare training configuration
    final_training_config = prepare_training_config(pipeline_config, training_config)
    
    # Step 3: Training Pipeline  
    print(f"\n Running training pipeline with '{config_name}' config...")
    lstm_results = run_training_pipeline(data, feature_cols, target_col, final_training_config, config_name)
    
    # Step 4: Save Combined Results with versioned names
    print(f"\n Saving results for '{config_name}' config...")
    saved_files = save_combined_results(pipeline_output, lstm_results, config_name)
    
    if not saved_files:
        print(" Failed to save results.")
        return False
    
    # Step 5: Analysis and Recommendations
    print(f"\n Analyzing '{config_name}' results...")
    recommendations = get_training_recommendations(lstm_results)
    print_final_summary(lstm_results, recommendations, saved_files, config_name)
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='VIX Direction Prediction Pipeline')
    parser.add_argument('--config', default='development', 
                       choices=['quick_test', 'development', 'production', 'light_model', 'heavy_model'],
                       help='Configuration to use')
    parser.add_argument('--test', action='store_true', help='Run quick test')
    parser.add_argument('--prod', action='store_true', help='Run production config')
    
    args = parser.parse_args()
    
    if args.test:
        success = run_quick_test()
    elif args.prod:
        success = run_production()
    else:
        success = main(args.config)
    
    if success:
        print("\nPipeline completed successfully!")
    else:
        print("\nPipeline failed.")
        sys.exit(1)