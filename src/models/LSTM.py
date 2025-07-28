"""
LSTM VIX Direction Prediction Module
Contains all LSTM components for import into main.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import Counter
import os
import json
import warnings
warnings.filterwarnings('ignore')

# Default training configuration
DEFAULT_CONFIG = {
    'batch_size': 16,
    'num_epochs': 30,
    'learning_rate': 0.001,
    'hidden_size': 64,
    'num_layers': 2,
    'dropout': 0.2,
    'weight_decay': 1e-5,
    'seq_len': 5,
    'horizon': 1,
    'n_splits': 5,
    'overlap_ratio': 0.5,
    'train_ratio': 0.8
}


class VIXPredictor(nn.Module):
    def __init__(self, input_size=7, hidden_size=64, num_layers=2, num_classes=3, dropout=0.2):
        super(VIXPredictor, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Classification head (BatchNorm removed to prevent batch size errors)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, num_classes)
        )
        
    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Take the output from the last time step
        last_output = lstm_out[:, -1, :]
        
        # Classification
        logits = self.classifier(last_output)
        
        return logits


def get_device_info():
    """Get device information for training"""
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    info = {
        'device': device_type,
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available()
    }
    
    if torch.cuda.is_available():
        try:
            info['cuda_version'] = torch.version.cuda
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['gpu_memory'] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        except:
            pass
    
    return info


def get_class_weights_tensor(class_weights_dict, device):
    """Convert class weights dictionary to tensor"""
    return torch.tensor([
        class_weights_dict.get(0.0, 1.0),
        class_weights_dict.get(1.0, 1.0),
        class_weights_dict.get(2.0, 1.0)
    ]).to(device)


def create_model_and_optimizer(input_size, device, config):
    """Create model and optimizer with given configuration"""
    model = VIXPredictor(
        input_size=input_size,
        hidden_size=config.get('hidden_size', 64),
        num_layers=config.get('num_layers', 2),
        num_classes=3,
        dropout=config.get('dropout', 0.2)
    ).to(device)
    
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config.get('learning_rate', 0.001), 
        weight_decay=config.get('weight_decay', 1e-5)
    )
    
    return model, optimizer


def train_model(model, train_loader, criterion, optimizer, device, num_epochs=50):
    """Train the LSTM model"""
    model.train()
    train_losses = []
    
    print(f"Training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            # Move to device
            data, targets = data.to(device), targets.long().to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (helps with LSTM training)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            
            # Statistics
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += targets.size(0)
            correct_predictions += (predicted == targets).sum().item()
        
        # Calculate average loss and accuracy
        avg_loss = epoch_loss / len(train_loader)
        accuracy = 100 * correct_predictions / total_predictions
        train_losses.append(avg_loss)
        
        # Print progress
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    return train_losses


def evaluate_model(model, test_loader, device):
    """Evaluate the model"""
    model.eval()
    all_predictions = []
    all_actuals = []
    
    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_actuals.extend(targets.numpy())
    
    return all_predictions, all_actuals


def get_safe_classification_report(actuals, predictions):
    """Generate classification report that handles missing classes"""
    try:
        # Get unique classes present in the data
        unique_classes = sorted(set(list(actuals) + list(predictions)))
        
        # Define target names for all possible classes
        all_target_names = {0: 'Down', 1: 'Neutral', 2: 'Up'}
        
        # Create target names only for present classes
        present_target_names = [all_target_names.get(cls, f'Class_{cls}') for cls in unique_classes]
        
        if len(unique_classes) > 1:
            report = classification_report(
                actuals, predictions, 
                labels=unique_classes,
                target_names=present_target_names,
                zero_division=0
            )
            return report
        else:
            class_name = all_target_names.get(unique_classes[0], f'Class_{unique_classes[0]}')
            return f"Only one class present: {class_name}"
            
    except Exception as e:
        return f"Classification report error: {e}"


def visualize_results(all_results, all_acts, all_preds, save_plots=True, config_name=None, version=None):
    """Create visualizations of the results"""
    if not all_results or len(all_results) == 0:
        print("No results to visualize")
        return
        
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Accuracy by fold
    folds = [r['fold'] for r in all_results]
    accuracies = [r['accuracy'] for r in all_results]
    
    ax1.bar(folds, accuracies, alpha=0.7, color='skyblue', edgecolor='navy')
    ax1.axhline(y=1/3, color='red', linestyle='--', linewidth=2, label='Random Baseline (33.3%)')
    ax1.set_xlabel('Fold')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Accuracy by Fold')
    ax1.legend()
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # Add accuracy values on bars
    for i, acc in enumerate(accuracies):
        ax1.text(i, acc + 0.02, f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Training loss progression
    train_losses = [r.get('train_loss') for r in all_results if r.get('train_loss') is not None]
    if train_losses:
        ax2.plot(folds[:len(train_losses)], train_losses, 'o-', color='orange', linewidth=2, markersize=8)
        ax2.set_xlabel('Fold')
        ax2.set_ylabel('Final Training Loss')
        ax2.set_title('Final Training Loss by Fold')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for i, loss in enumerate(train_losses):
            ax2.text(i, loss + max(train_losses)*0.02, f'{loss:.3f}', ha='center', va='bottom')
    else:
        ax2.text(0.5, 0.5, 'No training loss data available', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Training Loss (No Data)')
    
    # 3. Confusion Matrix
    if all_acts and all_preds:
        try:
            # Get unique classes
            unique_classes = sorted(set(list(all_acts) + list(all_preds)))
            class_names = {0: 'Down', 1: 'Neutral', 2: 'Up'}
            
            cm = confusion_matrix(all_acts, all_preds, labels=unique_classes)
            
            # Create labels for present classes only
            labels = [class_names.get(cls, f'Class_{cls}') for cls in unique_classes]
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
                       xticklabels=labels, yticklabels=labels)
            ax3.set_title('Confusion Matrix (All Folds)')
            ax3.set_xlabel('Predicted')
            ax3.set_ylabel('Actual')
        except Exception as e:
            ax3.text(0.5, 0.5, f'Confusion matrix error:\n{str(e)}', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=10)
            ax3.set_title('Confusion Matrix (Error)')
    else:
        ax3.text(0.5, 0.5, 'No predictions to show', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Confusion Matrix (No Data)')
    
    # 4. Dataset sizes
    train_sizes = [r.get('num_train_sequences', 0) for r in all_results]
    test_sizes = [r.get('num_test_sequences', 0) for r in all_results]
    
    x = np.arange(len(folds))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, train_sizes, width, label='Train Sequences', alpha=0.7, color='green')
    bars2 = ax4.bar(x + width/2, test_sizes, width, label='Test Sequences', alpha=0.7, color='red')
    ax4.set_xlabel('Fold')
    ax4.set_ylabel('Number of Sequences')
    ax4.set_title('Dataset Sizes by Fold')
    ax4.legend()
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'Fold {f}' for f in folds])
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{int(height)}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{int(height)}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot with config-specific name
    if save_plots:
        os.makedirs('results', exist_ok=True)
        if config_name and version:
            filename = f'results/lstm_training_{config_name}_v{version}.png'
        else:
            filename = 'results/lstm_training_results.png'
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Visualizations saved to: {filename}")
    
    plt.show()


def analyze_training_results(all_results):
    """Analyze and summarize training results"""
    if not all_results:
        print("No training results to analyze")
        return {
            'success': False,
            'avg_accuracy': 0,
            'std_accuracy': 0,
            'improvement_over_random': 0,
            'individual_accuracies': []
        }
    
    # Calculate overall metrics
    accuracies = [r['accuracy'] for r in all_results]
    avg_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    
    print(f"Cross-Validation Summary:")
    print(f"   Number of folds: {len(all_results)}")
    print(f"   Average accuracy: {avg_accuracy:.3f} ± {std_accuracy:.3f}")
    print(f"   Individual accuracies: {[round(acc, 3) for acc in accuracies]}")
    
    # Compare to random baseline
    random_baseline = 1/3  # 33.33% for 3-class problem
    improvement = (avg_accuracy - random_baseline) / random_baseline * 100
    
    print(f"\nPerformance Analysis:")
    print(f"   Random baseline (3-class): {random_baseline:.3f}")
    print(f"   Model performance: {avg_accuracy:.3f}")
    print(f"   Improvement over random: {improvement:+.1f}%")
    
    if avg_accuracy > random_baseline + 0.05:  # 5% buffer for significance
        print("   SUCCESS: Model significantly beats random prediction!")
        success = True
    elif avg_accuracy > random_baseline:
        print("   Model slightly beats random (may not be significant)")
        success = False
    else:
        print("   Model does not beat random prediction")
        success = False
    
    # Analyze predictions by fold
    print(f"\nDetailed Fold Analysis:")
    for result in all_results:
        print(f"   Fold {result['fold']}: {result['accuracy']:.3f} accuracy, "
              f"{result.get('num_train_sequences', 0)} train / {result.get('num_test_sequences', 0)} test sequences")
    
    return {
        'success': success,
        'avg_accuracy': float(avg_accuracy),
        'std_accuracy': float(std_accuracy),
        'improvement_over_random': float(improvement),
        'individual_accuracies': [float(acc) for acc in accuracies]
    }


def run_lstm_training(data, feature_cols, target_col, config):
    """
    Complete LSTM training pipeline that can be imported into main.py
    
    Args:
        data: DataFrame with features and target
        feature_cols: List of feature column names
        target_col: Target column name
        config: Training configuration dictionary
    
    Returns:
        Dictionary with training results and analysis
    """
    try:
        from models.splitter import RollingWindowSplitter
        from models.dataset import TimeSeriesWindowDataset
    except ImportError:
        print("Error importing required modules")
        return None
    
    print("=" * 80)
    print("LSTM VIX DIRECTION PREDICTION TRAINING")
    print("=" * 80)
    
    # Device setup
    device_info = get_device_info()
    device = torch.device(device_info['device'])
    print(f"Using device: {device}")
    print(f"PyTorch version: {device_info['pytorch_version']}")
    if device_info['cuda_available']:
        print(f"CUDA version: {device_info.get('cuda_version', 'N/A')}")
        if 'gpu_name' in device_info:
            print(f"GPU: {device_info['gpu_name']}")
    
    # Get config details for file naming
    config_name = config.get('config_name')
    version = config.get('version')
    
    print(f"\nConfiguration: {config}")
    if config_name:
        print(f"Config name: {config_name}")
        print(f"Version: {version}")
    
    # Cross-validation training
    print(f"\nCross-validation training...")
    splitter = RollingWindowSplitter(
        data, 
        n_splits=config['n_splits'], 
        overlap_ratio=config['overlap_ratio'], 
        train_ratio=config['train_ratio']
    )
    
    all_results = []
    all_class_weights = []
    
    for fold, (train_df, test_df, window_info) in enumerate(splitter.split()):
        print(f"\n{'='*60}")
        print(f"FOLD {fold}: {window_info['start_date'].strftime('%Y-%m-%d')} to {window_info['end_date'].strftime('%Y-%m-%d')}")
        print(f"{'='*60}")
        
        # Separate features and target
        train_features = train_df[feature_cols]
        train_target = train_df[target_col]
        test_features = test_df[feature_cols]
        test_target = test_df[target_col]
        
        # Scale ONLY features (not the target!)
        scaler = StandardScaler()
        train_features_scaled = scaler.fit_transform(train_features)
        test_features_scaled = scaler.transform(test_features)
        
        # Convert back to DataFrames with proper indices
        scaled_train_features = pd.DataFrame(
            train_features_scaled,
            index=train_features.index,
            columns=train_features.columns
        )
        scaled_test_features = pd.DataFrame(
            test_features_scaled,
            index=test_features.index,
            columns=test_features.columns
        )
        
        # Create datasets
        train_dataset = TimeSeriesWindowDataset(
            scaled_train_features, train_target, 
            seq_len=config['seq_len'], horizon=config['horizon']
        )
        test_dataset = TimeSeriesWindowDataset(
            scaled_test_features, test_target, 
            seq_len=config['seq_len'], horizon=config['horizon']
        )
        
        print(f"Train sequences: {len(train_dataset)}")
        print(f"Test sequences: {len(test_dataset)}")
        
        if len(train_dataset) == 0 or len(test_dataset) == 0:
            print("Insufficient data for this fold, skipping...")
            continue
        
        # Get class weights for this fold
        fold_class_weights = train_dataset.get_class_weights()
        all_class_weights.append(fold_class_weights)
        
        # Convert class weights to tensor
        weights_tensor = get_class_weights_tensor(fold_class_weights, device)
        
        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
        
        # Initialize fresh model for each fold
        fold_model, fold_optimizer = create_model_and_optimizer(len(feature_cols), device, config)
        
        # Model info
        total_params = sum(p.numel() for p in fold_model.parameters())
        print(f"Model parameters: {total_params:,}")
        
        # Setup training for this fold
        fold_criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        
        print(f"Class weights for fold {fold}: {weights_tensor.cpu().numpy()}")
        
        # Train the model
        print(f"Training model for fold {fold}...")
        train_losses = train_model(fold_model, train_loader, fold_criterion, 
                                 fold_optimizer, device, config['num_epochs'])
        
        # Evaluate the model
        print(f"Evaluating model for fold {fold}...")
        predictions, actuals = evaluate_model(fold_model, test_loader, device)
        
        # Calculate metrics
        accuracy = accuracy_score(actuals, predictions)
        
        print(f"\nFOLD {fold} RESULTS:")
        print(f"   Accuracy: {accuracy:.3f}")
        print(f"   Predictions: {predictions}")
        print(f"   Actuals:     {actuals}")
        
        # Safe classification report
        if len(set(actuals)) > 1:  # Only if we have multiple classes in test
            report = get_safe_classification_report(actuals, predictions)
            print(f"\nClassification Report:\n{report}")
        else:
            print("Only one class in test set - no classification report possible")
        
        # Store results
        all_results.append({
            'fold': fold,
            'window_info': {
                'start_date': window_info['start_date'].isoformat(),
                'end_date': window_info['end_date'].isoformat(),
                'total_days': window_info['total_days'],
                'train_days': window_info['train_days'],
                'test_days': window_info['test_days']
            },
            'accuracy': float(accuracy),
            'predictions': [int(p) for p in predictions],
            'actuals': [int(a) for a in actuals],
            'train_loss': float(train_losses[-1]) if train_losses else None,
            'class_weights': {str(k): float(v) for k, v in fold_class_weights.items()},
            'num_train_sequences': len(train_dataset),
            'num_test_sequences': len(test_dataset)
        })
    
    # Final Results Analysis
    print(f"\n{'='*80}")
    print("FINAL CROSS-VALIDATION RESULTS")
    print(f"{'='*80}")
    
    if all_results:
        # Use the analysis function
        analysis = analyze_training_results(all_results)
        
        # Calculate average class weights across all folds
        avg_weights = {}
        for class_id in [0, 1, 2]:
            weights = []
            for w in all_class_weights:
                if float(class_id) in w:
                    weights.append(w[float(class_id)])
            if weights:
                avg_weights[class_id] = np.mean(weights)
        
        print(f"\nAverage Class Weights Across Folds:")
        print(f"   Down (0): {avg_weights.get(0, 0):.2f}")
        print(f"   Neutral (1): {avg_weights.get(1, 0):.2f} <- Rare signals")
        print(f"   Up (2): {avg_weights.get(2, 0):.2f}")
        
        # Combine all predictions for overall analysis
        all_preds = []
        all_acts = []
        for result in all_results:
            all_preds.extend(result['predictions'])
            all_acts.extend(result['actuals'])
        
        if len(all_acts) > 0:
            pred_dist = Counter(all_preds)
            actual_dist = Counter(all_acts)
            
            print(f"\nOverall Prediction Distribution:")
            print(f"   Predicted: Down={pred_dist.get(0,0)}, Neutral={pred_dist.get(1,0)}, Up={pred_dist.get(2,0)}")
            print(f"   Actual:    Down={actual_dist.get(0,0)}, Neutral={actual_dist.get(1,0)}, Up={actual_dist.get(2,0)}")
            
            # Overall classification report with safe handling
            if len(set(all_acts)) > 1:
                overall_report = get_safe_classification_report(all_acts, all_preds)
                print(f"\nOverall Classification Report:")
                print(overall_report)
        
        # Create visualizations with config-aware naming
        print(f"Creating visualizations...")
        visualize_results(all_results, all_acts, all_preds, save_plots=True, 
                         config_name=config_name, version=version)
        
        # Save results with config-aware naming
        os.makedirs('results', exist_ok=True)
        results_output = {
            'config': config,
            'summary': analysis,
            'fold_results': all_results,
            'avg_class_weights': {str(k): float(v) for k, v in avg_weights.items()},
            'all_predictions': all_preds,
            'all_actuals': all_acts
        }
        
        # Use config-specific filename if available
        if config_name and version:
            results_file = f'results/lstm_training_{config_name}_v{version}.json'
        else:
            results_file = 'results/lstm_training_results.json'
        
        with open(results_file, 'w') as f:
            json.dump(results_output, f, indent=2, default=str)
        
        print(f"\nResults saved to: {results_file}")
        
        print(f"VIX Direction Prediction Training Complete!")
        print("Key Takeaways:")
        if analysis['success']:
            print("   Your model successfully learned VIX direction patterns!")
            print("   Consider analyzing which features are most predictive")
            print("   Ready for strategy development and backtesting")
        else:
            print("   Model shows some learning but needs improvement")
            print("   Consider: more data, different features, or model architecture")
            print("   Analyze feature importance and market regime effects")
        
        if config_name and version:
            print(f"   Check results/lstm_training_{config_name}_v{version}.png for detailed analysis")
        else:
            print("   Check results/lstm_training_results.png for detailed analysis")
        
        return results_output
        
    else:
        print("No valid results obtained from cross-validation")
        return None


def standalone_training():
    """Standalone training function for running LSTM.py directly"""
    try:
        from data.raw import download_fin_data, tickers
        from data.loader import load_and_engineer_features
    except ImportError:
        print("Error importing data modules for standalone training")
        return
    
    # Configuration
    config = DEFAULT_CONFIG.copy()
    config.update({
        'tickers': ['QQQ', 'SPY', '^VIX'],
        'days': 365,
        'num_epochs': 30,
        'config_name': 'standalone',
        'version': 1
    })
    
    # Load and engineer features
    print("Loading and engineering features...")
    try:
        raw_data = download_fin_data(tickers=tickers, lookback_days=config['days'])
        data, feature_cols, target_col = load_and_engineer_features(raw_data)
        print(f"Data loaded: {data.shape}")
        print(f"Features: {feature_cols}")
        print(f"Target: {target_col}")
        print(f"Data period: {data.index[0]} to {data.index[-1]}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Run training
    results = run_lstm_training(data, feature_cols, target_col, config)
    
    if results:
        print("Standalone training completed successfully!")
    else:
        print("Standalone training failed!")
    
    return results


if __name__ == "__main__":
    standalone_training()