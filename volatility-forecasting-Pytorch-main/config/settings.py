"""
Configuration Settings
Centralized configuration for all pipeline components
"""

# Default pipeline configuration
PIPELINE_CONFIG = {
    'tickers': ['QQQ', 'SPY', '^VIX'],
    'days': 365,                    # Number of days of historical data
    'seq_len': 5,                   # Sequence length for LSTM input
    'horizon': 1,                   # Prediction horizon (days ahead)
    'n_splits': 5,                  # Number of cross-validation splits
    'overlap_ratio': 0.5,           # Overlap ratio between windows
    'train_ratio': 0.8              # Train/test split ratio
}

# Default training configuration
TRAINING_CONFIG = {
    'batch_size': 16,               # Training batch size
    'num_epochs': 30,               # Number of training epochs
    'learning_rate': 0.001,         # Learning rate for optimizer
    'hidden_size': 64,              # LSTM hidden size
    'num_layers': 2,                # Number of LSTM layers
    'dropout': 0.2,                 # Dropout rate
    'weight_decay': 1e-5            # Weight decay for regularization
}

# Optimized configurations for different scenarios
CONFIGS = {
    'quick_test': {
        'pipeline': {
            **PIPELINE_CONFIG,
            'days': 90,
            'n_splits': 2
        },
        'training': {
            **TRAINING_CONFIG,
            'num_epochs': 10,
            'hidden_size': 32
        }
    },
    
    'development': {
        'pipeline': {
            **PIPELINE_CONFIG,
            'days': 180,
            'n_splits': 3
        },
        'training': {
            **TRAINING_CONFIG,
            'num_epochs': 50
        }
    },
    
    'production': {
        'pipeline': {
            **PIPELINE_CONFIG,
            'days': 730,  # 2 years
            'n_splits': 10
        },
        'training': {
            **TRAINING_CONFIG,
            'num_epochs': 100,
            'hidden_size': 128,
            'num_layers': 3
        }
    },
    
    'light_model': {
        'pipeline': PIPELINE_CONFIG.copy(),
        'training': {
            **TRAINING_CONFIG,
            'hidden_size': 32,
            'num_layers': 1,
            'dropout': 0.1
        }
    },
    
    'heavy_model': {
        'pipeline': PIPELINE_CONFIG.copy(),
        'training': {
            **TRAINING_CONFIG,
            'hidden_size': 128,
            'num_layers': 3,
            'dropout': 0.3,
            'num_epochs': 300
        }
    }
}

def get_config(config_name='development'):
    """
    Get configuration by name
    
    Args:
        config_name: Name of the configuration to use
        
    Returns:
        tuple: (pipeline_config, training_config)
    """
    if config_name not in CONFIGS:
        print(f"Warning: Configuration '{config_name}' not found. Using 'development' config.")
        config_name = 'development'
    
    config = CONFIGS[config_name]
    return config['pipeline'], config['training']

def print_config(config_name='development'):
    """Print configuration details"""
    pipeline_config, training_config = get_config(config_name)
    
    print(f"Configuration: {config_name}")
    print("=" * 40)
    print("Pipeline Config:")
    for key, value in pipeline_config.items():
        print(f"  {key}: {value}")
    
    print("\nTraining Config:")
    for key, value in training_config.items():
        print(f"  {key}: {value}")
    print("=" * 40)

# Model architecture recommendations based on data size
def get_recommended_config(num_sequences):
    """
    Get recommended configuration based on data size
    
    Args:
        num_sequences: Number of training sequences available
        
    Returns:
        str: Recommended configuration name
    """
    if num_sequences < 100:
        return 'light_model'
    elif num_sequences < 500:
        return 'development'
    elif num_sequences < 1000:
        return 'production'
    else:
        return 'heavy_model'