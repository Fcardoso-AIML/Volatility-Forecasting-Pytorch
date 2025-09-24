import numpy as np
import torch
from torch.utils.data import Dataset

import pandas as pd


class TimeSeriesWindowDataset(Dataset):
    def __init__(self, features_df, target_series, seq_len, horizon):
        self.X = features_df.values.astype(np.float32)
        self.y = target_series.values.astype(np.float32)  # Separate target
        self.seq_len = seq_len
        self.horizon = horizon
        self.num_samples = len(self.X) - seq_len - horizon + 1

    def __getitem__(self, idx):
        x = self.X[idx : idx + self.seq_len]  # Features sequence
        y = self.y[idx + self.seq_len + self.horizon - 1]  # Single target
        return torch.tensor(x), torch.tensor(y)
    
    def __len__(self):
        return self.num_samples 
    
    def get_class_weights(self):
        """Calculate class weights for imbalanced dataset"""
        from collections import Counter
        class_counts = Counter(self.y)
        total_samples = len(self.y)
        
        weights = {}
        for class_id, count in class_counts.items():
            weights[class_id] = total_samples / (len(class_counts) * count)
        
        print(f"Class distribution: {dict(class_counts)}")
        print(f"Class weights: {weights}")
        return weights
    

