import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Iterator, Tuple
class RollingWindowSplitter: #splits, handles overlap and scales data
    def __init__(self, 
                 data: pd.DataFrame, 
                 n_splits: int = 3, 
                 overlap_ratio: float = 0.5, 
                 train_ratio: float = 0.75, 
                 min_window_size: int = 40,
                 scaler_type=StandardScaler):
        """
        Financial time series splitter with overlapping windows for robust validation.
        
        Parameters:
        - data: Input DataFrame
        - n_splits: Number of validation windows (3 recommended for 90-day data)
        - overlap_ratio: Overlap between windows (0.5 = 50% overlap, recommended for daily volatility)
        - train_ratio: Train/test split within each window (0.75 recommended)
        - min_window_size: Minimum window size in days (40 minimum for meaningful patterns)
        - scaler_type: Sklearn scaler class
        """
        self.data = data
        self.n_splits = n_splits
        self.overlap_ratio = overlap_ratio
        self.train_ratio = train_ratio
        self.min_window_size = min_window_size
        self.scaler_type = scaler_type

    def split(self) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame, dict]]:
        """
        Generate overlapping windows for financial time series validation.
        
        Yields:
        - train_df: Training data for this window
        - test_df: Test data for this window  
        - window_info: Dictionary with window metadata
        """
        total_length = len(self.data)
        
        # optimal window size 
        window_size = max(self.min_window_size, int(total_length * 0.75))
        
        # step size for overlapping windows
        if self.n_splits == 1:
            step_size = 0  
        else:
            total_steps = total_length - window_size
            step_size = max(1, total_steps // (self.n_splits - 1))
            
            target_step = int(window_size * (1 - self.overlap_ratio))
            step_size = min(step_size, target_step)

        print(f"Window config: size={window_size}, step={step_size}, overlap={self.overlap_ratio:.1%}")
        
        windows_created = 0
        for i in range(self.n_splits):
            start = i * step_size
            end = start + window_size
            
            # Ensure we don't exceed data bounds
            if end > total_length:
                end = total_length
                start = max(0, end - window_size)
            
            # Skip if window is too small 
            if end - start < self.min_window_size:
                print(f"Skipping window {i}: too small ({end - start} < {self.min_window_size})")
                continue
                
            window = self.data.iloc[start:end].copy()
            
            train_end = int(len(window) * self.train_ratio)
            train_df = window.iloc[:train_end].copy()
            test_df = window.iloc[train_end:].copy()
            
            min_train_days = 20  
            min_test_days = 8    
            
            if len(train_df) < min_train_days or len(test_df) < min_test_days:
                print(f"Skipping window {i}: insufficient data (train={len(train_df)}, test={len(test_df)})")
                continue
            
            window_info = {
                'window_id': windows_created,
                'start_idx': start,
                'end_idx': end,
                'start_date': window.index[0],
                'end_date': window.index[-1],
                'total_days': len(window),
                'train_days': len(train_df),
                'test_days': len(test_df),
                'overlap_with_previous': (start > 0) and (start < step_size * i)
            }
            
            print(f"Window {windows_created}: {window_info['start_date'].strftime('%Y-%m-%d')} to {window_info['end_date'].strftime('%Y-%m-%d')} "
                  f"({window_info['total_days']} days, train:{window_info['train_days']}, test:{window_info['test_days']})")
            
            windows_created += 1
            yield train_df, test_df, window_info

    def scale(self, train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
        """
        Scale train and test data. Fit scaler on training data only to prevent data leakage.
        
        Returns:
        - scaled_train: Scaled training DataFrame
        - scaled_test: Scaled test DataFrame  
        - scaler: Fitted scaler object
        """
        scaler = self.scaler_type()
        
        scaled_train = pd.DataFrame(
            scaler.fit_transform(train),
            index=train.index,
            columns=train.columns
        )
        
        scaled_test = pd.DataFrame(
            scaler.transform(test),
            index=test.index,
            columns=test.columns
        )
        
        return scaled_train, scaled_test, scaler
    
    def get_recommended_config(self, data_length: int) -> dict:
        """
        Get recommended configuration based on data length.
        
        Returns recommended n_splits, overlap_ratio, and train_ratio.
        """
        if data_length < 60:
            return {
                'n_splits': 2,
                'overlap_ratio': 0.6,
                'train_ratio': 0.8,
                'warning': 'Dataset very small - consider gathering more data'
            }
        elif data_length < 120:
            return {
                'n_splits': 3,
                'overlap_ratio': 0.5,
                'train_ratio': 0.75,
                'warning': None
            }
        elif data_length < 250:
            return {
                'n_splits': 4,
                'overlap_ratio': 0.4,
                'train_ratio': 0.7,
                'warning': None
            }
        else:
            return {
                'n_splits': 5,
                'overlap_ratio': 0.3,
                'train_ratio': 0.7,
                'warning': None
            }

