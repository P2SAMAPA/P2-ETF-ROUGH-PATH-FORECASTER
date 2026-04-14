"""
Kernel engine for ROUGH-PATH-FORECASTER
Kernel Ridge Regression for multi-output prediction
"""

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
import pickle
import os


class KernelRidgeForecaster:
    """Kernel Ridge Regression forecaster for multi-output (multiple ETFs)"""
    
    def __init__(self, depth=3, alpha=1.0, kernel='rbf'):
        self.depth = depth
        self.alpha = alpha
        self.kernel = kernel
        self.model = None
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.trained = False
        self.n_outputs = None
    
    def fit(self, X_signatures, y_returns):
        """
        Fit Kernel Ridge model for multi-output
        X_signatures: list of signature vectors
        y_returns: (n_samples, n_etfs) - target returns
        """
        X = np.vstack(X_signatures)
        y = np.array(y_returns)
        
        self.n_outputs = y.shape[1] if len(y.shape) > 1 else 1
        
        # Scale features
        X_scaled = self.scaler_x.fit_transform(X)
        
        # Scale targets (each ETF separately)
        y_scaled = self.scaler_y.fit_transform(y)
        
        # Use MultiOutputRegressor for multi-output
        base_model = KernelRidge(alpha=self.alpha, kernel=self.kernel)
        self.model = MultiOutputRegressor(base_model)
        self.model.fit(X_scaled, y_scaled)
        
        self.trained = True
        return self
    
    def predict(self, X_signatures):
        """
        Predict returns for new signatures
        Returns (n_samples, n_etfs)
        """
        if not self.trained:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        X = np.vstack(X_signatures)
        X_scaled = self.scaler_x.transform(X)
        
        y_scaled = self.model.predict(X_scaled)
        y = self.scaler_y.inverse_transform(y_scaled)
        
        return np.array(y, dtype=np.float64)


class KernelCache:
    """Cache for kernel computations"""
    
    def __init__(self, cache_dir='data/cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def get(self, cache_key):
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                return None
        return None
    
    def set(self, cache_key, matrix):
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(matrix, f)
            return True
        except Exception:
            return False
    
    def clear(self):
        for filename in os.listdir(self.cache_dir):
            filepath = os.path.join(self.cache_dir, filename)
            if os.path.isfile(filepath):
                os.remove(filepath)
