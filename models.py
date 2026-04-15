"""
Model registry and ensemble forecaster for ROUGH-PATH-FORECASTER
"""

import numpy as np
from signature_core import SignatureComputer, NeumannSignatureKernel, AdaptiveDepthSelector
from kernel_engine import KernelRidgeForecaster
from log_ode import LogODEForecaster, RoughPathEstimator


class SignatureModel:
    """Wrapper for Signature + Kernel Ridge Regression"""
    
    def __init__(self, depth=3):
        self.depth = depth
        self.sig_computer = SignatureComputer(depth=depth)
        self.forecaster = KernelRidgeForecaster(depth=depth, alpha=0.1, kernel='rbf')
        self.trained = False
    
    def fit(self, X_paths, y_returns):
        """
        Fit model on path data
        X_paths: list of paths (n_samples, path_length, n_features)
        y_returns: (n_samples, n_etfs) - target returns for each ETF
        """
        signatures = []
        for path in X_paths:
            sig_vec = self.sig_computer.signature_vector(path)
            signatures.append(sig_vec)
        
        self.forecaster.fit(signatures, y_returns)
        self.trained = True
        return self
    
    def predict(self, X_paths):
        """Predict returns for new paths - returns (n_samples, n_etfs)"""
        if not self.trained:
            raise ValueError("Model not trained")
        
        signatures = []
        for path in X_paths:
            sig_vec = self.sig_computer.signature_vector(path)
            signatures.append(sig_vec)
        
        result = self.forecaster.predict(signatures)
        
        # Ensure 2D output (n_samples, n_etfs)
        if len(result.shape) == 1:
            result = result.reshape(-1, 1)
        
        return result


class EnsembleForecaster:
    """Ensemble of multiple forecasting methods at different depths"""
    
    def __init__(self, depths=[2, 3, 4], weights=None):
        self.depths = depths
        self.weights = weights or [0.2, 0.6, 0.2]
        self.models = {}
        self.trained = False
    
    def fit(self, X_paths, y_returns):
        """Fit all models at different depths"""
        for depth, weight in zip(self.depths, self.weights):
            model = SignatureModel(depth=depth)
            model.fit(X_paths, y_returns)
            self.models[depth] = {'model': model, 'weight': weight}
        
        self.trained = True
        return self
    
    def predict(self, X_paths, return_all=False):
        """Weighted ensemble prediction - returns (n_samples, n_etfs)"""
        if not self.trained:
            raise ValueError("Model not trained")
        
        if len(self.models) == 0:
            raise ValueError("No models in ensemble")
        
        predictions = []
        weights = []
        
        for depth, info in self.models.items():
            pred = info['model'].predict(X_paths)
            predictions.append(pred)
            weights.append(info['weight'])
        
        # Normalize weights
        weights = np.array(weights) / sum(weights)
        
        # Initialize ensemble prediction
        ensemble_pred = np.zeros_like(predictions[0], dtype=np.float64)
        
        for pred, w in zip(predictions, weights):
            ensemble_pred += w * pred
        
        if return_all:
            return ensemble_pred, predictions, weights
        
        return ensemble_pred
    
    def predict_single(self, X_path):
        """
        Predict for a single path - returns 1D array of predictions per ETF
        """
        result = self.predict([X_path])
        return result[0]  # Returns 1D array of shape (n_etfs,)


class ModelRegistry:
    """Registry for saving/loading models"""
    
    def __init__(self, save_dir='models_saved'):
        self.save_dir = save_dir
        import os
        os.makedirs(save_dir, exist_ok=True)
    
    def save(self, model, module, mode, window_start=None):
        """Save model to disk"""
        import pickle
        
        if window_start:
            filename = f"{self.save_dir}/{module}_{mode}_window_{window_start}.pkl"
        else:
            filename = f"{self.save_dir}/{module}_{mode}.pkl"
        
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        
        return filename
    
    def load(self, module, mode, window_start=None):
        """Load model from disk"""
        import pickle
        
        if window_start:
            filename = f"{self.save_dir}/{module}_{mode}_window_{window_start}.pkl"
        else:
            filename = f"{self.save_dir}/{module}_{mode}.pkl"
        
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None
    
    def delete(self, module, mode, window_start=None):
        """Delete saved model"""
        import os
        
        if window_start:
            filename = f"{self.save_dir}/{module}_{mode}_window_{window_start}.pkl"
        else:
            filename = f"{self.save_dir}/{module}_{mode}.pkl"
        
        if os.path.exists(filename):
            os.remove(filename)
            return True
        return False
    
    def list_models(self, module=None, mode=None):
        """List all saved models"""
        import os
        import glob
        
        pattern = f"{self.save_dir}/*.pkl"
        if module:
            pattern = f"{self.save_dir}/{module}_*.pkl"
        if mode and module:
            pattern = f"{self.save_dir}/{module}_{mode}_*.pkl"
        
        files = glob.glob(pattern)
        models = []
        for f in files:
            basename = os.path.basename(f)
            models.append(basename.replace('.pkl', ''))
        
        return models
