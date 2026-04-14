"""
Model registry and ensemble forecaster for ROUGH-PATH-FORECASTER
"""

import numpy as np
from signature_core import SignatureComputer, NeumannSignatureKernel, AdaptiveDepthSelector
from kernel_engine import GaussianProcessForecaster, KernelRidgeForecaster
from log_ode import LogODEForecaster, RoughPathEstimator


class SignatureGPModel:
    """Wrapper for Signature + Gaussian Process"""
    
    def __init__(self, depth=3, use_gp=True):
        self.depth = depth
        self.use_gp = use_gp
        self.sig_computer = SignatureComputer(depth=depth)
        self.kernel = NeumannSignatureKernel(depth=depth)
        
        if use_gp:
            self.forecaster = GaussianProcessForecaster(depth=depth)
        else:
            self.forecaster = KernelRidgeForecaster(depth=depth)
        
        self.trained = False
    
    def fit(self, X_paths, y_returns):
        """Fit model on path data"""
        # Compute signatures for all paths
        signatures = []
        for path in X_paths:
            sig_vec = self.sig_computer.signature_vector(path)
            signatures.append(sig_vec)
        
        self.forecaster.fit(signatures, y_returns)
        self.trained = True
        
        return self
    
    def predict(self, X_paths):
        """Predict returns for new paths"""
        if not self.trained:
            raise ValueError("Model not trained")
        
        signatures = []
        for path in X_paths:
            sig_vec = self.sig_computer.signature_vector(path)
            signatures.append(sig_vec)
        
        result = self.forecaster.predict(signatures)
        # Ensure result is numpy array
        if isinstance(result, list):
            result = np.array(result)
        return result


class LogODEModel:
    """Wrapper for Log-ODE model"""
    
    def __init__(self, log_sig_dim, hidden_dims=[64, 64]):
        self.log_sig_dim = log_sig_dim
        self.hidden_dims = hidden_dims
        self.model = LogODEForecaster(log_sig_dim, hidden_dims)
        self.roughness_estimator = RoughPathEstimator()
        self.trained = False
    
    def fit(self, log_sig_paths, returns, epochs=100):
        """Fit Log-ODE model"""
        self.model.train(log_sig_paths, returns, epochs=epochs)
        self.trained = True
        return self
    
    def predict(self, initial_log_sig, time_steps=10):
        """Predict future trajectory"""
        if not self.trained:
            raise ValueError("Model not trained")
        
        result = self.model.predict(initial_log_sig, time_steps)
        if isinstance(result, list):
            result = np.array(result)
        return result
    
    def get_roughness(self, log_sig_trajectory):
        """Get path roughness estimate"""
        return self.roughness_estimator.compute_roughness(log_sig_trajectory)


class EnsembleForecaster:
    """Ensemble of multiple forecasting methods"""
    
    def __init__(self, depths=[2, 3, 4], weights=None):
        self.depths = depths
        self.weights = weights or [0.2, 0.6, 0.2]  # Emphasis on depth 3
        self.models = {}
        self.depth_selector = AdaptiveDepthSelector(depths=depths)
    
    def fit(self, X_paths, y_returns):
        """Fit all models at different depths"""
        for depth, weight in zip(self.depths, self.weights):
            model = SignatureGPModel(depth=depth, use_gp=True)
            model.fit(X_paths, y_returns)
            self.models[depth] = {'model': model, 'weight': weight}
        
        return self
    
    def predict(self, X_paths, return_all=False):
        """Weighted ensemble prediction"""
        predictions = []
        weights = []
        
        for depth, info in self.models.items():
            pred = info['model'].predict(X_paths)
            # Convert to numpy array if needed
            if isinstance(pred, list):
                pred = np.array(pred)
            predictions.append(pred)
            weights.append(info['weight'])
        
        # Normalize weights
        weights = np.array(weights) / sum(weights)
        
        # Initialize ensemble prediction with same shape as first prediction
        ensemble_pred = np.zeros_like(predictions[0], dtype=np.float64)
        
        for pred, w in zip(predictions, weights):
            # Ensure pred is numpy array
            pred_array = np.array(pred)
            ensemble_pred += w * pred_array
        
        if return_all:
            return ensemble_pred, predictions, weights
        
        return ensemble_pred
    
    def select_best_depth(self, X_paths, y_returns):
        """Select optimal depth based on validation performance"""
        return self.depth_selector.select_depth(X_paths, y_returns)


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
