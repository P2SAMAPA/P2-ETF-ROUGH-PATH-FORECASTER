"""
Kernel engine for ROUGH-PATH-FORECASTER
Gaussian Process and Kernel Ridge Regression on signature space
"""

import numpy as np
import torch
import gpytorch
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from functools import lru_cache
import pickle


class SignatureGPModel(gpytorch.models.ExactGP):
    """Gaussian Process model with signature kernel"""
    
    def __init__(self, train_x, train_y, likelihood, kernel):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = kernel
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class SignatureKernel(gpytorch.kernels.Kernel):
    """Custom signature kernel for GPyTorch"""
    
    def __init__(self, neumann_kernel, **kwargs):
        super().__init__(**kwargs)
        self.neumann_kernel = neumann_kernel
    
    def forward(self, x1, x2, diag=False, **params):
        # x1, x2 are indices into path list
        if diag:
            return torch.ones(len(x1))
        
        n1 = len(x1)
        n2 = len(x2)
        K = torch.zeros((n1, n2))
        
        for i in range(n1):
            for j in range(n2):
                # Need to map indices back to paths
                # This is handled by the GP wrapper
                K[i, j] = torch.tensor(1.0)  # Placeholder - actual kernel in wrapper
        
        return K


class GaussianProcessForecaster:
    """Gaussian Process forecaster on signature space"""
    
    def __init__(self, depth=3, neumann_kernel=None):
        self.depth = depth
        self.neumann_kernel = neumann_kernel
        self.model = None
        self.likelihood = None
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
    
    def fit(self, X_signatures, y_returns):
        """
        Fit GP model
        X_signatures: list of signature vectors
        y_returns: target returns (n_samples, n_etfs)
        """
        # Stack signatures into matrix
        X = np.vstack(X_signatures)
        y = y_returns.mean(axis=1)  # Predict average return across ETFs or per-ETF
        
        # Scale
        X_scaled = self.scaler_x.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        # Convert to tensors
        train_x = torch.tensor(X_scaled).float()
        train_y = torch.tensor(y_scaled).float()
        
        # Initialize likelihood and model
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        kernel = gpytorch.kernels.RBFKernel()
        self.model = SignatureGPModel(train_x, train_y, self.likelihood, kernel)
        
        # Train
        self.model.train()
        self.likelihood.train()
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        
        for i in range(100):
            optimizer.zero_grad()
            output = self.model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
        
        self.model.eval()
        self.likelihood.eval()
        
        return self
    
    def predict(self, X_signatures, return_std=True):
        """Predict returns for new signatures"""
        X = np.vstack(X_signatures)
        X_scaled = self.scaler_x.transform(X)
        
        test_x = torch.tensor(X_scaled).float()
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = self.likelihood(self.model(test_x))
            mean = predictions.mean.numpy()
            std = predictions.stddev.numpy()
        
        mean = self.scaler_y.inverse_transform(mean.reshape(-1, 1)).flatten()
        std = std * self.scaler_y.scale_[0]  # Approximate inverse scaling
        
        if return_std:
            return mean, std
        return mean


class KernelRidgeForecaster:
    """Kernel Ridge Regression forecaster"""
    
    def __init__(self, depth=3, alpha=1.0, kernel='rbf'):
        self.depth = depth
        self.alpha = alpha
        self.kernel = kernel
        self.model = None
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
    
    def fit(self, X_signatures, y_returns):
        """Fit Kernel Ridge model"""
        X = np.vstack(X_signatures)
        y = y_returns.mean(axis=1)
        
        X_scaled = self.scaler_x.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        self.model = KernelRidge(alpha=self.alpha, kernel=self.kernel)
        self.model.fit(X_scaled, y_scaled)
        
        return self
    
    def predict(self, X_signatures):
        """Predict returns"""
        X = np.vstack(X_signatures)
        X_scaled = self.scaler_x.transform(X)
        
        y_scaled = self.model.predict(X_scaled)
        y = self.scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
        
        return y


class KernelCache:
    """Cache for kernel computations to avoid recomputation"""
    
    def __init__(self, cache_dir='data/cache'):
        self.cache_dir = cache_dir
        import os
        os.makedirs(cache_dir, exist_ok=True)
    
    @lru_cache(maxsize=100)
    def get_kernel_matrix(self, cache_key, paths_hash):
        """Cached kernel matrix retrieval"""
        # Implementation with file-based caching
        cache_file = f"{self.cache_dir}/kernel_{cache_key}.pkl"
        
        import os
        import pickle
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        return None
    
    def save_kernel_matrix(self, cache_key, matrix):
        """Save kernel matrix to cache"""
        cache_file = f"{self.cache_dir}/kernel_{cache_key}.pkl"
        import pickle
        with open(cache_file, 'wb') as f:
            pickle.dump(matrix, f)
