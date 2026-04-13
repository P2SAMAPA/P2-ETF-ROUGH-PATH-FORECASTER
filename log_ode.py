"""
Log-ODE for ROUGH-PATH-FORECASTER
Neural controlled differential equation on log-signature space
"""

import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
from scipy.integrate import odeint as scipy_odeint


class VectorField(nn.Module):
    """Neural vector field for Log-ODE"""
    
    def __init__(self, input_dim, hidden_dims=[64, 64], output_dim=None):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim
        
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.Tanh())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, t, z):
        return self.net(z)


class LogODESolver:
    """Solve Log-ODE for path development"""
    
    def __init__(self, vector_field, solver='dopri5', rtol=1e-5, atol=1e-6):
        self.vector_field = vector_field
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
    
    def solve(self, z0, t_eval):
        """
        Solve ODE from initial condition z0 at times t_eval
        """
        z0_tensor = torch.tensor(z0, dtype=torch.float32)
        t_tensor = torch.tensor(t_eval, dtype=torch.float32)
        
        with torch.no_grad():
            sol = odeint(
                self.vector_field, z0_tensor, t_tensor,
                rtol=self.rtol, atol=self.atol, method=self.solver
            )
        
        return sol.numpy()
    
    def solve_scipy(self, z0, t_eval):
        """SciPy ODE solver (CPU-only fallback)"""
        def func(z, t):
            with torch.no_grad():
                z_tensor = torch.tensor(z, dtype=torch.float32)
                t_tensor = torch.tensor(t, dtype=torch.float32)
                dz = self.vector_field(t_tensor, z_tensor).numpy()
            return dz
        
        sol = scipy_odeint(func, z0, t_eval)
        return sol


class LogODEForecaster:
    """Forecaster using Log-ODE dynamics"""
    
    def __init__(self, log_sig_dim, hidden_dims=[64, 64], solver='dopri5'):
        self.log_sig_dim = log_sig_dim
        self.vector_field = VectorField(log_sig_dim, hidden_dims, log_sig_dim)
        self.solver_obj = LogODESolver(self.vector_field, solver=solver)
        self.trained = False
    
    def train(self, log_sig_paths, returns, epochs=100, lr=0.001):
        """
        Train vector field to predict returns from log-signature paths
        """
        optimizer = torch.optim.Adam(self.vector_field.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        
        for epoch in range(epochs):
            total_loss = 0
            for path, target in zip(log_sig_paths, returns):
                # Path is (T, d) log-signature trajectory
                path_tensor = torch.tensor(path, dtype=torch.float32)
                target_tensor = torch.tensor(target, dtype=torch.float32)
                
                t = torch.linspace(0, 1, len(path))
                
                # Integrate from initial condition
                z0 = path_tensor[0]
                pred = odeint(self.vector_field, z0, t, method='dopri5')
                
                loss = loss_fn(pred[-1], target_tensor)
                total_loss += loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss.item():.6f}")
        
        self.trained = True
        return self
    
    def predict(self, initial_log_sig, time_steps):
        """
        Predict future log-signature evolution
        """
        if not self.trained:
            raise ValueError("Model not trained")
        
        t_eval = np.linspace(0, 1, time_steps)
        trajectory = self.solver_obj.solve(initial_log_sig, t_eval)
        
        return trajectory


class LyapunovExponents:
    """Compute Lyapunov exponents from Log-ODE for stability analysis"""
    
    @staticmethod
    def compute_spectrum(vector_field, trajectory, dt=0.01):
        """
        Approximate Lyapunov spectrum from trajectory
        """
        n_steps = len(trajectory)
        dim = trajectory.shape[1]
        
        # Initialize orthonormal basis
        Q = np.eye(dim)
        exponents = np.zeros(dim)
        
        for i in range(n_steps - 1):
            # Jacobian approximation via finite differences
            J = np.zeros((dim, dim))
            eps = 1e-6
            
            for d in range(dim):
                perturb = np.zeros(dim)
                perturb[d] = eps
                
                # Forward difference
                with torch.no_grad():
                    z = torch.tensor(trajectory[i] + perturb, dtype=torch.float32)
                    t = torch.tensor(0.0, dtype=torch.float32)
                    f_plus = vector_field(t, z).numpy()
                    
                    z = torch.tensor(trajectory[i] - perturb, dtype=torch.float32)
                    f_minus = vector_field(t, z).numpy()
                
                J[:, d] = (f_plus - f_minus) / (2 * eps)
            
            # QR decomposition
            Q_new = J @ Q
            Q, R = np.linalg.qr(Q_new)
            
            # Update exponents
            exponents += np.log(np.abs(np.diag(R))) / (n_steps * dt)
        
        # Sort descending
        exponents = np.sort(exponents)[::-1]
        
        return exponents


class RoughPathEstimator:
    """Estimate path roughness from log-signature variation"""
    
    @staticmethod
    def estimate_hurst(log_sig_trajectory):
        """
        Estimate Hurst exponent from log-signature path
        Hurst > 0.5: persistent, < 0.5: mean-reverting
        """
        # Compute variation at different scales
        n = len(log_sig_trajectory)
        scales = np.arange(2, min(50, n // 4))
        variations = []
        
        for scale in scales:
            # Subsample at scale
            idx = np.arange(0, n, scale)
            subsampled = log_sig_trajectory[idx]
            
            # Compute variation
            diffs = np.diff(subsampled, axis=0)
            var = np.mean(np.sum(diffs**2, axis=1))
            variations.append(var)
        
        # Log-log regression
        log_scales = np.log(scales)
        log_vars = np.log(variations)
        
        # Slope = 2H - 2
        from sklearn.linear_model import LinearRegression
        reg = LinearRegression()
        reg.fit(log_scales.reshape(-1, 1), log_vars.reshape(-1, 1))
        slope = reg.coef_[0][0]
        
        H = (slope + 2) / 2
        H = np.clip(H, 0.1, 0.9)
        
        return H
    
    @staticmethod
    def compute_roughness(log_sig_trajectory):
        """
        Compute path roughness measure
        """
        # p-variation for p=2
        diffs = np.diff(log_sig_trajectory, axis=0)
        quadratic_variation = np.sum(diffs**2)
        
        # Normalized by length
        roughness = quadratic_variation / len(log_sig_trajectory)
        
        return roughness
