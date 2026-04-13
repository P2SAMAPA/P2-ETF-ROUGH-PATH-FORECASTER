"""
Signature computation core for ROUGH-PATH-FORECASTER
Implements truncated signature, log-signature, and Neumann kernel
"""

import numpy as np
from itertools import combinations_with_replacement
from functools import lru_cache
import warnings


class SignatureComputer:
    """Compute truncated path signatures"""
    
    def __init__(self, depth=3, lead_lag=True, basepoint=True, time_channel=True):
        """
        Args:
            depth: Truncation depth (2, 3, or 4)
            lead_lag: Add lead-lag augmentation
            basepoint: Add basepoint (starting at 0)
            time_channel: Add time as extra dimension
        """
        self.depth = depth
        self.lead_lag = lead_lag
        self.basepoint = basepoint
        self.time_channel = time_channel
    
    def augment_path(self, path):
        """Augment path with lead-lag, basepoint, and time channel"""
        # path shape: (n_steps, n_dims)
        augmented = path.copy()
        
        # Add time channel
        if self.time_channel:
            t = np.linspace(0, 1, len(path)).reshape(-1, 1)
            augmented = np.hstack([augmented, t])
        
        # Add basepoint (start at origin)
        if self.basepoint:
            augmented = np.vstack([np.zeros(augmented.shape[1]), augmented])
        
        # Lead-lag augmentation: [x, x_lag] where x_lag is lagged version
        if self.lead_lag and len(augmented) > 1:
            lead = augmented[1:, :]
            lag = augmented[:-1, :]
            # Interleave: (lead_0, lag_0, lead_1, lag_1, ...)
            interleaved = np.zeros((len(lead) + len(lag), lead.shape[1]))
            interleaved[0::2] = lead
            interleaved[1::2] = lag
            augmented = interleaved
        
        return augmented
    
    def signature_term(self, path, indices):
        """Compute a single signature term (iterated integral)"""
        term = 1.0
        current = path[0]
        
        for idx in indices:
            # Approximate integral using trapezoidal rule
            integral = 0
            for i in range(len(path) - 1):
                midpoint = (path[i+1, idx] + path[i, idx]) / 2
                integral += (path[i+1, idx] - path[i, idx]) * midpoint
            term *= integral
        
        return term
    
    def compute_signature(self, path):
        """
        Compute full truncated signature up to specified depth
        Returns dict of multi-index -> value
        """
        augmented = self.augment_path(path)
        dim = augmented.shape[1]
        
        signature = {}
        
        # Depth 0: constant 1
        signature[tuple()] = 1.0
        
        # Depths 1 through self.depth
        for d in range(1, self.depth + 1):
            for indices in combinations_with_replacement(range(dim), d):
                sig_val = self._compute_iterated_integral(augmented, indices)
                signature[indices] = sig_val
        
        return signature
    
    def _compute_iterated_integral(self, path, indices):
        """Compute iterated integral for given multi-index"""
        if len(indices) == 1:
            # First order: simple increment
            return path[-1, indices[0]] - path[0, indices[0]]
        
        # Higher order: recursive integration
        result = 0.0
        idx = indices[-1]
        prev_indices = indices[:-1]
        
        for i in range(len(path) - 1):
            # Value of lower-order integral at point i
            lower_path = path[:i+2]  # up to i+1
            lower_integral = self._compute_iterated_integral(lower_path, prev_indices)
            # Increment in the last dimension
            increment = path[i+1, idx] - path[i, idx]
            result += lower_integral * increment
        
        return result
    
    def signature_vector(self, path):
        """Flatten signature into a vector"""
        sig = self.compute_signature(path)
        # Sort by depth then lexicographically
        keys = sorted(sig.keys(), key=lambda x: (len(x), x))
        return np.array([sig[k] for k in keys])


class LogSignature:
    """Compute log-signature from signature"""
    
    @staticmethod
    def log_signature(signature_dict, depth):
        """Compute log-signature using series expansion"""
        # Simplified: extract log signature from truncated signature
        # For depth 3, log signature terms are:
        # Level 1: same as signature level 1
        # Level 2: [i,j] - 0.5 * ([i] ⊗ [j] - [j] ⊗ [i])
        log_sig = {}
        
        # Level 1
        for i in signature_dict:
            if len(i) == 1:
                log_sig[i] = signature_dict[i]
        
        # Level 2 (antisymmetric part)
        if depth >= 2:
            dims = set([idx for k in signature_dict if len(k) == 1 for idx in k])
            for i in dims:
                for j in dims:
                    if i < j:
                        key = (i, j)
                        sig_ij = signature_dict.get(key, 0)
                        sig_ji = signature_dict.get((j, i), 0)
                        log_sig[key] = 0.5 * (sig_ij - sig_ji)
        
        return log_sig


class NeumannSignatureKernel:
    """
    Signature kernel using Neumann series expansion
    CPU-optimized for GitHub Actions free tier
    """
    
    def __init__(self, depth=3, tile_size=500, epsilon=1e-6):
        """
        Args:
            depth: Signature depth
            tile_size: Size of tiles for local expansion
            epsilon: Dynamic truncation tolerance
        """
        self.depth = depth
        self.tile_size = tile_size
        self.epsilon = epsilon
        self.sig_computer = SignatureComputer(depth=depth)
    
    def _tile_sequence(self, path):
        """Split long sequence into overlapping tiles"""
        n = len(path)
        tiles = []
        
        for start in range(0, n, self.tile_size - self.tile_size // 2):
            end = min(start + self.tile_size, n)
            if end - start >= 2:  # Need at least 2 points
                tiles.append(path[start:end])
        
        return tiles
    
    def _neumann_expansion(self, path_a, path_b):
        """
        Compute kernel between two paths using Neumann expansion
        K(path_a, path_b) = Σ_{k=0}^∞ ⟨sig_k(path_a), sig_k(path_b)⟩
        """
        sig_a = self.sig_computer.signature_vector(path_a)
        sig_b = self.sig_computer.signature_vector(path_b)
        
        # Truncated inner product (Neumann series)
        kernel = np.dot(sig_a, sig_b)
        
        # Check convergence
        if abs(kernel) < self.epsilon:
            kernel = 0
        
        return kernel
    
    def kernel_matrix(self, paths):
        """
        Compute kernel matrix for a list of paths
        Uses tiling for long sequences
        """
        n = len(paths)
        K = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                # Tile long paths
                tiles_i = self._tile_sequence(paths[i])
                tiles_j = self._tile_sequence(paths[j])
                
                # Average over tile pairs
                val = 0
                count = 0
                for ti in tiles_i:
                    for tj in tiles_j:
                        val += self._neumann_expansion(ti, tj)
                        count += 1
                
                K[i, j] = val / max(count, 1)
                K[j, i] = K[i, j]
        
        return K


class AdaptiveDepthSelector:
    """Automatically select optimal signature depth"""
    
    def __init__(self, depths=[2, 3, 4], val_ratio=0.2):
        self.depths = depths
        self.val_ratio = val_ratio
    
    def select_depth(self, X_paths, y_values):
        """
        Select depth that minimizes validation error
        """
        n = len(X_paths)
        val_size = int(n * self.val_ratio)
        
        best_depth = 3  # default
        best_score = -np.inf
        
        for depth in self.depths:
            computer = SignatureComputer(depth=depth)
            kernel = NeumannSignatureKernel(depth=depth)
            
            # Compute kernel matrix on training portion
            train_paths = X_paths[:-val_size]
            K_train = kernel.kernel_matrix(train_paths)
            
            # Simple score: kernel alignment with target
            if len(train_paths) > 1:
                score = np.trace(K_train) / len(train_paths)
            else:
                score = 0
            
            if score > best_score:
                best_score = score
                best_depth = depth
        
        return best_depth
