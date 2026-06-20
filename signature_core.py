"""
Signature computation core for ROUGH-PATH-FORECASTER
v2.0 - Implements Chen's Iteration, Correct Lead-Lag, Efficient Tensor Math
"""

import numpy as np
from itertools import combinations_with_replacement

class SignatureComputer:
    """Compute truncated path signatures using Chen's Iteration (O(N * |sig| * D))"""
    
    def __init__(self, depth=3, lead_lag=True, basepoint=True, time_channel=True):
        self.depth = depth
        self.lead_lag = lead_lag
        self.basepoint = basepoint
        self.time_channel = time_channel
    
    def augment_path(self, path):
        """
        Correctly augment path: Lead-Lag -> Basepoint -> Time Channel
        path shape: (n_steps, n_dims)
        """
        n, d = path.shape
        augmented = path.copy()
        
        # 1. LEAD-LAG: Preserves quadratic covariation
        # Correct formulation: (X_t, X_t), (X_{t+1}, X_t), (X_{t+1}, X_{t+1})...
        if self.lead_lag and n > 1:
            lead_lag_path = np.zeros((2 * n - 1, 2 * d))
            for i in range(n):
                idx = 2 * i
                lead_lag_path[idx, :d] = path[i]
                lead_lag_path[idx, d:] = path[i]
                if i < n - 1:
                    lead_lag_path[idx + 1, :d] = path[i+1]
                    lead_lag_path[idx + 1, d:] = path[i]
            augmented = lead_lag_path
        
        # 2. BASEPOINT: Insert row of zeros at the beginning
        if self.basepoint:
            zero_row = np.zeros((1, augmented.shape[1]))
            augmented = np.vstack([zero_row, augmented])
        
        # 3. TIME CHANNEL: Scaled to [0, 1] over the FINAL augmented length
        if self.time_channel:
            final_n = augmented.shape[0]
            t = np.linspace(0, 1, final_n).reshape(-1, 1)
            augmented = np.column_stack([augmented, t])
            
        return augmented
    
    def compute_signature(self, path):
        """
        Compute full truncated signature using Chen's Iteration.
        This is O(N * |sig| * D) instead of the old O(N^D) recursive crash loop.
        """
        augmented = self.augment_path(path)
        dim = augmented.shape[1]
        
        # Initialize signature dictionary with empty tuple (level 0)
        sig = {tuple(): 1.0}
        
        # Chen's Iteration: incrementally build signature path step by path step
        for i in range(1, len(augmented)):
            dx = augmented[i] - augmented[i-1]
            
            # Create a copy to update simultaneously
            new_sig = sig.copy()
            
            for key, val in sig.items():
                # Only multiply if we haven't exceeded max depth
                if len(key) < self.depth:
                    for d in range(dim):
                        new_key = key + (d,)
                        new_sig[new_key] = new_sig.get(new_key, 0.0) + val * dx[d]
            
            sig = new_sig
            
        return sig
    
    def signature_vector(self, path):
        """Flatten signature into a vector, sorted by depth then lexicographically"""
        sig = self.compute_signature(path)
        keys = sorted(sig.keys(), key=lambda x: (len(x), x))
        return np.array([sig[k] for k in keys])


class LogSignature:
    """Compute log-signature from signature (Level 1 and Level 2 brackets)"""
    
    @staticmethod
    def log_signature(signature_dict, depth):
        """Compute log-signature using series expansion"""
        log_sig = {}
        
        # Level 1: same as signature level 1
        for key, value in signature_dict.items():
            if len(key) == 1:
                log_sig[key] = value
        
        # Level 2 (antisymmetric part / Lie bracket)
        if depth >= 2:
            dims = sorted(list(set([idx for k in signature_dict if len(k) == 1 for idx in k])))
            for i in dims:
                for j in dims:
                    if i < j:
                        key_ij = (i, j)
                        key_ji = (j, i)
                        sig_ij = signature_dict.get(key_ij, 0)
                        sig_ji = signature_dict.get(key_ji, 0)
                        log_sig[key_ij] = 0.5 * (sig_ij - sig_ji)
                        
        return log_sig


class SignatureKernel:
    """
    Standard Linear Signature Kernel.
    Replaced fake Neumann tiling with mathematically sound inner product.
    """
    
    def __init__(self, depth=3):
        self.depth = depth
        self.sig_computer = SignatureComputer(depth=depth)
    
    def kernel_matrix(self, paths):
        """Compute kernel matrix for a list of paths"""
        n = len(paths)
        # Pre-compute all signature vectors
        sig_vectors = [self.sig_computer.signature_vector(p) for p in paths]
        
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                # Standard linear kernel on the signature space
                val = np.dot(sig_vectors[i], sig_vectors[j])
                K[i, j] = val
                K[j, i] = val
                
        return K
    
    def kernel_vector(self, path, reference_paths):
        """Compute kernel between a path and a list of reference paths"""
        path_sig = self.sig_computer.signature_vector(path)
        result = np.zeros(len(reference_paths))
        for i, ref_path in enumerate(reference_paths):
            ref_sig = self.sig_computer.signature_vector(ref_path)
            result[i] = np.dot(path_sig, ref_sig)
        return result


class AdaptiveDepthSelector:
    """Automatically select optimal signature depth based on target variance"""
    
    def __init__(self, depths=[2, 3, 4], val_ratio=0.2):
        self.depths = depths
        self.val_ratio = val_ratio
    
    def select_depth(self, X_paths, y_values):
        """
        Select depth that maximizes correlation with the target.
        (Old version just picked the one with the largest L2 norm, which was mathematically meaningless).
        """
        n = len(X_paths)
        if n < 5:
            return 3 # Default fallback
            
        best_depth = 3
        best_score = -np.inf
        
        y = np.array(y_values)
        
        for depth in self.depths:
            try:
                computer = SignatureComputer(depth=depth)
                sig_vectors = np.array([computer.signature_vector(p) for p in X_paths])
                
                # Normalize signatures to prevent scale bias
                sig_norms = np.linalg.norm(sig_vectors, axis=1, keepdims=True)
                sig_norms[sig_norms == 0] = 1
                sig_vectors = sig_vectors / sig_norms
                
                # Calculate simple linear correlation between signature PCs and target
                # (Fast proxy for kernel regression performance)
                cov_matrix = np.cov(sig_vectors.T, y)
                score = np.sum(np.abs(cov_matrix[:-1, -1])) # Sum of absolute covariances
                
                if score > best_score:
                    best_score = score
                    best_depth = depth
            except Exception:
                continue
                
        return best_depth
