"""
Utilities for ROUGH-PATH-FORECASTER
Logging, caching, parallel processing, GitHub Actions helpers
"""

import logging
import time
import os
import pickle
import hashlib
from functools import wraps
from datetime import datetime
import numpy as np


class Logger:
    """Simple logging utility"""
    
    def __init__(self, name="ROUGH-PATH-FORECASTER", level=logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def info(self, msg):
        self.logger.info(msg)
    
    def warning(self, msg):
        self.logger.warning(msg)
    
    def error(self, msg):
        self.logger.error(msg)
    
    def debug(self, msg):
        self.logger.debug(msg)


class Timer:
    """Context manager for timing operations"""
    
    def __init__(self):
        self.start = None
        self.end = None
        self.seconds = 0.0
        self.minutes = 0.0
    
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        self.seconds = self.end - self.start
        self.minutes = self.seconds / 60
        # Don't suppress exceptions - let them propagate
        return False
    
    def reset(self):
        """Reset the timer"""
        self.start = None
        self.end = None
        self.seconds = 0.0
        self.minutes = 0.0
    
    def elapsed_seconds(self):
        """Get elapsed seconds without exiting context"""
        if self.start is None:
            return 0.0
        return time.time() - self.start
    
    def elapsed_minutes(self):
        """Get elapsed minutes without exiting context"""
        return self.elapsed_seconds() / 60


class CacheManager:
    """Manage caching of expensive computations"""
    
    def __init__(self, cache_dir='data/cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_hash(self, key):
        """Generate hash from key"""
        if isinstance(key, str):
            return hashlib.md5(key.encode()).hexdigest()
        else:
            return hashlib.md5(pickle.dumps(key)).hexdigest()
    
    def get(self, key):
        """Get cached item"""
        hash_key = self._get_hash(key)
        cache_file = os.path.join(self.cache_dir, f"{hash_key}.pkl")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                return None
        return None
    
    def set(self, key, value):
        """Cache an item"""
        hash_key = self._get_hash(key)
        cache_file = os.path.join(self.cache_dir, f"{hash_key}.pkl")
        
        with open(cache_file, 'wb') as f:
            pickle.dump(value, f)
        
        return True
    
    def clear(self, older_than_days=None):
        """Clear cache"""
        if older_than_days:
            cutoff = time.time() - (older_than_days * 86400)
            for filename in os.listdir(self.cache_dir):
                filepath = os.path.join(self.cache_dir, filename)
                if os.path.getmtime(filepath) < cutoff:
                    os.remove(filepath)
        else:
            for filename in os.listdir(self.cache_dir):
                filepath = os.path.join(self.cache_dir, filename)
                if os.path.isfile(filepath):
                    os.remove(filepath)
    
    def exists(self, key):
        """Check if key exists in cache"""
        hash_key = self._get_hash(key)
        cache_file = os.path.join(self.cache_dir, f"{hash_key}.pkl")
        return os.path.exists(cache_file)


class ParallelProcessor:
    """Simple parallel processing for CPU-bound tasks"""
    
    def __init__(self, n_jobs=-1):
        import multiprocessing
        if n_jobs == -1:
            self.n_jobs = multiprocessing.cpu_count()
        else:
            self.n_jobs = n_jobs
    
    def map(self, func, items):
        """Parallel map"""
        from concurrent.futures import ProcessPoolExecutor
        
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            results = list(executor.map(func, items))
        
        return results
    
    def starmap(self, func, items):
        """Parallel starmap for multiple arguments"""
        from concurrent.futures import ProcessPoolExecutor
        
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            results = list(executor.map(lambda args: func(*args), items))
        
        return results


class MemoryOptimizer:
    """Memory optimization utilities"""
    
    @staticmethod
    def reduce_memory_usage(df):
        """Reduce DataFrame memory usage by downcasting"""
        import pandas as pd
        import numpy as np
        
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type != 'object':
                c_min = df[col].min()
                c_max = df[col].max()
                
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                else:
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
        
        return df
    
    @staticmethod
    def chunk_data(data, chunk_size):
        """Split data into chunks"""
        for i in range(0, len(data), chunk_size):
            yield data[i:i + chunk_size]
    
    @staticmethod
    def get_memory_usage_mb(obj):
        """Get memory usage of object in MB"""
        import sys
        return sys.getsizeof(obj) / (1024 * 1024)


class GitHubActionsHelpers:
    """Helpers for GitHub Actions environment"""
    
    @staticmethod
    def is_github_actions():
        """Check if running in GitHub Actions"""
        return os.environ.get('GITHUB_ACTIONS') == 'true'
    
    @staticmethod
    def get_runner_info():
        """Get GitHub Actions runner info"""
        return {
            'is_actions': GitHubActionsHelpers.is_github_actions(),
            'runner_os': os.environ.get('RUNNER_OS', 'unknown'),
            'runner_arch': os.environ.get('RUNNER_ARCH', 'unknown'),
            'github_run_id': os.environ.get('GITHUB_RUN_ID', 'local'),
            'github_run_number': os.environ.get('GITHUB_RUN_NUMBER', '0')
        }
    
    @staticmethod
    def set_output(name, value):
        """Set GitHub Actions output"""
        if GitHubActionsHelpers.is_github_actions():
            github_output = os.environ.get('GITHUB_OUTPUT')
            if github_output:
                with open(github_output, 'a') as f:
                    f.write(f"{name}={value}\n")
            else:
                print(f"::set-output name={name}::{value}")
    
    @staticmethod
    def set_failed(message):
        """Mark job as failed with message"""
        print(f"::error::{message}")
        exit(1)
    
    @staticmethod
    def group(name):
        """Create log group in GitHub Actions"""
        if GitHubActionsHelpers.is_github_actions():
            print(f"::group::{name}")
        return name
    
    @staticmethod
    def end_group():
        """End log group"""
        if GitHubActionsHelpers.is_github_actions():
            print("::endgroup::")
    
    @staticmethod
    def get_memory_limit_mb():
        """Get memory limit in MB (GitHub Actions free tier: ~7GB)"""
        try:
            import psutil
            return psutil.virtual_memory().available // (1024 * 1024)
        except ImportError:
            # Fallback - assume 7GB
            return 7 * 1024
    
    @staticmethod
    def save_cache(key, data, cache_dir='data/cache'):
        """Save data to GitHub Actions cache"""
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"{key}.pkl")
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        return cache_file
    
    @staticmethod
    def load_cache(key, cache_dir='data/cache'):
        """Load data from GitHub Actions cache"""
        cache_file = os.path.join(cache_dir, f"{key}.pkl")
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None
