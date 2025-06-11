"""
Comprehensive benchmarking framework for air quality imputation methods.

This module provides:
- Standard benchmark datasets and data loaders
- Performance metrics and statistical testing
- Reproducibility features
- Result tracking and comparison
- GPU acceleration support
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
import time
import json
import hashlib
from pathlib import Path
import warnings
from datetime import datetime
import sqlite3
import pickle
import gzip
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
from functools import partial
import logging
from scipy import stats
import sklearn.metrics as metrics
from sklearn.model_selection import TimeSeriesSplit, KFold
import h5py
import zarr
import dask.array as da
from numba import jit, cuda
import psutil
import platform
import subprocess
import uuid
import git
import sys
import os

# Optional imports for GPU acceleration
try:
    import cupy as cp  # GPU acceleration
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    cp = None

try:
    import pyopencl as cl  # OpenCL support
    OPENCL_AVAILABLE = True
except ImportError:
    OPENCL_AVAILABLE = False
    cl = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ReproducibilityInfo:
    """Tracks reproducibility information for benchmarks."""
    benchmark_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    git_commit: Optional[str] = None
    git_dirty: bool = False
    python_version: str = field(default_factory=lambda: sys.version)
    platform_info: Dict[str, str] = field(default_factory=dict)
    package_versions: Dict[str, str] = field(default_factory=dict)
    random_seeds: Dict[str, int] = field(default_factory=dict)
    environment_variables: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize platform and package information."""
        # Platform info
        self.platform_info = {
            'system': platform.system(),
            'node': platform.node(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor()
        }
        
        # Git info
        try:
            repo = git.Repo(search_parent_directories=True)
            self.git_commit = repo.head.object.hexsha
            self.git_dirty = repo.is_dirty()
        except:
            pass
        
        # Package versions
        import importlib
        packages = ['numpy', 'pandas', 'scikit-learn', 'torch', 'tensorflow']
        for pkg in packages:
            try:
                mod = importlib.import_module(pkg)
                self.package_versions[pkg] = getattr(mod, '__version__', 'unknown')
            except:
                self.package_versions[pkg] = 'not installed'
        
        # Environment variables
        relevant_vars = ['CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'PYTHONHASHSEED']
        for var in relevant_vars:
            if var in os.environ:
                self.environment_variables[var] = os.environ[var]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'benchmark_id': self.benchmark_id,
            'timestamp': self.timestamp.isoformat(),
            'git_commit': self.git_commit,
            'git_dirty': self.git_dirty,
            'python_version': self.python_version,
            'platform_info': self.platform_info,
            'package_versions': self.package_versions,
            'random_seeds': self.random_seeds,
            'environment_variables': self.environment_variables
        }
    
    def generate_certificate_hash(self) -> str:
        """Generate unique hash for reproducibility certificate."""
        data_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()


@dataclass
class BenchmarkDataset:
    """Standard benchmark dataset for imputation evaluation."""
    name: str
    data: pd.DataFrame
    metadata: Dict[str, Any]
    missing_patterns: Dict[str, np.ndarray]
    true_values: Optional[pd.DataFrame] = None
    description: str = ""
    citation: str = ""
    
    def __post_init__(self):
        """Validate dataset structure."""
        if self.data.empty:
            raise ValueError("Dataset cannot be empty")
        
        # Ensure datetime index
        if not isinstance(self.data.index, pd.DatetimeIndex):
            raise ValueError("Dataset must have DatetimeIndex")
        
        # Generate dataset hash for reproducibility
        data_str = self.data.to_json()
        self.hash = hashlib.sha256(data_str.encode()).hexdigest()[:16]


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    method_name: str
    dataset_name: str
    metrics: Dict[str, float]
    runtime: float
    memory_usage: float
    parameters: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    hardware_info: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'method_name': self.method_name,
            'dataset_name': self.dataset_name,
            'metrics': self.metrics,
            'runtime': self.runtime,
            'memory_usage': self.memory_usage,
            'parameters': self.parameters,
            'timestamp': self.timestamp.isoformat(),
            'hardware_info': self.hardware_info
        }


class BenchmarkDatasetManager:
    """Manages standard benchmark datasets for air quality imputation."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path.home() / '.airimpute' / 'benchmarks'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.datasets: Dict[str, BenchmarkDataset] = {}
        
    def create_synthetic_dataset(
        self,
        name: str,
        n_timesteps: int = 8760,  # One year hourly
        n_stations: int = 20,
        freq: str = 'H',
        missing_rate: float = 0.2,
        pattern: str = 'random',
        seasonal_components: int = 3,
        noise_level: float = 0.1,
        seed: int = 42
    ) -> BenchmarkDataset:
        """Create synthetic dataset with known ground truth."""
        np.random.seed(seed)
        
        # Generate time index
        time_index = pd.date_range(
            start='2023-01-01',
            periods=n_timesteps,
            freq=freq
        )
        
        # Generate spatial coordinates
        station_coords = np.random.uniform(-50, 50, size=(n_stations, 2))
        station_names = [f'Station_{i:03d}' for i in range(n_stations)]
        
        # Generate true signals with spatial and temporal correlation
        true_data = self._generate_spatiotemporal_data(
            n_timesteps, n_stations, station_coords,
            seasonal_components, noise_level
        )
        
        # Create DataFrame
        df = pd.DataFrame(
            true_data,
            index=time_index,
            columns=station_names
        )
        
        # Store complete data
        true_values = df.copy()
        
        # Generate missing patterns
        missing_patterns = self._generate_missing_patterns(
            df.shape, missing_rate, pattern, seed
        )
        
        # Apply missing patterns
        df_missing = df.copy()
        for pattern_name, mask in missing_patterns.items():
            df_missing[mask] = np.nan
        
        # Create metadata
        metadata = {
            'n_timesteps': n_timesteps,
            'n_stations': n_stations,
            'frequency': freq,
            'missing_rate': missing_rate,
            'pattern_type': pattern,
            'seasonal_components': seasonal_components,
            'noise_level': noise_level,
            'station_coordinates': station_coords,
            'generation_seed': seed
        }
        
        dataset = BenchmarkDataset(
            name=name,
            data=df_missing,
            metadata=metadata,
            missing_patterns=missing_patterns,
            true_values=true_values,
            description=f"Synthetic dataset with {pattern} missing pattern",
            citation="Generated by AirImpute benchmarking framework"
        )
        
        self.datasets[name] = dataset
        self._cache_dataset(dataset)
        
        return dataset
    
    def _generate_spatiotemporal_data(
        self,
        n_timesteps: int,
        n_stations: int,
        coords: np.ndarray,
        n_components: int,
        noise_level: float
    ) -> np.ndarray:
        """Generate realistic spatiotemporal data."""
        # Time array
        t = np.arange(n_timesteps)
        
        # Initialize data
        data = np.zeros((n_timesteps, n_stations))
        
        # Add seasonal components
        for k in range(n_components):
            # Temporal pattern
            period = n_timesteps / (k + 1)
            temporal = np.sin(2 * np.pi * t / period)
            
            # Spatial pattern (based on distance)
            spatial_center = np.random.uniform(-50, 50, size=2)
            distances = np.linalg.norm(coords - spatial_center, axis=1)
            spatial = np.exp(-distances / (20 * (k + 1)))
            
            # Combine
            data += np.outer(temporal, spatial) * (10 / (k + 1))
        
        # Add trend
        trend = np.linspace(0, 5, n_timesteps)
        data += trend[:, np.newaxis]
        
        # Add spatially correlated noise
        noise = self._generate_correlated_noise(
            n_timesteps, n_stations, coords, noise_level
        )
        data += noise
        
        # Ensure positive values (concentration data)
        data = np.maximum(data, 0)
        
        return data
    
    def _generate_correlated_noise(
        self,
        n_timesteps: int,
        n_stations: int,
        coords: np.ndarray,
        noise_level: float
    ) -> np.ndarray:
        """Generate spatially correlated noise."""
        # Compute spatial correlation matrix
        distances = np.linalg.norm(
            coords[:, np.newaxis, :] - coords[np.newaxis, :, :],
            axis=2
        )
        correlation = np.exp(-distances / 20)  # Correlation length scale
        
        # Generate correlated noise
        noise = np.random.multivariate_normal(
            mean=np.zeros(n_stations),
            cov=correlation * noise_level,
            size=n_timesteps
        )
        
        return noise
    
    def _generate_missing_patterns(
        self,
        shape: Tuple[int, int],
        missing_rate: float,
        pattern: str,
        seed: int
    ) -> Dict[str, np.ndarray]:
        """Generate various missing data patterns."""
        np.random.seed(seed)
        patterns = {}
        
        # Random missing
        if pattern in ['random', 'all']:
            mask = np.random.random(shape) < missing_rate
            patterns['random'] = mask
        
        # Block missing (consecutive time periods)
        if pattern in ['block', 'all']:
            mask = np.zeros(shape, dtype=bool)
            n_blocks = int(shape[0] * missing_rate / 24)  # 24-hour blocks
            for _ in range(n_blocks):
                start = np.random.randint(0, shape[0] - 24)
                station = np.random.randint(0, shape[1])
                mask[start:start+24, station] = True
            patterns['block'] = mask
        
        # Station failure (entire stations missing for periods)
        if pattern in ['station', 'all']:
            mask = np.zeros(shape, dtype=bool)
            n_failures = int(shape[1] * missing_rate)
            failed_stations = np.random.choice(shape[1], n_failures, replace=False)
            for station in failed_stations:
                start = np.random.randint(0, shape[0] // 2)
                end = np.random.randint(shape[0] // 2, shape[0])
                mask[start:end, station] = True
            patterns['station'] = mask
        
        # Systematic missing (regular intervals)
        if pattern in ['systematic', 'all']:
            mask = np.zeros(shape, dtype=bool)
            interval = int(1 / missing_rate)
            mask[::interval, :] = True
            patterns['systematic'] = mask
        
        return patterns
    
    def load_real_dataset(
        self,
        name: str,
        data_path: Path,
        format: str = 'csv',
        **kwargs
    ) -> BenchmarkDataset:
        """Load real-world dataset for benchmarking."""
        # Load data based on format
        if format == 'csv':
            df = pd.read_csv(data_path, **kwargs)
        elif format == 'parquet':
            df = pd.read_parquet(data_path, **kwargs)
        elif format == 'hdf5':
            df = pd.read_hdf(data_path, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Process and validate
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
        
        # Calculate missing patterns
        missing_mask = df.isna().values
        missing_patterns = {'original': missing_mask}
        
        # Create metadata
        metadata = {
            'source_file': str(data_path),
            'shape': df.shape,
            'missing_rate': missing_mask.mean(),
            'time_range': [df.index.min(), df.index.max()],
            'variables': list(df.columns)
        }
        
        dataset = BenchmarkDataset(
            name=name,
            data=df,
            metadata=metadata,
            missing_patterns=missing_patterns,
            description=f"Real dataset from {data_path.name}"
        )
        
        self.datasets[name] = dataset
        self._cache_dataset(dataset)
        
        return dataset
    
    def _cache_dataset(self, dataset: BenchmarkDataset):
        """Cache dataset to disk for faster loading."""
        cache_path = self.cache_dir / f"{dataset.name}_{dataset.hash}.pkl.gz"
        with gzip.open(cache_path, 'wb') as f:
            pickle.dump(dataset, f)
    
    def list_datasets(self) -> List[str]:
        """List available datasets."""
        return list(self.datasets.keys())
    
    def get_dataset(self, name: str) -> BenchmarkDataset:
        """Get dataset by name."""
        if name not in self.datasets:
            # Try loading from cache
            cache_files = list(self.cache_dir.glob(f"{name}_*.pkl.gz"))
            if cache_files:
                with gzip.open(cache_files[0], 'rb') as f:
                    dataset = pickle.load(f)
                    self.datasets[name] = dataset
            else:
                raise ValueError(f"Dataset '{name}' not found")
        
        return self.datasets[name]


class PerformanceMetrics:
    """Comprehensive performance metrics for imputation evaluation."""
    
    @staticmethod
    def calculate_all_metrics(
        true_values: np.ndarray,
        imputed_values: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate all standard metrics."""
        if mask is None:
            mask = ~np.isnan(true_values)
        
        # Flatten and filter
        true_flat = true_values[mask]
        imputed_flat = imputed_values[mask]
        
        metrics_dict = {}
        
        # Error metrics
        metrics_dict['mae'] = np.mean(np.abs(true_flat - imputed_flat))
        metrics_dict['rmse'] = np.sqrt(np.mean((true_flat - imputed_flat) ** 2))
        metrics_dict['mape'] = np.mean(np.abs((true_flat - imputed_flat) / (true_flat + 1e-8))) * 100
        
        # Correlation metrics
        metrics_dict['pearson_r'], metrics_dict['pearson_p'] = stats.pearsonr(true_flat, imputed_flat)
        metrics_dict['spearman_r'], metrics_dict['spearman_p'] = stats.spearmanr(true_flat, imputed_flat)
        
        # Regression metrics
        metrics_dict['r2'] = metrics.r2_score(true_flat, imputed_flat)
        metrics_dict['explained_variance'] = metrics.explained_variance_score(true_flat, imputed_flat)
        
        # Distribution metrics
        metrics_dict['ks_statistic'], metrics_dict['ks_pvalue'] = stats.ks_2samp(true_flat, imputed_flat)
        
        # Bias metrics
        metrics_dict['mean_bias'] = np.mean(imputed_flat - true_flat)
        metrics_dict['median_bias'] = np.median(imputed_flat - true_flat)
        
        # Percentile errors
        for p in [10, 25, 50, 75, 90]:
            true_p = np.percentile(true_flat, p)
            imputed_p = np.percentile(imputed_flat, p)
            metrics_dict[f'percentile_{p}_error'] = abs(imputed_p - true_p)
        
        return metrics_dict
    
    @staticmethod
    def calculate_temporal_metrics(
        true_series: pd.Series,
        imputed_series: pd.Series,
        lag_range: range = range(1, 25)
    ) -> Dict[str, float]:
        """Calculate time series specific metrics."""
        metrics_dict = {}
        
        # Autocorrelation preservation
        true_acf = [true_series.autocorr(lag) for lag in lag_range]
        imputed_acf = [imputed_series.autocorr(lag) for lag in lag_range]
        metrics_dict['acf_mae'] = np.mean(np.abs(np.array(true_acf) - np.array(imputed_acf)))
        
        # Spectral analysis
        true_fft = np.abs(np.fft.fft(true_series.values))[:len(true_series)//2]
        imputed_fft = np.abs(np.fft.fft(imputed_series.values))[:len(imputed_series)//2]
        metrics_dict['spectral_rmse'] = np.sqrt(np.mean((true_fft - imputed_fft) ** 2))
        
        return metrics_dict
    
    @staticmethod
    def calculate_spatial_metrics(
        true_matrix: np.ndarray,
        imputed_matrix: np.ndarray,
        coordinates: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate spatial correlation metrics."""
        metrics_dict = {}
        
        # Spatial correlation preservation
        true_corr = np.corrcoef(true_matrix.T)
        imputed_corr = np.corrcoef(imputed_matrix.T)
        
        metrics_dict['spatial_corr_mae'] = np.mean(np.abs(true_corr - imputed_corr))
        metrics_dict['spatial_corr_rmse'] = np.sqrt(np.mean((true_corr - imputed_corr) ** 2))
        
        # Semivariogram analysis if coordinates provided
        if coordinates is not None:
            true_semivar = PerformanceMetrics._calculate_semivariogram(true_matrix, coordinates)
            imputed_semivar = PerformanceMetrics._calculate_semivariogram(imputed_matrix, coordinates)
            metrics_dict['semivariogram_rmse'] = np.sqrt(np.mean((true_semivar - imputed_semivar) ** 2))
        
        return metrics_dict
    
    @staticmethod
    def _calculate_semivariogram(data: np.ndarray, coords: np.ndarray, n_bins: int = 20) -> np.ndarray:
        """Calculate empirical semivariogram."""
        n_stations = data.shape[1]
        
        # Calculate pairwise distances
        distances = []
        semivariances = []
        
        for i in range(n_stations):
            for j in range(i + 1, n_stations):
                dist = np.linalg.norm(coords[i] - coords[j])
                diff = data[:, i] - data[:, j]
                semivar = 0.5 * np.mean(diff ** 2)
                
                distances.append(dist)
                semivariances.append(semivar)
        
        # Bin by distance
        distances = np.array(distances)
        semivariances = np.array(semivariances)
        
        bins = np.linspace(0, distances.max(), n_bins + 1)
        binned_semivar = np.zeros(n_bins)
        
        for i in range(n_bins):
            mask = (distances >= bins[i]) & (distances < bins[i + 1])
            if mask.sum() > 0:
                binned_semivar[i] = semivariances[mask].mean()
        
        return binned_semivar


class StatisticalTesting:
    """Statistical testing framework for comparing imputation methods."""
    
    @staticmethod
    def friedman_test(results: Dict[str, List[float]], alpha: float = 0.05) -> Dict[str, Any]:
        """Friedman test for comparing multiple methods."""
        # Prepare data
        methods = list(results.keys())
        scores = np.array([results[method] for method in methods]).T
        
        # Perform test
        statistic, pvalue = stats.friedmanchisquare(*scores.T)
        
        # Post-hoc analysis if significant
        post_hoc = None
        if pvalue < alpha:
            post_hoc = StatisticalTesting._nemenyi_test(scores, methods)
        
        return {
            'statistic': statistic,
            'pvalue': pvalue,
            'significant': pvalue < alpha,
            'post_hoc': post_hoc,
            'method_ranks': StatisticalTesting._calculate_ranks(scores, methods)
        }
    
    @staticmethod
    def _calculate_ranks(scores: np.ndarray, methods: List[str]) -> Dict[str, float]:
        """Calculate average ranks for methods."""
        ranks = stats.rankdata(-scores, axis=1)  # Negative for ascending order
        avg_ranks = ranks.mean(axis=0)
        
        return {method: rank for method, rank in zip(methods, avg_ranks)}
    
    @staticmethod
    def _nemenyi_test(scores: np.ndarray, methods: List[str]) -> Dict[str, Dict[str, bool]]:
        """Nemenyi post-hoc test."""
        n_datasets, n_methods = scores.shape
        ranks = stats.rankdata(-scores, axis=1)
        avg_ranks = ranks.mean(axis=0)
        
        # Critical difference
        q_alpha = 2.569  # For alpha=0.05, k=5 methods
        cd = q_alpha * np.sqrt(n_methods * (n_methods + 1) / (6 * n_datasets))
        
        # Pairwise comparisons
        results = {}
        for i, method_i in enumerate(methods):
            results[method_i] = {}
            for j, method_j in enumerate(methods):
                if i != j:
                    diff = abs(avg_ranks[i] - avg_ranks[j])
                    results[method_i][method_j] = diff > cd
        
        return results
    
    @staticmethod
    def bootstrap_confidence_intervals(
        metric_func: Callable,
        true_values: np.ndarray,
        imputed_values: np.ndarray,
        n_bootstrap: int = 1000,
        confidence: float = 0.95
    ) -> Dict[str, float]:
        """Calculate bootstrap confidence intervals for metrics."""
        n_samples = len(true_values)
        bootstrap_metrics = []
        
        for _ in range(n_bootstrap):
            # Sample with replacement
            indices = np.random.choice(n_samples, n_samples, replace=True)
            metric = metric_func(true_values[indices], imputed_values[indices])
            bootstrap_metrics.append(metric)
        
        # Calculate percentiles
        alpha = (1 - confidence) / 2
        lower = np.percentile(bootstrap_metrics, alpha * 100)
        upper = np.percentile(bootstrap_metrics, (1 - alpha) * 100)
        
        return {
            'mean': np.mean(bootstrap_metrics),
            'std': np.std(bootstrap_metrics),
            'lower': lower,
            'upper': upper,
            'confidence': confidence
        }


class BenchmarkRunner:
    """Main benchmark execution engine with GPU support and reproducibility tracking."""
    
    def __init__(
        self,
        dataset_manager: BenchmarkDatasetManager,
        results_dir: Optional[Path] = None,
        use_gpu: bool = True,
        n_jobs: int = -1,
        reproducibility_info: Optional[ReproducibilityInfo] = None,
        random_seed: int = 42
    ):
        self.dataset_manager = dataset_manager
        self.results_dir = results_dir or Path.home() / '.airimpute' / 'results'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.use_gpu = use_gpu and self._check_gpu_available()
        self.n_jobs = n_jobs if n_jobs > 0 else mp.cpu_count()
        self.random_seed = random_seed
        
        # Initialize reproducibility tracking
        self.reproducibility_info = reproducibility_info or ReproducibilityInfo()
        self.reproducibility_info.random_seeds['main'] = random_seed
        self._set_random_seeds(random_seed)
        
        # Initialize results database
        self.db_path = self.results_dir / 'benchmark_results.db'
        self._init_database()
        
        # Hardware info
        self.hardware_info = self._get_hardware_info()
        
        # Create run directory for this benchmark
        self.run_id = self.reproducibility_info.benchmark_id
        self.run_dir = self.results_dir / f'run_{self.run_id[:8]}'
        self.run_dir.mkdir(exist_ok=True)
        
        # Save reproducibility info
        self._save_reproducibility_info()
    
    def _set_random_seeds(self, seed: int):
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        import random
        random.seed(seed)
        
        # Set environment variable for hash randomization
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        # PyTorch
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        except ImportError:
            pass
        
        # TensorFlow
        try:
            import tensorflow as tf
            tf.random.set_seed(seed)
        except ImportError:
            pass
    
    def _save_reproducibility_info(self):
        """Save reproducibility information to file."""
        info_path = self.run_dir / 'reproducibility_info.json'
        with open(info_path, 'w') as f:
            json.dump(self.reproducibility_info.to_dict(), f, indent=2)
        
        # Also save as YAML for better readability
        try:
            import yaml
            yaml_path = self.run_dir / 'reproducibility_info.yaml'
            with open(yaml_path, 'w') as f:
                yaml.dump(self.reproducibility_info.to_dict(), f, default_flow_style=False)
        except ImportError:
            pass
    
    def _check_gpu_available(self) -> bool:
        """Check if GPU acceleration is available."""
        if CUDA_AVAILABLE:
            try:
                cp.cuda.Device(0).compute_capability
                return True
            except:
                pass
        
        if OPENCL_AVAILABLE:
            try:
                platforms = cl.get_platforms()
                return len(platforms) > 0
            except:
                pass
        
        return False
    
    def _get_hardware_info(self) -> Dict[str, str]:
        """Get system hardware information."""
        info = {
            'cpu_count': str(mp.cpu_count()),
            'cpu_name': 'Unknown',
            'memory_gb': str(psutil.virtual_memory().total // (1024**3)),
            'gpu_available': str(self.use_gpu)
        }
        
        if self.use_gpu and CUDA_AVAILABLE:
            try:
                device = cp.cuda.Device(0)
                info['gpu_name'] = device.name.decode()
                info['gpu_memory_gb'] = str(device.mem_info[1] // (1024**3))
                info['cuda_version'] = cp.cuda.runtime.getVersion()
            except:
                pass
        
        return info
    
    def _init_database(self):
        """Initialize results database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS benchmark_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                method_name TEXT,
                dataset_name TEXT,
                metrics TEXT,
                runtime REAL,
                memory_usage REAL,
                parameters TEXT,
                timestamp TEXT,
                hardware_info TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def run_benchmark(
        self,
        methods: Dict[str, Callable],
        datasets: List[str],
        cv_splits: int = 5,
        save_predictions: bool = False,
        parallel: bool = True
    ) -> pd.DataFrame:
        """Run comprehensive benchmark across methods and datasets."""
        results = []
        
        for dataset_name in datasets:
            logger.info(f"Loading dataset: {dataset_name}")
            dataset = self.dataset_manager.get_dataset(dataset_name)
            
            if parallel and len(methods) > 1:
                # Parallel execution
                with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                    futures = {}
                    for method_name, method_func in methods.items():
                        future = executor.submit(
                            self._evaluate_method,
                            method_name, method_func, dataset, cv_splits, save_predictions
                        )
                        futures[future] = method_name
                    
                    for future in futures:
                        try:
                            result = future.result(timeout=3600)  # 1 hour timeout
                            results.append(result)
                        except Exception as e:
                            logger.error(f"Error in {futures[future]}: {str(e)}")
            else:
                # Sequential execution
                for method_name, method_func in methods.items():
                    try:
                        result = self._evaluate_method(
                            method_name, method_func, dataset, cv_splits, save_predictions
                        )
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Error in {method_name}: {str(e)}")
        
        # Convert to DataFrame
        results_df = pd.DataFrame([r.to_dict() for r in results])
        
        # Save results
        self._save_results(results_df)
        
        return results_df
    
    def _evaluate_method(
        self,
        method_name: str,
        method_func: Callable,
        dataset: BenchmarkDataset,
        cv_splits: int,
        save_predictions: bool
    ) -> BenchmarkResult:
        """Evaluate single method on dataset."""
        logger.info(f"Evaluating {method_name} on {dataset.name}")
        
        # Prepare data
        data = dataset.data.values
        mask = ~np.isnan(data)
        
        # Cross-validation setup
        if dataset.metadata.get('frequency') == 'H':
            cv = TimeSeriesSplit(n_splits=cv_splits)
        else:
            cv = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
        
        # Metrics storage
        cv_metrics = []
        runtimes = []
        memory_usage = []
        
        # GPU data preparation if applicable
        if self.use_gpu and hasattr(method_func, 'gpu_enabled'):
            data_gpu = cp.asarray(data)
            mask_gpu = cp.asarray(mask)
        
        # Cross-validation
        for fold, (train_idx, test_idx) in enumerate(cv.split(data)):
            # Create train/test split
            train_data = data.copy()
            test_mask = np.zeros_like(mask, dtype=bool)
            test_mask[test_idx] = True
            
            # Additional masking for evaluation
            eval_mask = mask & test_mask
            train_data[eval_mask] = np.nan
            
            # Measure performance
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # Run imputation
            if self.use_gpu and hasattr(method_func, 'gpu_enabled'):
                imputed = method_func(cp.asarray(train_data))
                imputed = cp.asnumpy(imputed)
            else:
                imputed = method_func(train_data)
            
            runtime = time.time() - start_time
            memory = psutil.Process().memory_info().rss / 1024 / 1024 - start_memory
            
            # Calculate metrics
            if dataset.true_values is not None:
                true_vals = dataset.true_values.values
                metrics = PerformanceMetrics.calculate_all_metrics(
                    true_vals[eval_mask],
                    imputed[eval_mask]
                )
            else:
                metrics = self._calculate_reconstruction_metrics(
                    data[eval_mask],
                    imputed[eval_mask]
                )
            
            cv_metrics.append(metrics)
            runtimes.append(runtime)
            memory_usage.append(memory)
            
            # Save predictions if requested
            if save_predictions:
                self._save_predictions(
                    dataset.name, method_name, fold,
                    imputed, eval_mask
                )
        
        # Aggregate metrics
        aggregated_metrics = {}
        for metric in cv_metrics[0].keys():
            values = [m[metric] for m in cv_metrics]
            aggregated_metrics[f'{metric}_mean'] = np.mean(values)
            aggregated_metrics[f'{metric}_std'] = np.std(values)
        
        # Create result
        result = BenchmarkResult(
            method_name=method_name,
            dataset_name=dataset.name,
            metrics=aggregated_metrics,
            runtime=np.mean(runtimes),
            memory_usage=np.mean(memory_usage),
            parameters=self._extract_method_params(method_func),
            hardware_info=self.hardware_info
        )
        
        # Store in database
        self._store_result(result)
        
        return result
    
    def _calculate_reconstruction_metrics(
        self,
        original: np.ndarray,
        imputed: np.ndarray
    ) -> Dict[str, float]:
        """Calculate metrics for reconstruction quality."""
        return {
            'reconstruction_error': np.mean(np.abs(original - imputed)),
            'correlation': np.corrcoef(original, imputed)[0, 1]
        }
    
    def _extract_method_params(self, method_func: Callable) -> Dict[str, Any]:
        """Extract method parameters if available."""
        if hasattr(method_func, 'get_params'):
            return method_func.get_params()
        elif hasattr(method_func, '__self__') and hasattr(method_func.__self__, 'get_params'):
            return method_func.__self__.get_params()
        else:
            return {}
    
    def _save_predictions(
        self,
        dataset_name: str,
        method_name: str,
        fold: int,
        predictions: np.ndarray,
        mask: np.ndarray
    ):
        """Save imputation predictions."""
        save_path = self.results_dir / 'predictions' / dataset_name / method_name
        save_path.mkdir(parents=True, exist_ok=True)
        
        np.savez_compressed(
            save_path / f'fold_{fold}.npz',
            predictions=predictions,
            mask=mask
        )
    
    def _store_result(self, result: BenchmarkResult):
        """Store result in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO benchmark_results 
            (method_name, dataset_name, metrics, runtime, memory_usage, 
             parameters, timestamp, hardware_info)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            result.method_name,
            result.dataset_name,
            json.dumps(result.metrics),
            result.runtime,
            result.memory_usage,
            json.dumps(result.parameters),
            result.timestamp.isoformat(),
            json.dumps(result.hardware_info)
        ))
        
        conn.commit()
        conn.close()
    
    def _save_results(self, results_df: pd.DataFrame):
        """Save results to various formats."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # CSV
        results_df.to_csv(
            self.results_dir / f'benchmark_results_{timestamp}.csv',
            index=False
        )
        
        # JSON
        results_df.to_json(
            self.results_dir / f'benchmark_results_{timestamp}.json',
            orient='records',
            indent=2
        )
        
        # LaTeX table
        self._generate_latex_table(results_df, timestamp)
    
    def _generate_latex_table(self, results_df: pd.DataFrame, timestamp: str):
        """Generate LaTeX table for academic papers."""
        # Pivot table for better presentation
        pivot = results_df.pivot_table(
            index='method_name',
            columns='dataset_name',
            values=['rmse_mean', 'mae_mean', 'r2_mean'],
            aggfunc='mean'
        )
        
        latex_table = pivot.to_latex(
            float_format='%.4f',
            bold_rows=True,
            caption='Benchmark results for air quality imputation methods',
            label='tab:benchmark_results'
        )
        
        with open(self.results_dir / f'benchmark_table_{timestamp}.tex', 'w') as f:
            f.write(latex_table)
    
    def compare_methods(
        self,
        results_df: pd.DataFrame,
        metric: str = 'rmse_mean',
        statistical_test: bool = True
    ) -> Dict[str, Any]:
        """Compare methods with statistical testing."""
        comparison = {}
        
        # Aggregate by method
        method_scores = {}
        for method in results_df['method_name'].unique():
            scores = results_df[results_df['method_name'] == method][metric].values
            method_scores[method] = scores.tolist()
        
        # Ranking
        mean_scores = {m: np.mean(s) for m, s in method_scores.items()}
        comparison['ranking'] = sorted(mean_scores.items(), key=lambda x: x[1])
        
        # Statistical testing
        if statistical_test and len(method_scores) > 2:
            comparison['friedman_test'] = StatisticalTesting.friedman_test(method_scores)
        
        return comparison


# GPU-accelerated imputation kernels
@cuda.jit
def cuda_linear_interpolation_kernel(data, mask, output):
    """CUDA kernel for linear interpolation."""
    i, j = cuda.grid(2)
    
    if i < data.shape[0] and j < data.shape[1]:
        if not mask[i, j]:
            # Find nearest valid neighbors
            left_idx = i
            right_idx = i
            
            # Search left
            while left_idx > 0 and not mask[left_idx - 1, j]:
                left_idx -= 1
            if left_idx > 0:
                left_idx -= 1
            
            # Search right
            while right_idx < data.shape[0] - 1 and not mask[right_idx + 1, j]:
                right_idx += 1
            if right_idx < data.shape[0] - 1:
                right_idx += 1
            
            # Interpolate
            if mask[left_idx, j] and mask[right_idx, j]:
                if left_idx == right_idx:
                    output[i, j] = data[left_idx, j]
                else:
                    weight = (i - left_idx) / (right_idx - left_idx)
                    output[i, j] = data[left_idx, j] * (1 - weight) + data[right_idx, j] * weight
            elif mask[left_idx, j]:
                output[i, j] = data[left_idx, j]
            elif mask[right_idx, j]:
                output[i, j] = data[right_idx, j]
            else:
                output[i, j] = 0.0
        else:
            output[i, j] = data[i, j]


class GPUAcceleratedMethods:
    """GPU-accelerated imputation methods for benchmarking."""
    
    def __init__(self, backend: str = 'cuda'):
        self.backend = backend
        self.gpu_enabled = True
        
        if backend == 'cuda':
            self._init_cuda()
        elif backend == 'opencl':
            self._init_opencl()
    
    def _init_cuda(self):
        """Initialize CUDA backend."""
        if not CUDA_AVAILABLE:
            raise ImportError("CuPy not available for CUDA acceleration")
        self.cp = cp
        self.device = cp.cuda.Device(0)
    
    def _init_opencl(self):
        """Initialize OpenCL backend."""
        if not OPENCL_AVAILABLE:
            raise ImportError("PyOpenCL not available for OpenCL acceleration")
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)
    
    def gpu_linear_interpolation(self, data: np.ndarray) -> np.ndarray:
        """GPU-accelerated linear interpolation."""
        if self.backend == 'cuda':
            # Transfer to GPU
            data_gpu = cp.asarray(data)
            mask_gpu = ~cp.isnan(data_gpu)
            output_gpu = cp.zeros_like(data_gpu)
            
            # Configure kernel
            threads_per_block = (16, 16)
            blocks_per_grid = (
                (data.shape[0] + threads_per_block[0] - 1) // threads_per_block[0],
                (data.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
            )
            
            # Launch kernel
            cuda_linear_interpolation_kernel[blocks_per_grid, threads_per_block](
                data_gpu, mask_gpu, output_gpu
            )
            
            # Transfer back
            return cp.asnumpy(output_gpu)
        
        else:  # OpenCL
            # OpenCL implementation
            kernel_code = """
            __kernel void linear_interpolation(
                __global const float* data,
                __global const int* mask,
                __global float* output,
                const int rows,
                const int cols
            ) {
                int i = get_global_id(0);
                int j = get_global_id(1);
                
                if (i < rows && j < cols) {
                    int idx = i * cols + j;
                    
                    if (mask[idx]) {
                        output[idx] = data[idx];
                    } else {
                        // Interpolation logic
                        float sum = 0.0f;
                        int count = 0;
                        
                        // Check neighbors
                        if (i > 0 && mask[(i-1)*cols + j]) {
                            sum += data[(i-1)*cols + j];
                            count++;
                        }
                        if (i < rows-1 && mask[(i+1)*cols + j]) {
                            sum += data[(i+1)*cols + j];
                            count++;
                        }
                        
                        output[idx] = count > 0 ? sum / count : 0.0f;
                    }
                }
            }
            """
            
            # Build and execute
            prg = cl.Program(self.ctx, kernel_code).build()
            
            data_flat = data.astype(np.float32).flatten()
            mask_flat = (~np.isnan(data)).astype(np.int32).flatten()
            output = np.zeros_like(data_flat)
            
            mf = cl.mem_flags
            data_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data_flat)
            mask_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=mask_flat)
            output_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, output.nbytes)
            
            prg.linear_interpolation(
                self.queue, data.shape, None,
                data_buf, mask_buf, output_buf,
                np.int32(data.shape[0]), np.int32(data.shape[1])
            )
            
            cl.enqueue_copy(self.queue, output, output_buf)
            
            return output.reshape(data.shape)


# Convenience function for running benchmarks
def run_comprehensive_benchmark(
    methods: Dict[str, Callable],
    n_synthetic_datasets: int = 5,
    real_datasets: Optional[List[Path]] = None,
    use_gpu: bool = True,
    output_dir: Optional[Path] = None
) -> pd.DataFrame:
    """Run comprehensive benchmark with default settings."""
    
    # Initialize components
    dataset_manager = BenchmarkDatasetManager()
    
    # Create synthetic datasets
    synthetic_names = []
    for i in range(n_synthetic_datasets):
        name = f'synthetic_{i}'
        dataset_manager.create_synthetic_dataset(
            name=name,
            n_timesteps=8760,  # One year
            n_stations=20,
            missing_rate=0.2 + 0.1 * i,  # Varying missing rates
            pattern='all',  # All patterns
            seed=42 + i
        )
        synthetic_names.append(name)
    
    # Load real datasets if provided
    real_names = []
    if real_datasets:
        for i, path in enumerate(real_datasets):
            name = f'real_{path.stem}'
            dataset_manager.load_real_dataset(name, path)
            real_names.append(name)
    
    # Initialize runner
    runner = BenchmarkRunner(
        dataset_manager=dataset_manager,
        results_dir=output_dir,
        use_gpu=use_gpu
    )
    
    # Run benchmarks
    all_datasets = synthetic_names + real_names
    results_df = runner.run_benchmark(
        methods=methods,
        datasets=all_datasets,
        cv_splits=5,
        save_predictions=True,
        parallel=True
    )
    
    # Generate comparison report
    comparison = runner.compare_methods(results_df)
    
    # Print summary
    print("\n" + "="*80)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*80)
    print(f"\nMethods evaluated: {list(methods.keys())}")
    print(f"Datasets: {len(all_datasets)} ({len(synthetic_names)} synthetic, {len(real_names)} real)")
    print(f"\nTop performers (by RMSE):")
    for i, (method, score) in enumerate(comparison['ranking'][:3]):
        print(f"{i+1}. {method}: {score:.4f}")
    
    if 'friedman_test' in comparison:
        print(f"\nStatistical significance (Friedman test):")
        print(f"p-value: {comparison['friedman_test']['pvalue']:.4f}")
        if comparison['friedman_test']['significant']:
            print("Significant differences found between methods")
    
    return results_df


# Example usage
if __name__ == "__main__":
    # Example methods to benchmark
    from .methods.simple import MeanImputation, ForwardFillImputation
    from .methods.interpolation import LinearInterpolation, SplineInterpolation
    from .methods.machine_learning import RandomForestImputation, XGBoostImputation
    
    methods = {
        'mean': MeanImputation().impute,
        'forward_fill': ForwardFillImputation().impute,
        'linear': LinearInterpolation().impute,
        'spline': SplineInterpolation().impute,
        'random_forest': RandomForestImputation().impute,
        'xgboost': XGBoostImputation().impute
    }
    
    # Run benchmark
    results = run_comprehensive_benchmark(
        methods=methods,
        n_synthetic_datasets=3,
        use_gpu=True
    )
    
    print("\nBenchmark completed successfully!")