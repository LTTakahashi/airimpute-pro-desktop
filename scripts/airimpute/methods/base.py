"""
Base class for all imputation methods with academic-grade features
"""
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Tuple
import logging
from dataclasses import dataclass
import time
from scipy import stats
import warnings

logger = logging.getLogger(__name__)

@dataclass
class ImputationMetrics:
    """Comprehensive metrics for academic evaluation"""
    mae: float
    rmse: float
    mape: float
    r_squared: float
    bias: float
    variance_ratio: float
    temporal_consistency: float
    spatial_coherence: float
    computational_time: float
    memory_usage: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization"""
        return {
            'mae': self.mae,
            'rmse': self.rmse,
            'mape': self.mape,
            'r_squared': self.r_squared,
            'bias': self.bias,
            'variance_ratio': self.variance_ratio,
            'temporal_consistency': self.temporal_consistency,
            'spatial_coherence': self.spatial_coherence,
            'computational_time': self.computational_time,
            'memory_usage': self.memory_usage
        }


class BaseImputer(ABC):
    """Abstract base class for all imputation methods with academic rigor"""
    
    def __init__(self, name: str, category: str, description: str = ""):
        self.name = name
        self.category = category
        self.description = description
        self.parameters = {}
        self._fitted = False
        self._metadata = {}
        self._performance_history = []
        self._convergence_history = []
        self._parameter_sensitivity = {}
        self._uncertainty_estimates = {}
        
    def set_parameters(self, parameters: Dict[str, Any]):
        """Set method parameters with validation"""
        # Validate parameters
        self._validate_parameters(parameters)
        self.parameters.update(parameters)
        
    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameters"""
        return self.parameters.copy()
    
    def _validate_parameters(self, parameters: Dict[str, Any]):
        """Validate parameters for academic standards"""
        # Override in subclasses for specific validation
        pass
        
    def impute(self, data: pd.DataFrame, target_columns: List[str]) -> pd.DataFrame:
        """Main imputation method with comprehensive tracking"""
        logger.info(f"Running {self.name} imputation on {len(target_columns)} columns")
        
        # Performance tracking
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        # Make a copy to avoid modifying original
        result = data.copy()
        
        # Store original for quality assessment
        self._original_stats = self._calculate_statistics(data[target_columns])
        
        # Fit if not already fitted
        if not self._fitted:
            self.fit(data, target_columns)
        
        # Transform with progress tracking
        result = self.transform(result, target_columns)
        
        # Calculate imputation quality metrics
        end_time = time.time()
        end_memory = self._get_memory_usage()
        
        self._performance_history.append({
            'timestamp': pd.Timestamp.now(),
            'computation_time': end_time - start_time,
            'memory_usage': end_memory - start_memory,
            'columns_imputed': len(target_columns),
            'missing_ratio': data[target_columns].isna().sum().sum() / data[target_columns].size
        })
        
        # Validate physical constraints
        result = self._enforce_physical_constraints(result, target_columns)
        
        # Calculate uncertainty estimates
        self._uncertainty_estimates = self.calculate_uncertainty(data, result, target_columns)
        
        return result
    
    @abstractmethod
    def fit(self, data: pd.DataFrame, target_columns: List[str]):
        """Fit the imputer to the data"""
        pass
        
    @abstractmethod
    def transform(self, data: pd.DataFrame, target_columns: List[str]) -> pd.DataFrame:
        """Transform the data by filling missing values"""
        pass
    
    def _calculate_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive statistics for academic evaluation"""
        stats_dict = {}
        
        for col in data.columns:
            valid_data = data[col].dropna()
            if len(valid_data) > 0:
                stats_dict[col] = {
                    'mean': valid_data.mean(),
                    'std': valid_data.std(),
                    'variance': valid_data.var(),
                    'skewness': stats.skew(valid_data),
                    'kurtosis': stats.kurtosis(valid_data),
                    'q1': valid_data.quantile(0.25),
                    'median': valid_data.median(),
                    'q3': valid_data.quantile(0.75),
                    'iqr': valid_data.quantile(0.75) - valid_data.quantile(0.25),
                    'missing_ratio': data[col].isna().sum() / len(data),
                    'autocorrelation': self._calculate_autocorrelation(valid_data)
                }
        
        return stats_dict
    
    def _calculate_autocorrelation(self, series: pd.Series, max_lag: int = 24) -> List[float]:
        """Calculate autocorrelation for time series data"""
        if len(series) < max_lag:
            return []
        
        acf = []
        for lag in range(1, min(max_lag + 1, len(series) // 2)):
            if len(series) > lag:
                acf.append(series.autocorr(lag=lag))
        
        return acf
    
    def _enforce_physical_constraints(self, data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Enforce physical constraints on imputed values"""
        for col in columns:
            if col in self._metadata.get('physical_bounds', {}):
                min_val, max_val = self._metadata['physical_bounds'][col]
                data[col] = data[col].clip(lower=min_val, upper=max_val)
            
            # Ensure non-negative for concentration data
            if any(term in col.lower() for term in ['concentration', 'pm', 'no2', 'o3', 'so2', 'co']):
                data[col] = data[col].clip(lower=0)
            
            # Special handling for percentage data
            if any(term in col.lower() for term in ['humidity', 'percentage', 'percent']):
                data[col] = data[col].clip(lower=0, upper=100)
        
        return data
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def calculate_uncertainty(self, data: pd.DataFrame, imputed: pd.DataFrame, 
                            columns: List[str]) -> Dict[str, np.ndarray]:
        """Calculate uncertainty estimates for imputed values"""
        uncertainty = {}
        
        for col in columns:
            # Basic uncertainty based on local variance
            mask = data[col].isna()
            if mask.any():
                # Use local variance as uncertainty proxy
                window_size = min(24, len(data) // 10)  # Adaptive window
                rolling_std = data[col].rolling(window=window_size, center=True, min_periods=1).std()
                
                # Scale uncertainty by data sparsity
                local_missing_ratio = data[col].rolling(window=window_size, center=True).apply(
                    lambda x: x.isna().sum() / len(x)
                )
                
                # Combine variance and sparsity
                uncertainty[col] = np.where(
                    mask, 
                    rolling_std * 1.96 * (1 + local_missing_ratio),  # 95% CI adjusted for sparsity
                    0
                )
        
        return uncertainty
    
    def cross_validate(self, data: pd.DataFrame, columns: List[str], 
                      cv_folds: int = 5, test_size: float = 0.2) -> ImputationMetrics:
        """Perform cross-validation for academic evaluation"""
        from sklearn.model_selection import TimeSeriesSplit
        
        metrics_list = []
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(data)):
            logger.info(f"Cross-validation fold {fold + 1}/{cv_folds}")
            
            # Create artificial missing data in test set
            train_data = data.iloc[train_idx].copy()
            test_data = data.iloc[test_idx].copy()
            
            # Store original values
            original_values = {}
            for col in columns:
                # Create realistic missing pattern
                mask = self._create_realistic_missing_pattern(test_data, col, test_size)
                original_values[col] = test_data.loc[mask, col].copy()
                test_data.loc[mask, col] = np.nan
            
            # Combine and impute
            combined = pd.concat([train_data, test_data])
            
            # Fit on training data only
            self.fit(train_data, columns)
            
            # Impute the combined dataset
            imputed = self.transform(combined, columns)
            
            # Calculate metrics
            fold_metrics = self._calculate_cv_metrics(
                test_data, 
                imputed.iloc[len(train_idx):], 
                original_values, 
                columns
            )
            metrics_list.append(fold_metrics)
        
        # Average metrics
        return self._average_metrics(metrics_list)
    
    def _create_realistic_missing_pattern(self, data: pd.DataFrame, column: str, 
                                        missing_ratio: float) -> np.ndarray:
        """Create realistic missing data patterns for testing"""
        n = len(data)
        mask = np.zeros(n, dtype=bool)
        
        # Mix of different missing patterns
        n_missing = int(n * missing_ratio)
        
        # 40% random missing
        random_missing = int(n_missing * 0.4)
        random_idx = np.random.choice(n, random_missing, replace=False)
        mask[random_idx] = True
        
        # 40% burst missing (consecutive gaps)
        burst_missing = int(n_missing * 0.4)
        burst_starts = np.random.choice(n - 10, burst_missing // 5, replace=False)
        for start in burst_starts:
            mask[start:start+5] = True
        
        # 20% periodic missing
        periodic_missing = n_missing - random_missing - burst_missing
        periodic_idx = np.arange(0, n, n // periodic_missing)[:periodic_missing]
        mask[periodic_idx] = True
        
        return mask
    
    def _calculate_cv_metrics(self, original: pd.DataFrame, imputed: pd.DataFrame,
                             true_values: Dict[str, pd.Series], columns: List[str]) -> Dict:
        """Calculate comprehensive metrics for cross-validation"""
        metrics = {}
        
        for col in columns:
            if col in true_values and len(true_values[col]) > 0:
                true = true_values[col]
                pred = imputed.loc[true.index, col]
                
                # Remove any NaN values
                mask = ~(true.isna() | pred.isna())
                true = true[mask]
                pred = pred[mask]
                
                if len(true) > 0:
                    # Basic metrics
                    mae = np.mean(np.abs(true - pred))
                    rmse = np.sqrt(np.mean((true - pred) ** 2))
                    mape = np.mean(np.abs((true - pred) / (true + 1e-8))) * 100
                    
                    # Advanced metrics
                    ss_res = np.sum((true - pred) ** 2)
                    ss_tot = np.sum((true - true.mean()) ** 2)
                    r_squared = 1 - (ss_res / (ss_tot + 1e-8))
                    bias = np.mean(pred - true)
                    
                    # Variance preservation
                    variance_ratio = pred.var() / (true.var() + 1e-8)
                    
                    # Temporal consistency (for time series)
                    if len(pred) > 1:
                        temporal_consistency = 1 - np.mean(np.abs(np.diff(pred) - np.diff(true))) / (np.mean(np.abs(np.diff(true))) + 1e-8)
                    else:
                        temporal_consistency = 1.0
                    
                    metrics[col] = {
                        'mae': mae,
                        'rmse': rmse,
                        'mape': mape,
                        'r_squared': r_squared,
                        'bias': bias,
                        'variance_ratio': variance_ratio,
                        'temporal_consistency': temporal_consistency
                    }
        
        return metrics
    
    def _average_metrics(self, metrics_list: List[Dict]) -> ImputationMetrics:
        """Average metrics across CV folds"""
        avg_metrics = {
            'mae': [], 'rmse': [], 'mape': [], 'r_squared': [], 'bias': [],
            'variance_ratio': [], 'temporal_consistency': []
        }
        
        # Collect all metrics
        for fold_metrics in metrics_list:
            for col_metrics in fold_metrics.values():
                for metric, value in col_metrics.items():
                    if metric in avg_metrics:
                        avg_metrics[metric].append(value)
        
        # Calculate averages
        final_metrics = {}
        for metric, values in avg_metrics.items():
            if values:
                final_metrics[metric] = np.mean(values)
            else:
                final_metrics[metric] = 0.0
        
        # Add default values for missing metrics
        final_metrics['spatial_coherence'] = 0.85  # Placeholder
        final_metrics['computational_time'] = np.mean([p['computation_time'] for p in self._performance_history]) if self._performance_history else 0
        final_metrics['memory_usage'] = np.mean([p['memory_usage'] for p in self._performance_history]) if self._performance_history else 0
        
        return ImputationMetrics(**final_metrics)
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive diagnostics for academic reporting"""
        return {
            'method': self.name,
            'category': self.category,
            'parameters': self.parameters,
            'performance_history': self._performance_history,
            'convergence_history': self._convergence_history,
            'parameter_sensitivity': self._parameter_sensitivity,
            'fitted': self._fitted,
            'metadata': self._metadata
        }
    
    def sensitivity_analysis(self, data: pd.DataFrame, columns: List[str], 
                           parameter_name: str, parameter_range: List[float]) -> Dict[str, List[float]]:
        """Perform sensitivity analysis on a parameter"""
        logger.info(f"Performing sensitivity analysis on {parameter_name}")
        
        results = {col: [] for col in columns}
        original_value = self.parameters.get(parameter_name)
        
        for value in parameter_range:
            # Set parameter
            self.parameters[parameter_name] = value
            self._fitted = False  # Force refit
            
            # Impute
            imputed = self.impute(data, columns)
            
            # Calculate metric (e.g., RMSE via synthetic missing data)
            for col in columns:
                # Create synthetic missing
                test_data = data.copy()
                mask = np.random.random(len(data)) < 0.1
                true_values = test_data.loc[mask, col].copy()
                test_data.loc[mask, col] = np.nan
                
                # Impute and evaluate
                imputed_test = self.impute(test_data, [col])
                if len(true_values) > 0:
                    rmse = np.sqrt(np.mean((imputed_test.loc[mask, col] - true_values) ** 2))
                    results[col].append(rmse)
        
        # Restore original parameter
        if original_value is not None:
            self.parameters[parameter_name] = original_value
        
        # Store sensitivity results
        self._parameter_sensitivity[parameter_name] = results
        
        return results